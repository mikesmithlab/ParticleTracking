import os

import dask.dataframe as dd
import numpy as np
import trackpy as tp
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from ParticleTracking.statistics import order, voronoi_cells, \
    correlations, level, edge_distance, histograms, duty

tqdm.pandas()


class PropertyCalculator:

    def __init__(self, datastore):
        self.data = datastore
        self.core_name = os.path.splitext(self.data.filename)[0]

    def link(self, search_range=15, memory=3):
        self.data.df = tp.link(self.data.df.reset_index(), search_range,
                               memory=memory).set_index('frame')
        self.data.save()

    def count(self):
        """
        Calculates average number of detected particles in a dataframes.

        Adds result in 'number_of_particles' key in metadata dictionary.
        """
        n = self.data.df.x.groupby('frame').count()
        n_mean = n.mean()
        self.data.metadata['number_of_particles'] = round(n_mean)

    def duty_cycle(self):
        """
        Calculates the duty cycle from the audio channel of the video.

        Saves the result in the Duty column of the dataframe.
        """
        vid_name = self.data.metadata['video_filename']
        num_frames = self.data.metadata['number_of_frames']
        duty_cycle_vals = duty.duty(vid_name, num_frames)
        diff = duty_cycle_vals[-1] = duty_cycle_vals[0]
        if abs(diff) < 200:
            direc = 'both'
        elif diff > 200:
            direc = 'up'
        else:
            direc = 'down'
        self.data.metadata['direc'] = direc
        self.data.add_frame_property('Duty', duty_cycle_vals)

    def order(self):
        """
        Calculates the order parameter and number of neighbours.

        Saves results in 'order_r', 'order_i' and 'neighbors' columns
        in the dataframe.
        """
        dask_data = dd.from_pandas(self.data.df, chunksize=10000)
        meta = dask_data._meta.copy()
        meta['order_r'] = np.array([], dtype='float32')
        meta['order_i'] = np.array([], dtype='float32')
        meta['neighbors'] = np.array([], dtype='uint8')
        with ProgressBar():
            self.data.df = (dask_data.groupby('frame')
                            .apply(order.order_process, meta=meta)
                            .compute(scheduler='processes'))
        self.data.df['order'] = np.abs(
            self.data.df.order_r + 1j * self.data.df.order_i)
        self.data.save()

    def order_mean(self):
        dask_data = dd.from_pandas(self.data.df, chunksize=10000)
        meta = dask_data._meta.copy()
        meta['order_r_mean'] = np.array([], dtype='float32')
        meta['order_i_mean'] = np.array([], dtype='float32')
        meta['neighbors_mean'] = np.array([], dtype='uint8')
        with ProgressBar():
            self.data.df = (dask_data.groupby('frame')
                            .apply(order.order_process_mean, meta=meta)
                            .compute(scheduler='processes'))
        self.data.df['order_mean'] = np.abs(
            self.data.df.order_r_mean + 1j * self.data.df.order_i_mean)
        self.data.save()

    def density(self):
        """
        Calculates the density, shape_factor for each particle.
        Also checks whether particle is on the edge of the cell or not.

        Saves result in 'density', 'shape_factor' and 'on_edge' columns
        in the dataframe.
        """
        dask_data = dd.from_pandas(self.data.df, chunksize=10000)
        meta = dask_data._meta.copy()
        meta['density'] = np.array([], dtype='float32')
        meta['shape_factor'] = np.array([], dtype='float32')
        meta['on_edge'] = np.array([], dtype='bool')
        with ProgressBar():
            self.data.df = (dask_data.groupby('frame')
                            .apply(voronoi_cells.density,
                                   meta=meta,
                                   boundary=self.data.metadata['boundary'])
                            .compute(scheduler='processes'))
        self.data.save()

    def distance(self):
        """
        Calculates distance of each particle from the edge of the cell.
        """
        self.data.df['edge_distance'] = edge_distance.distance(
            self.data.df[['x', 'y']].values, self.data.metadata['boundary'])

    def correlations(self, frame_no, r_min=1, r_max=20, dr=0.02):
        """
        Calculates the positional and orientational correlations for a given
        frame.

        Parameters
        ----------
        frame_no: int
        r_min: minimum radius
        r_max: maximum radius
        dr: bin width

        Returns
        -------
        r: radius values in pixels
        g: positional correlations
        g6: orientational correlations
        """
        boundary = self.data.metadata['boundary']

        r, g, g6 = correlations.corr(self.data.df.loc[frame_no],
                                     boundary,
                                     r_min,
                                     r_max,
                                     dr)
        return r, g, g6

    def correlations_all_duties(self, r_min=1, r_max=20, dr=0.02):
        """
        Calculates the positional and orientational correlations for each
        duty cycle using all points from all frames with the same duty cycle.

        Returns
        -------
        Dataframe with duty, r, g, g6 columns
        """
        df = self.data.df
        boundary = self.data.metadata['boundary']
        res = df.groupby('Duty').progress_apply(
            correlations.corr_multiple_frames, boundary=boundary, r_min=r_min,
            r_max=r_max, dr=dr)
        return res

    def duty(self):
        """Return the duty cycle of each frame"""
        return self.data.df.groupby('frame').first()['Duty']

    def histogram(self, frames, column, bins):
        """Calculate a histogram for a given property"""
        counts, bins = histograms.histogram(self.data.df, frames, column,
                                            bins=bins)
        return counts, bins

    def order_duty(self):
        self.data.df['order_mag'] = np.abs(
            self.data.df.order_r +
            1j * self.data.df.order_i
        )
        group = self.data.df.groupby('Duty')['order_mag'].mean()
        return group.index.values, group.values

    def density_duty(self):
        group = self.data.df.groupby('Duty')['density'].mean()
        return group.index.values, group.values

    def order_histogram_duties(self):
        """
        Calculates the histogram of the order parameter.

        Uses all the particles that are in each duty cycle.
        """
        duty = np.unique(self.duty())
        bins = []
        freqs = []
        for d in duty:
            orders = self.data.df.loc[
                self.data.df.Duty == d, ['order_r', 'order_i']
            ].values
            order_mag = np.abs(orders[:, 0] + 1j * orders[:, 1])
            n, b = np.histogram(order_mag, bins=100, density=True)
            bins.append(b)
            freqs.append(n)
        return duty, bins, freqs

    def density_histogram_duties(self):
        duty = np.unique(self.duty())
        bins = []
        freqs = []
        for d in duty:
            densities = self.data.df.loc[
                self.data.df.Duty == d,
                'density'
            ].values
            n, b = np.histogram(densities, bins=100, density=True)
            bins.append(b)
            freqs.append(n)
        return duty, bins, freqs

    def rolling_coordinates(self):
        group = self.data.df.groupby('particle')
        rolled = group[['x', 'y']].rolling(window=10, min_periods=1).mean() \
            .rename(columns={'x': 'x_mean', 'y': 'y_mean'})
        self.data.df = self.data.df.merge(rolled, on=['particle', 'frame'])
        self.data.save()


if __name__ == "__main__":
    from ParticleTracking import dataframes, statistics

    # file = filedialogs.load_filename()
    file = "/media/data/Data/August2019/N26_2300_particles_at_600/15890001 _test.hdf5"
    data = dataframes.DataStore(file, load=True)
    calc = statistics.PropertyCalculator(data)
    # duty, bins, freqs = calc.density_histogram_duties()
    # calc.rolling_coordinates()
    # calc.order_mean()
    calc.link()
