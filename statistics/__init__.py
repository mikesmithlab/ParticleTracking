import os

import dask.dataframe as dd
import numpy as np
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from ParticleTracking.statistics import order, voronoi_cells, \
    correlations, level, edge_distance, histograms, duty

tqdm.pandas()


class PropertyCalculator:

    def __init__(self, datastore):
        self.data = datastore
        self.core_name = os.path.splitext(self.data.filename)[0]

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


if __name__ == "__main__":
    from ParticleTracking import dataframes, statistics

    file = "/media/data/Data/July2019/RampsN29/15790009.hdf5"
    data = dataframes.DataStore(file, load=True)
    calc = statistics.PropertyCalculator(data)
