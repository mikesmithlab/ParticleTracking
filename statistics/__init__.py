import os

import dask.dataframe as dd
import numpy as np
from dask.diagnostics import ProgressBar

from ParticleTracking.statistics import order, voronoi_cells, \
    correlations, level, edge_distance, histograms


class PropertyCalculator:

    def __init__(self, datastore):
        self.data = datastore
        self.core_name = os.path.splitext(self.data.filename)[0]

    def order(self):
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

    def duty(self):
        return self.data.df.groupby('frame').first()['Duty']

    def check_level(self):
        x = self.data.get_column('x')
        y = self.data.get_column('y')
        points = np.vstack((x, y)).transpose()
        boundary = self.data.get_boundary(0)
        level.check_level(points, boundary)

    def histogram(self, frames, column, bins):
        counts, bins = histograms.histogram(self.data.df, frames, column,
                                            bins=bins)
        return counts, bins


if __name__ == "__main__":
    from Generic import filedialogs
    from ParticleTracking import dataframes, statistics
    import time
    file = filedialogs.load_filename()
    data = dataframes.DataStore(file, load=True)
    calc = statistics.PropertyCalculator(data)
    t = time.time()
    calc.correlations(10)
    print(data.df.head())
    print(time.time() - t)

