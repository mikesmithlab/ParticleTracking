import os

import dask.dataframe as dd
import numpy as np
from dask.diagnostics import ProgressBar

from ParticleTracking.statistics import order, voronoi_cells, \
    correlations, level, edge_distance


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

    # def correlations(self, frame_no, r_min=1, r_max=10, dr=0.02):
    #     data = self.data.get_info(
    #         frame_no, ['x', 'y', 'r', 'complex order', 'Edge Distance'])
    #     boundary = self.data.get_boundary(frame_no)
    #
    #     r, g, g6 = correlations.corr(data, boundary, r_min, r_max, dr)
    #     plt.figure()
    #     plt.plot(r, g)
    #     plt.show()
    #
    #     corr_data = dataframes.CorrData(self.core_name)
    #     corr_data.add_row(r, frame_no, 'r')
    #     corr_data.add_row(g, frame_no, 'g')
    #     corr_data.add_row(g6, frame_no, 'g6')

    def check_level(self):
        x = self.data.get_column('x')
        y = self.data.get_column('y')
        points = np.vstack((x, y)).transpose()
        boundary = self.data.get_boundary(0)
        level.check_level(points, boundary)


def flatten(arr):
    arr = list(arr)
    arr = [a for sublist in arr for a in sublist]
    return arr


if __name__ == "__main__":
    from Generic import filedialogs
    from ParticleTracking import dataframes, statistics
    import time
    file = filedialogs.load_filename()
    data = dataframes.DataStore(file, load=True)
    calc = statistics.PropertyCalculator(data)
    t = time.time()
    calc.distance()
    print(data.df.head())
    print(time.time() - t)

