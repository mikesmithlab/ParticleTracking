import multiprocessing as mp
import os
from itertools import repeat, starmap

import numpy as np
from tqdm import tqdm

from ParticleTracking.statistics import order, voronoi_cells, \
    polygon_distances, correlations, level

"""
Multiprocessing will only work on linux systems.

Also need to make the following changes to multiprocessing.connection.py module:
https://github.com/python/cpython/commit/bccacd19fa7b56dcf2fbfab15992b6b94ab6666b
"""


class PropertyCalculator:

    def __init__(self, datastore):
        self.data = datastore
        self.core_name = os.path.splitext(self.data.filename)[0]

    def order(self, multiprocessing=False, overwrite=False):
        if ('real order' in self.data.get_headings()) and (overwrite is False):
            return 0
        rad = self.data.get_column('r').mean() * 4
        # points = self.data.get_info_all_frames(['x', 'y'])
        points = self.data.get_info_all_frames_generator(['x', 'y'])
        print('points got')
        if multiprocessing:
            p = mp.Pool(4)
            orders_r, orders_i, orders_mag, neighbors, mean, sus = zip(
                *p.starmap(order.order_and_neighbors,
                           tqdm(zip(points, repeat(rad)),
                                'Order', total=self.data.num_frames)))
            p.close()
            p.join()
        else:
            orders_r, orders_i, orders_mag, neighbors, mean, sus = zip(
                *starmap(order.order_and_neighbors,
                         tqdm(zip(points, repeat(rad)),
                              'Order', total=self.data.num_frames)))
        orders_r = np.float32(flatten(orders_r))
        orders_i = np.float32(flatten(orders_i))
        orders_mag = np.float32(flatten(orders_mag))
        neighbors = np.uint8(flatten(neighbors))
        mean = np.float32(mean)
        sus = np.float32(sus)

        self.data.add_particle_property('order_r', orders_r)
        self.data.add_particle_property('order_i', orders_i)
        self.data.add_particle_property('order_mag', orders_mag)
        self.data.add_particle_property('neighbors', neighbors)
        self.data.add_frame_property('order_mean', mean)
        self.data.add_frame_property('order_sus', sus)

        self.data.save()

    def density(self, multiprocess=False):
        points = self.data.get_info_all_frames(['x', 'y', 'r'])
        boundary = self.data.get_metadata('boundary')
        if multiprocess:
            p = mp.Pool(4)
            densities, shape_factor, edges, density_mean = zip(
                *p.starmap(voronoi_cells.density,
                           tqdm(zip(points, repeat(boundary)),
                                'Density',
                                total=len(points))))
            p.close()
            p.join()
        else:
            densities, shape_factor, edges, density_mean = zip(
                *starmap(voronoi_cells.density,
                     tqdm(zip(points, repeat(boundary)),
                          'Density',
                          total=len(points))))
        densities = np.float32(flatten(densities))
        shape_factor = np.float32(flatten(shape_factor))
        edges = flatten(edges)

        self.data.add_particle_property('density', densities)
        self.data.add_particle_property('shape_factor', shape_factor)
        self.data.add_particle_property('on_edge', edges)
        self.data.add_metadata('density', np.mean(density_mean))

        self.data.save()

    def distance(self, multiprocess=False):
        points = self.data.get_column(['x', 'y'])

        if multiprocess:
            n = len(points)
            points_list = [points[:n//4, :], points[n//4:2*n//4, :],
                           points[2*n//4:3*n//4, :], points[3*n//4:, :]]
            with mp.Pool(4) as p:
                distance = p.map(self.distance_process, points_list)
            distance = flatten(distance)
        else:
            distance = self.distance_process(points)
        distance = np.float32(distance)
        self.data.add_particle_property('edge_distance', distance)

        self.data.save()

    def distance_process(self, points):
        return polygon_distances.to_points(self.data.get_metadata('boundary'), points)

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
    file = filedialogs.load_filename()
    data = dataframes.DataStore(file, load=True)
    calc = statistics.PropertyCalculator(data)
    calc.order(multiprocessing=True, overwrite=True)
    # calc.density()
    # calc.distance()
    print(data.df.head())
    # print(data.df.dtypes)

