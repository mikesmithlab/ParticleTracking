import multiprocessing as mp
import os
import sys
from math import pi

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from numba import jit

from ParticleTracking import dataframes
from . import order, voronoi_cells, polygon_distances, correlations, level

from memory_profiler import profile

# from pathos import multiprocessing as mp

"""
Multiprocessing will only work on linux systems.

Also need to make the following changes to multiprocessing.connection.py module:
https://github.com/python/cpython/commit/bccacd19fa7b56dcf2fbfab15992b6b94ab6666b
"""


class PropertyCalculator:

    def __init__(self, datastore):
        self.td = datastore
        self.td.fill_frame_data()
        self.core_name = os.path.splitext(self.td.filename)[0]
        # self.name = name

    def order(self, multiprocessing=False):
        points = self.td.get_info_all_frames(['x', 'y'])
        if multiprocessing:
            p = mp.Pool(4)
            orders, neighbors, orders_r, mean, sus = zip(
                *p.map(order.order_and_neighbors, tqdm(points)))
            p.close()
            p.join()
        else:
            orders, neighbors, orders_r, mean, sus = zip(
                *map(order.order_and_neighbors, tqdm(points)))
        orders = flatten(orders)
        orders_r = flatten(orders_r)
        neighbors = flatten(neighbors)
        self.td.add_particle_properties(
            ['complex order', 'real order', 'neighbors'],
            [orders, orders_r, neighbors])
        self.td.add_frame_properties(
            ['mean order', 'susceptibility'],
            [mean, sus])

    def density(self, multiprocess=False):
        if not multiprocess:
            densities, shape_factor, edges, density_mean = \
                self.density_process([0, self.td.num_frames])
        else:

            with mp.Pool(4) as p:
                chunk = self.td.num_frames // 4
                starts = [0, chunk, 2 * chunk, 3 * chunk]
                ends = [chunk, 2 * chunk, 3 * chunk, self.td.num_frames]
                densities, shape_factor, edges, density_mean = zip(*p.map(
                    self.density_process, list(zip(starts, ends))))
            densities = flatten(densities)
            shape_factor = flatten(shape_factor)
            edges = flatten(edges)
            density_mean = flatten(density_mean)

        self.td.add_particle_property('local density', densities)
        self.td.add_particle_property('shape factor', shape_factor)
        self.td.add_particle_property('on edge', edges)
        self.td.add_frame_property('local density', density_mean)

    def density_process(self, bounds):
        start, end = bounds
        densities = np.array([])
        shape_factor = np.array([])
        edges = np.array([])
        density_mean = []
        for n in tqdm(range(start, end), 'density'):
            particles = self.td.get_info(n, ['x', 'y', 'size'])
            boundary = self.td.get_boundary(n)
            area, sf, edge = voronoi_cells.calculate(
                particles, boundary)
            density = (particles[:, :2].mean() ** 2 * pi) / area
            densities = np.append(densities, density)
            shape_factor = np.append(shape_factor, sf)
            edges = np.append(edges, edge)
            density_mean.append(np.mean(density))
        return densities, shape_factor, edges, density_mean

    def distance(self, multiprocess=False):
        points = self.td.get_column(['x', 'y'])

        if multiprocess:
            n = len(points)
            points_list = [points[:n//4, :], points[n//4:2*n//4, :],
                           points[2*n//4:3*n//4, :], points[3*n//4:, :]]
            with mp.Pool(4) as p:
                distance = p.map(self.distance_process, points_list)
            distance = flatten(distance)
        else:
            distance = self.distance_process(points)
        self.td.add_particle_property('Edge Distance', distance)

    def distance_process(self, points):
        return polygon_distances.to_points(self.td.get_boundary(0), points)

    def correlations(self, frame_no, r_min=1, r_max=10, dr=0.02):
        data = self.td.get_info(
            frame_no, ['x', 'y', 'r', 'complex order', 'Edge Distance'])
        boundary = self.td.get_boundary(frame_no)

        r, g, g6 = correlations.corr(data, boundary, r_min, r_max, dr)
        plt.figure()
        plt.plot(r, g)
        plt.show()

        corr_data = dataframes.CorrData(self.core_name)
        corr_data.add_row(r, frame_no, 'r')
        corr_data.add_row(g, frame_no, 'g')
        corr_data.add_row(g6, frame_no, 'g6')

    def check_level(self):
        x = self.td.get_column('x')
        y = self.td.get_column('y')
        points = np.vstack((x, y)).transpose()
        boundary = self.td.get_boundary(0)
        level.check_level(points, boundary)


def flatten(arr):
    arr = list(arr)
    arr = [a for sublist in arr for a in sublist]
    return arr
