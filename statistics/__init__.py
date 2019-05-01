import os
from tqdm import tqdm
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from ParticleTracking import dataframes
import multiprocessing as mp


from . import order, voronoi_cells, polygon_distances, correlations, level


class PropertyCalculator:

    def __init__(self, datastore):
        self.td = datastore
        self.td.fill_frame_data()
        self.core_name = os.path.splitext(self.td.filename)[0]

    def order(self, multiprocess=False):
        if not multiprocess:
            orders_complex, orders_abs, no_of_neighbors, frame_order, frame_sus = \
                self.order_process([0, self.td.num_frames])
        else:
            p = mp.Pool()
            chunk = self.td.num_frames // 4
            starts = [0, chunk, 2 * chunk, 3 * chunk]
            ends = [chunk, 2 * chunk, 3 * chunk, self.td.num_frames]
            orders_complex, orders_abs, no_of_neighbors, frame_order, frame_sus = \
                zip(*p.map(self.order_process, list(zip(starts, ends))))
            orders_complex = self.flatten(orders_complex)
            orders_abs = self.flatten(orders_abs)
            no_of_neighbors = self.flatten(no_of_neighbors)
            frame_order = self.flatten(frame_order)
            frame_sus = self.flatten(frame_sus)
        self.td.add_particle_property('complex order', orders_complex)
        self.td.add_particle_property('real order', orders_abs)
        self.td.add_particle_property('neighbors', no_of_neighbors)
        self.td.add_frame_property('mean order', frame_order)
        self.td.add_frame_property('susceptibility', frame_sus)

    def order_process(self, bounds):
        start, end = bounds
        orders, neighbors = zip(*[
            order.order_and_neighbors(self.td.get_info(n, ['x', 'y']))
            for n in tqdm(range(start, end), 'Order')])
        orders_r = [np.abs(sublist) for sublist in orders]
        frame_order = [np.mean(sublist) for sublist in orders_r]
        frame_sus = [np.var(sublist) for sublist in orders_r]
        orders = self.flatten(orders)
        orders_r = self.flatten(orders_r)
        neighbors = self.flatten(neighbors)
        return orders, orders_r, neighbors, frame_order, frame_sus

    def density(self, multiprocess=False):
        if not multiprocess:
            densities, shape_factor, edges, density_mean = \
                self.density_process([0, self.td.num_frames])
        else:
            p = mp.Pool(4)
            chunk = self.td.num_frames // 4
            print(chunk, self.td.num_frames)
            starts = [0, chunk, 2 * chunk, 3 * chunk]
            ends = [chunk, 2 * chunk, 3 * chunk, self.td.num_frames]
            densities, shape_factor, edges, density_mean = zip(*p.map(
                self.density_process, list(zip(starts, ends))))
            densities = self.flatten(densities)
            shape_factor = self.flatten(shape_factor)
            edges = self.flatten(edges)
            density_mean = self.flatten(density_mean)

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

    @staticmethod
    def flatten(arr):
        arr = list(arr)
        arr = [a for sublist in arr for a in sublist]
        return arr

    def distance(self):
        boundary = self.td.get_boundary(0)
        x = self.td.get_column('x')
        y = self.td.get_column('y')
        points = np.vstack((x, y)).transpose()
        distance = polygon_distances.to_points(boundary, points)
        self.td.add_particle_property('Edge Distance', distance)

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