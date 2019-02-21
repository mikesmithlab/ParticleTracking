import os
from tqdm import tqdm
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from ParticleTracking import dataframes


from . import order, voronoi_cells, polygon_distances, correlations, level


class PropertyCalculator:

    def __init__(self, datastore):
        self.td = datastore
        self.td.fill_frame_data()
        self.core_name = os.path.splitext(self.td.filename)[0]

    def order(self):
        orders_complex = np.array([])
        orders_abs = np.array([])
        no_of_neighbors = np.array([])
        frame_order = []
        frame_sus = []
        for n in tqdm(range(self.td.num_frames), 'Order'):
            points = self.td.get_info(n, ['x', 'y'])
            orders, neighbors = order.order_and_neighbors(points)

            orders_r = np.abs(orders)

            orders_complex = np.append(orders_complex, orders)
            orders_abs = np.append(orders_abs, orders_r)

            no_of_neighbors = np.append(no_of_neighbors, neighbors)

            frame_order.append(np.mean(orders_r))
            frame_sus.append(np.var(orders_r))

        self.td.add_particle_property('complex order', orders_complex)
        self.td.add_particle_property('real order', orders_abs)
        self.td.add_particle_property('neighbors', no_of_neighbors)
        self.td.add_frame_property('mean order', frame_order)
        self.td.add_frame_property('susceptibility', frame_sus)

    def density(self):

        densities = np.array([])
        shape_factor = np.array([])
        edges = np.array([])
        density_mean = []
        for n in tqdm(range(self.td.num_frames), 'Density'):
            particles = self.td.get_info(n, ['x', 'y', 'size'])
            boundary = self.td.get_boundary(n)
            area, sf, edge = voronoi_cells.calculate(
                particles, boundary)
            density = (particles[:, :2].mean()**2 * pi) / area
            densities = np.append(densities, density)
            shape_factor = np.append(shape_factor, sf)
            edges = np.append(edges, edge)
            density_mean.append(np.mean(density))

        self.td.add_particle_property('local density', densities)
        self.td.add_particle_property('shape factor', shape_factor)
        self.td.add_particle_property('on edge', edges)
        self.td.add_frame_property('local density', density_mean)

    def distance(self):
        boundary = self.td.get_boundary(0)
        x = self.td.get_column('x')
        y = self.td.get_column('y')
        points = np.vstack((x, y)).transpose()
        distance = polygon_distances.to_points(boundary, points)
        self.td.add_particle_property('Edge Distance', distance)

    def correlations(self, frame_no, r_min=1, r_max=10, dr=0.02):
        data = self.td.get_info(
            frame_no, ['x', 'y', 'size', 'complex order', 'Edge Distance'])
        boundary = self.td.get_boundary(frame_no)

        r, g, g6 = correlations.corr(data, boundary, r_min, r_max, dr)
        plt.figure()
        plt.plot(r, g)
        plt.show()

        corr_data = dataframes.CorrData(self.corename)
        corr_data.add_row(r, frame_no, 'r')
        corr_data.add_row(g, frame_no, 'g')
        corr_data.add_row(g6, frame_no, 'g6')

    def check_level(self):
        x = self.td.get_column('x')
        y = self.td.get_column('y')
        points = np.vstack((x, y)).transpose()
        boundary = self.td.get_boundary(0)
        level.check_level(points, boundary)