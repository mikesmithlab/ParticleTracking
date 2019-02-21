import numpy as np
from ParticleTracking import dataframes
from ParticleTracking import graphs
from Generic import plotting
import scipy.spatial as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.path as mpath
from numba import jit
from shapely.geometry import Polygon, Point, MultiPolygon, LineString, MultiPoint
from shapely.geometry.polygon import LinearRing
import os
from math import pi
from tqdm import tqdm
from shapely.strtree import STRtree
import time
from shapely.prepared import prep

class PropertyCalculator:
    """Class to calculate the properties associated with tracking"""
    def __init__(self, datastore):
        self.td = datastore
        self.td.fill_frame_data()
        self.corename = os.path.splitext(self.td.filename)[0]
        plot_data_name = self.corename + '_plot_data.hdf5'
        self.plot_data = dataframes.PlotData(plot_data_name)

    def order_parameter(self):
        """
        Calculate the hexatic order parameter.

        Also stores the number of neighbors, mean order per frame
         and mean susceptibility per frame

        Stores both the complex results and the magnitude of the result.
        """
        order_params = np.array([])
        neighbors_arr = np.array([])
        frame_order = []
        frame_sus = []
        for n in tqdm(range(self.td.num_frames), 'Order'):
            points = self.td.get_info(n, ['x', 'y'])
            list_indices, point_indices = self._find_delaunay_indices(points)
            neighbors = self._count_neighbors(list_indices)
            neighbors_arr = np.append(neighbors_arr, neighbors)
            # The indices of neighbouring vertices of vertex k are
            # point_indices[list_indices[k]:list_indices[k+1]].
            vectors = self._find_vectors(points, list_indices, point_indices)
            angles = self._calculate_angles(vectors)
            orders = self._calculate_orders(angles, list_indices)
            order_params = np.append(order_params, orders)
            orders_r = np.abs(orders)
            frame_order.append(np.mean(np.abs(orders)))
            frame_sus.append(np.mean(np.power(orders_r - np.mean(orders_r), 2)))
        real_order = np.abs(order_params)

        # Add data to dataframe
        self.td.add_particle_property('complex order', order_params)
        self.td.add_particle_property('real order', real_order)
        self.td.add_particle_property('neighbors', neighbors_arr)
        self.td.add_frame_property('mean order', frame_order)
        self.td.add_frame_property('susceptibility', frame_sus)

    def voronoi_cells(self):
        """
        Calculates the density and shape factor of each voronoi cell.
        Also decides whether the particle is at the edge or not.
        """
        boundary = self.td.get_boundary(0)
        boundary = Polygon(boundary)
        local_density_all = np.array([])
        shape_factor_all = np.array([])
        on_edge_all = np.array([])
        for n in tqdm(range(self.td.num_frames), 'Voronoi'):
            info = self.td.get_info(n, ['x', 'y', 'size'])
            particle_area = info[:, 2].mean()**2 * pi
            vor = sp.Voronoi(info[:, :2])
            regions, vertices = voronoi_finite_polygons_2d(vor)
            inside = find_points_inside(vertices, boundary)
            polygons, on_edge = get_polygons(regions, vertices, inside)
            polygons = intersect_all_polygons(polygons, boundary, on_edge)
            # plot_polygons(polygons)
            area, shape_factor = area_and_shapefactor(polygons)
            density = particle_area / np.array(area)
            local_density_all = np.append(local_density_all, density)
            shape_factor_all = np.append(shape_factor_all, shape_factor)
            on_edge_all = np.append(on_edge_all, on_edge)
        self.td.add_particle_property('local density', local_density_all)
        self.td.add_particle_property('shape factor', shape_factor_all)
        self.td.add_particle_property('on edge', on_edge_all)

    def edge_distance(self):
        boundary = self.td.get_boundary(0)
        x = self.td.get_column('x')
        y = self.td.get_column('y')
        points = np.vstack((x, y)).transpose()
        bound = Polygon(boundary)
        distance = []
        for point in tqdm(points, 'Edge Distance'):
            distance.append(bound.exterior.distance(Point(point)))
        self.td.add_particle_property('Edge Distance', distance)

    def correlations(self, frame_no, r_min=1, r_max=10, dr=0.02):
        """
        Calculates spatial and orientational correlations.

        Uses definitions of g and g6 from Komatsu2015

        Parameters
        ----------
        frame_no: The frame to calculate for
        r_min: lower bound for r in diameters
        r_max: upper bound for r in diameters
        dr: bin size for r in diameters

        Notes
        -----
        Adds data to the *_corr.hdf5 dataframe.

        """
        if 'complex order' not in self.td.get_headings():
            self.order_parameter()
        if 'Edge Distance' not in self.td.get_headings():
            self.edge_distance()
        data = self.td.get_info(
            frame_no, ['x', 'y', 'size', 'complex order', 'Edge Distance'])
        diameter = np.mean(np.real(data[:, 2])) * 2
        data[:, 4] /= diameter # pix -> diameters

        boundary = self.td.get_boundary(frame_no)
        area = calculate_area_from_boundary(boundary) / diameter**2  # d**2
        density = len(data) / area  # number density in tray

        # Calculate a sample space diagram containing the distances between
        # pairs of particles in units of diameters
        dists = sp.distance.pdist(np.real(data[:, :2])/diameter)
        dists = sp.distance.squareform(dists)

        r_values = np.arange(r_min, r_max, dr)
        g = np.zeros(len(r_values))
        g6 = np.zeros(len(r_values))
        for i, r in enumerate(r_values):
            # Find indices for points which are greater than (r+dr) from the
            # boundary.
            j_indices = np.squeeze(np.argwhere(data[:, 4] > r+dr))

            # Calculate divisor for each value of r, describing the expected
            # number of pairs at this separation
            n = len(j_indices)
            divisor = 2 * np.pi * r * dr * density * (n-1)

            # Remove points nearer than (r+dr) from one axis of dists
            dists_include = dists[j_indices, :]

            # Count the number of pairs with distances in the bin
            counted = np.argwhere(abs(dists_include-r-dr/2) <= dr/2)
            g[i] = len(counted) / divisor

            # Multiply and sum pairs of order parameters using vector
            # dot product
            order1s = data[counted[:, 0], 3]
            order2s = data[counted[:, 1], 3]
            g6[i] = np.abs(np.vdot(order1s, order2s)) / divisor

        corr_data = dataframes.CorrData(self.corename)
        corr_data.add_row(r_values, frame_no, 'r')
        corr_data.add_row(g, frame_no, 'g')
        corr_data.add_row(g6, frame_no, 'g6')

    def level_checks(self):
        fig_name = self.corename + '_level_figs.png'

        frames = np.arange(0, self.td.num_frames)
        average_x = np.zeros(self.td.num_frames)
        average_x_err = np.zeros(self.td.num_frames)
        average_y = np.zeros(self.td.num_frames)
        average_y_err = np.zeros(self.td.num_frames)

        rad = np.mean(self.td.get_info(1, ['size']))

        boundary = self.td.get_boundary(0)
        center_of_tray = np.mean(boundary, axis=0)

        max_dist = np.linalg.norm(boundary[0, :] - center_of_tray)

        dist_bins = np.linspace(0, max_dist, 10)
        dist_data = np.zeros((self.td.num_frames, len(dist_bins)-1))

        angle_bins = np.linspace(0, 2*np.pi, 17)
        angle_data = np.zeros((self.td.num_frames, len(angle_bins)-1))
        all_x = np.array([])
        all_y = np.array([])
        for f in frames:
            data = self.td.get_info(f, ['x', 'y'])
            vectors = data - center_of_tray
            all_x = np.append(all_x, vectors[:, 0])
            all_y = np.append(all_y, vectors[:, 1])
            vectors_norm = np.linalg.norm(vectors, axis=1)

            average_x[f] = np.mean(vectors[:, 0])
            average_x_err[f] = np.std(vectors[:, 0])
            average_y[f] = np.mean(vectors[:, 1])
            average_y_err[f] = np.std(vectors[:, 1])

            dist_data[f, :], _ = np.histogram(vectors_norm, dist_bins)
            angles = np.arctan2(vectors[:, 0], vectors[:, 1])
            angles += np.pi
            angle_data[f, :], _ = np.histogram(angles, angle_bins)

        average_x /= rad
        average_x_err = rad
        average_y /= rad
        average_y_err /= rad
        # plot = graphs.Plotter(2, 2)
        # plot.add_scatter(0, frames, average_x)
        # plot.add_scatter(0, frames, average_y)
        # plot.config_axes(0, xlabel='frames',
        #                  ylabel='$\Delta x / \sigma, \Delta y / \sigma$',
        #                  legend=['x', 'y'])
        dist_bins /= rad
        hist_means = np.mean(dist_data, axis=0)
        hist_err = np.std(dist_data, axis=0)
        # plot.add_bar(1, dist_bins[:-1], hist_means, yerr=hist_err)
        # plot.config_axes(1, xlabel='$r/\sigma$', ylabel='Frequency')
        #
        angle_means = np.mean(angle_data, axis=0)
        angle_err = np.std(angle_data, axis=0)
        # plot.add_polar_bar(2, angle_bins[:-1], angle_means, angle_err)
        #
        # all_x /= rad
        # all_y /= rad
        # plot.add_hexbin(3, all_x, all_y, gridsize=20, mincnt=10)
        # plot.config_axes(3, xlabel='$x/\sigma$', ylabel='$y/\sigma$')
        #
        # plot.save(fig_name)
        #
        # plot.show()

        # graphs.polar_histogram(angle_bins, angle_means,
        #                        filename=self.corename+'_angle_hist.png',
        #                        y_err=angle_err)
        #
        # plt.figure()
        all_x /= rad
        all_y /= rad
        hb = plt.hexbin(all_x, all_y, gridsize=20, mincnt=0)
        # cb = plt.colorbar()
        # plt.show()

        ### Save Data ###


        self.plot_data.add_column('mean x pos', average_x)
        self.plot_data.add_column('mean y pos', average_y)
        self.plot_data.add_column('mean pos frames', frames)

        self.plot_data.add_column('mean dist bins', dist_bins)
        self.plot_data.add_column('mean dist hist', hist_means)
        self.plot_data.add_column('mean dist hist err', hist_err)

        self.plot_data.add_column('angle dist bins', angle_bins)
        self.plot_data.add_column('angle dist hist', angle_means)
        self.plot_data.add_column('angle dist hist err', angle_err)

        self.plot_data.add_column('hexbin x', all_x)
        self.plot_data.add_column('hexbin y', all_y)

    def average_density(self):
        if 'local density' not in self.td.get_headings():
            self.density()
        density = np.zeros(self.td.num_frames)
        for f in range(self.td.num_frames):
            density[f] = np.mean(self.td.get_info(f, ['local density']))
        self.td.add_frame_property('local density', density)

    @staticmethod
    def show_property_with_condition(points, prop, cond, val, vor=None):
        prop = np.array(prop)
        if cond == '>':
            points_met = np.nonzero(prop > val)
        elif cond == '==':
            points_met = np.nonzero(prop == val)
        elif cond == '<':
            points_met = np.nonzero(prop < val)
        plt.figure()
        plt.plot(points[:, 0], points[:, 1], 'x')
        plt.plot(points[points_met, 0], points[points_met, 1], 'o')
        if vor:
            sp.voronoi_plot_2d(vor)
        plt.show()

    @staticmethod
    def _count_neighbors(list_indices):
        neighbors = list_indices[1:] - list_indices[:-1]
        return neighbors

    @staticmethod
    def _find_vectors(points, list_indices, point_indices):
        neighbors = points[point_indices]
        points_to_subtract = np.zeros(np.shape(neighbors))
        for p in range(len(points)):
            points_to_subtract[list_indices[p]:list_indices[p+1], :] = \
                points[p]
        vectors = neighbors - points_to_subtract
        return vectors

    @staticmethod
    def _calculate_orders(angles, list_indices):
        step = np.exp(6j * angles)
        orders = []
        for p in range(len(list_indices)-1):
            part = step[list_indices[p]:list_indices[p+1]]
            total = sum(part)/len(part)
            # orders.append(abs(total)**2)
            orders.append(total)
        return orders

    @staticmethod
    def _find_delaunay_indices(points):
        tess = sp.Delaunay(points)
        return tess.vertex_neighbor_vertices

    @staticmethod
    def _calculate_angles(vectors):
        angles = np.angle(vectors[:, 0] + 1j*vectors[:, 1])
        return angles


def plot_polygons(polygons):
    plt.figure()
    for poly in polygons:
        coords = np.array(poly.exterior.coords)
        plt.fill(coords[:, 0], coords[:, 1])
    plt.show()


def find_points_inside(vertices, boundary):
    path = mpath.Path(boundary.exterior.coords)
    flags = path.contains_points(vertices)
    return flags


def get_polygons(regions, vertices, inside):
    polygons = [Polygon(vertices[r]) for r in regions]
    on_edge = [not all(inside[r]) for r in regions]
    return polygons, on_edge


def area_and_shapefactor(polygons):
    area = np.array([p.area for p in polygons])
    sf = np.array([p.length for p in polygons])**2 / (4 * pi * area)
    return area, sf

@jit
def calculate_polygon_area(x, y):
    p1 = 0
    p2 = 0
    for i in range(len(x)):
        p1 += x[i] * y[i-1]
        p2 += y[i] * x[i-1]
    area = 0.5 * abs(p1-p2)
    return area

@jit
def sort_polygon_vertices(points):
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    angles = np.arctan2((points[:, 1] - cy), (points[:, 0] - cx))
    sort_indices = np.argsort(angles)
    x = points[sort_indices, 0]
    y = points[sort_indices, 1]
    return x, y


def calculate_area_from_boundary(boundary):
    if len(np.shape(boundary)) == 1:
        area = np.pi * boundary[2]**2
    else:
        x, y = sort_polygon_vertices(boundary)
        area = calculate_polygon_area(x, y)
    return area


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def intersect_all_polygons(polygons, boundary, on_edge):
    new_polygons = []
    for i, poly in enumerate(polygons):
        if on_edge[i]:
            new_polygons.append(poly.intersection(boundary))
        else:
            new_polygons.append(poly)
    return new_polygons



if __name__ == "__main__":
    import time
    dataframe = dataframes.DataStore(
            "/home/ppxjd3/Videos/Solid/grid.hdf5",
            load=True)
    PC = PropertyCalculator(dataframe)
    # PC.order_parameter()
    PC.voronoi_cells()
    # PC.edge_distance()

    print(dataframe.particle_data.head())
    print(dataframe.frame_data.head())
