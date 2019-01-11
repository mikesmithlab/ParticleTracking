import numpy as np
from ParticleTracking import dataframes
from ParticleTracking import graphs
import scipy.spatial as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.path as mpath
from numba import jit
import os


class PropertyCalculator:
    """Class to calculate the properties associated with tracking"""
    def __init__(self, tracking_dataframe):
        self.td = tracking_dataframe
        self.num_frames = self.td.num_frames
        self.corename = os.path.splitext(self.td.filename)[0]

    def calculate_orientational_correlation(self, frame_no):
        fig_name = self.corename + \
            '_orientational_correlation_{}.png'.format(frame_no)
        r_name = self.corename + \
            '_orientational_correlation_{}_r.txt'.format(frame_no)
        g6_name = self.corename + \
            '_orientational_correlation_{}_g.txt'.format(frame_no)

        data = self.td.get_info(frame_no, ['x', 'y', 'size', 'order'])
        diameter = np.mean(np.real(data[:, 2])) * 2

        dists = sp.distance.pdist(np.real(data[:, :2])/diameter)
        dists = sp.distance.squareform(dists)

        dr = 0.02  # diameters
        r_values = np.arange(1, 20, dr)
        G6 = np.zeros(r_values.shape)
        for i, r in enumerate(r_values):
            indices = np.argwhere(abs(dists-r) <= dr/2)
            order1s = data[indices[:, 0], 3]
            order2s = data[indices[:, 1], 3]
            G6[i] = np.abs(np.vdot(order1s, order2s)) / len(indices)

        plt.figure()
        plt.loglog(r_values, G6, '-')
        plt.loglog(r_values, max(G6)*r_values**(-1/4), 'r-')
        plt.xlabel('$r/d$')
        plt.ylabel('$G_6(r)$')
        plt.savefig(fig_name)

        np.savetxt(r_name, r_values)
        np.savetxt(g6_name, G6)

    def average_order_parameter(self):
        frames = self.td.num_frames
        orders = np.zeros(frames)
        for f in range(frames):
            orders[f] = self.td.get_info(f, ['loc_rot_invar']).mean()
        plt.figure()
        plt.plot(orders, 'x')
        plt.xlabel('frame')
        plt.ylabel('$<\phi_l >$')
        plt.savefig(self.corename + '_order_ramp.png')

    def calculate_pair_correlation(self, frame_no):
        fig_name = self.corename + '_pair_correlation_{}.png'.format(frame_no)
        r_name = self.corename + '_pair_correlation_{}_r.txt'.format(frame_no)
        g_name = self.corename + '_pair_correlation_{}_g.txt'.format(frame_no)

        data = self.td.get_info(frame_no, ['x', 'y', 'size'])
        pos = data[:, :2]
        diameter = data[:, 2].mean() * 2

        boundary = self.td.get_boundary(frame_no)
        area = calculate_area_from_boundary(boundary) / diameter**2
        n = len(pos)
        density = n / area

        dists = sp.distance.pdist(pos) / diameter
        g, r = np.histogram(dists, bins=np.arange(1, 10, 0.01))
        dr = r[1] - r[0]
        r = r[:-1] + dr/2
        g = g / (2*np.pi*r*dr*density*(n-1))

        plt.figure()
        plt.loglog(r, g-1, '-')
        plt.loglog(r, (g.max()-1)*r**(-1/3))
        plt.xlabel('$r/d$')
        plt.ylabel('$g(r) - 1$')
        plt.savefig(fig_name)

        np.savetxt(r_name, r)
        np.savetxt(g_name, g)

    def calculate_level_checks(self, animate=False):
        fig_name = self.corename + '_mean_position.png'
        f_name = self.corename + '_mean_position_f.txt'
        r_name = self.corename + '_mean_position_r.txt'

        frames = np.arange(0, self.num_frames)
        average_x = np.zeros(self.num_frames)
        average_x_err = np.zeros(self.num_frames)
        average_y = np.zeros(self.num_frames)
        average_y_err = np.zeros(self.num_frames)

        rad = np.mean(self.td.get_info(1, ['size']))

        boundary = self.td.get_boundary(0)
        center_of_tray = np.mean(boundary, axis=0)

        max_dist = np.linalg.norm(boundary[0, :] - center_of_tray)

        dist_bins = np.linspace(0, max_dist, 10)
        dist_data = np.zeros((self.num_frames, len(dist_bins)-1))

        angle_bins = np.linspace(0, 2*np.pi, 17)
        angle_data = np.zeros((self.num_frames, len(angle_bins)-1))
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


        # dist_bins = dist_bins + (dist_bins[1] - dist_bins[0])/2
        dist_bins /= rad
        average_x /= rad
        average_x_err = rad
        average_y /= rad
        average_y_err /= rad
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.errorbar(frames, average_x, yerr=average_x_err, fmt='rx',
                     errorevery=20, capsize=2)
        plt.xlabel('frame')
        plt.ylabel('$\Delta x$ / $\sigma$')
        plt.subplot(1,2,2)
        plt.errorbar(frames, average_y, yerr=average_y_err, fmt='bx',
                     errorevery=20, capsize=2)
        plt.xlabel('frame')
        plt.ylabel('$\Delta y$ / $\sigma$')
        plt.subplots_adjust(wspace=0.3)
        plt.savefig(fig_name)


        plt.figure()
        hist_means = np.mean(dist_data, axis=0)
        hist_err = np.std(dist_data, axis=0)
        plt.bar(dist_bins[:-1], hist_means, yerr=hist_err,
                width=dist_bins[1]-dist_bins[0], align='edge')
        plt.xlabel('Distance r/$\sigma$')
        plt.ylabel('Frequency')
        plt.savefig(self.corename + '_rad_dist.png')


        angle_means = np.mean(angle_data, axis=0)
        angle_err = np.std(angle_data, axis=0)
        graphs.polar_histogram(angle_bins, angle_means,
                               filename=self.corename+'_angle_hist.png',
                               y_err=angle_err)

        plt.figure()
        all_x /= rad
        all_y /= rad
        hb = plt.hexbin(all_x, all_y, gridsize=20, mincnt=10, cmap=plt.cm.YlOrRd_r)
        cb = plt.colorbar()
        plt.show()



    def calculate_shape_factor(self):
        """Calculates Moucka and Nezbedas shape factor"""
        boundary = self.td.get_boundary(0)
        CropVor = CroppedVoronoi(boundary)
        shape_factor_all = np.array([])
        for n in range(self.num_frames+1):
            info = self.td.get_info(n, ['x', 'y'])
            vor = CropVor.add_points(info)
            VorArea = VoronoiArea(vor)
            area = np.array(list(map(VorArea.area, range(len(info)))))
            circumference = np.array(list(map(VorArea.perimeter, range(len(info)))))
            shape_factor = circumference**2 / (4 * np.pi * area)
            shape_factor_all = np.append(shape_factor_all, shape_factor)
        self.td.add_property('shape factor', shape_factor_all)

    def calculate_local_density(self):
        """
        Calculates the local density of each particle in a dataframe.

        Calculates the area of the convex Hull of the voronoi vertices
        corresponding to the cell of each particle.

        Additional particles were added to the boundary of the system
        before calculating the voronoi cell which were then used to bound
        the cells of the outermost particles to the edge of the system.
        """
        boundary = self.td.get_boundary(0)
        CropVor = CroppedVoronoi(boundary)
        local_density_all = np.array([])
        for n in range(self.num_frames+1):
            info = self.td.get_info(n, ['x', 'y', 'size'])
            particle_area = info[:, 2].mean()**2 * np.pi
            vor = CropVor.add_points(info[:, :2])
            VorArea = VoronoiArea(vor)
            area = list(map(VorArea.area, range(len(info))))
            density = particle_area / np.array(area)
            local_density_all = np.append(local_density_all, density)
        self.td.add_property('local_density', local_density_all)

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

    def calculate_local_rotational_invarient(self):
        orders = self.td.get_column('order')
        loc_rot_invar = np.abs(orders)**2
        self.td.add_property('loc_rot_invar', loc_rot_invar)

    def calculate_hexatic_order_parameter(self):
        order_params = np.array([])
        for n in range(self.num_frames+1):
            points = self.td.get_info(n, ['x', 'y'])
            list_indices, point_indices = self._find_delaunay_indices(points)
            # The indices of neighbouring vertices of vertex k are
            # point_indices[list_indices[k]:list_indices[k+1]].
            vectors = self._find_vectors(points, list_indices, point_indices)
            angles = self._calculate_angles(vectors)
            orders = self._calculate_orders(angles, list_indices)
            order_params = np.append(order_params, orders)
        self.td.add_property('order', order_params)

    def find_edge_points(self, check=False):
        edges_array = np.array([], dtype=bool)
        for f in range(self.num_frames+1):
            points = self.td.get_info(f, ['x', 'y'])
            boundary = self.td.get_boundary(f)
            vor = sp.Voronoi(points)
            vertices_outside = self.voronoi_vertices_outside(vor, boundary)
            edges = self.is_point_on_edge(vor, vertices_outside)
            edges_array = np.append(edges_array, edges)
            if check and f == 0:
                plt.figure()
                plt.plot(points[:, 0], points[:, 1], 'o')
                edges_index = np.nonzero(edges == 1)
                plt.plot(points[edges_index, 0], points[edges_index, 1], 'x')
                if len(np.shape(boundary)) > 1:
                    plt.plot(boundary[:, 0], boundary[:, 1], 'b-')
                    plt.plot(boundary[[-1, 0], 0], boundary[[-1, 0], 1], 'b-')
                else:
                    circle = plt.Circle(
                            (boundary[0], boundary[1]),
                            boundary[2],
                            color='r',
                            fill=False)
                    plt.gcf().gca().add_artist(circle)
                plt.show()
        self.td.add_property('on_edge', edges_array)

    @staticmethod
    def is_point_on_edge(vor, vertices_outside):
        edges = np.zeros(len(vor.points), dtype=bool)
        for point_index, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            if np.any(vertices_outside[region]):
                edges[point_index] = True
        return edges

    @staticmethod
    def voronoi_vertices_outside(vor, boundary):
        if len(np.shape(boundary)) == 1:
            vertices_from_centre = vor.vertices - boundary[0:2]
            vertices_outside = np.linalg.norm(vertices_from_centre, axis=1) > \
                boundary[2]
        else:
            path = mpath.Path(boundary)
            vertices_inside = path.contains_points(vor.vertices)
            vertices_outside = ~vertices_inside
        return vertices_outside

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


def generate_circular_boundary_points(cx, cy, rad, n):
    """
    Generates an array of points around a circular boundary

    Parameters
    ----------
    cx, cy is the center coordinate of the circle
    rad is the radius of the circle
    n is the number of points around the circle

    Returns
    -------
    points: (n, 2) array of the output points
    """
    angles = np.linspace(0, 2*np.pi, n)
    x = cx + rad * np.cos(angles)
    y = cy + rad * np.sin(angles)
    points = np.vstack((x, y)).transpose()
    return points


def generate_polygonal_boundary_points(boundary, n, add_dist=0):
    if add_dist > 0:
        boundary = move_points_from_center(boundary, add_dist)
    x = np.array([])
    y = np.array([])
    for p in range(-1, len(boundary)-1):
        xi = np.linspace(boundary[p, 0], boundary[p+1, 0], n//len(boundary))
        yi = np.linspace(boundary[p, 1], boundary[p+1, 1], n//len(boundary))
        x = np.append(x, xi)
        y = np.append(y, yi)
    points = np.vstack((x, y)).transpose()
    return points


def move_points_from_center(points, dist):
    center = (points[:, 0].mean(), points[:, 1].mean())
    points_from_center = points - center
    dists = np.linalg.norm(points_from_center, axis=1)
    dists += dist
    angles = np.angle(points_from_center[:, 0]+1j*points_from_center[:, 1])
    new_x = dists * np.cos(angles)
    new_y = dists * np.sin(angles)
    new_points = np.vstack((new_x, new_y)).transpose()
    new_points += center
    return new_points


def hull_area_2d(points):
    """Calculates area of a 2D convex hull"""
    hull = sp.ConvexHull(points)
    area = hull.volume
    return area


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
def calculate_polygon_perimeter(x, y):
    p = 0
    for i in np.arange(-1, len(x)-1):
        p += ((x[i+1]-x[i])**2 + (y[i+1] - y[i])**2)**0.5
    return p



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
    if len(np.shape(boundary))==1:
        area = np.pi * boundary[2]**2
    else:
        x, y = sort_polygon_vertices(boundary)
        area = calculate_polygon_area(x, y)
    return area

class VoronoiArea:
    """Calculates area of a voronoi cell for a given point"""

    def __init__(self, vor):
        self.vor = vor

    def area(self, point_index):
        region_index = self.vor.regions[self.vor.point_region[point_index]]
        region_points = self.vor.vertices[region_index]
        x, y = sort_polygon_vertices(region_points)
        area = calculate_polygon_area(x, y)
        return area

    def perimeter(self, point_index):
        region_index = self.vor.regions[self.vor.point_region[point_index]]
        region_points = self.vor.vertices[region_index]
        x, y = sort_polygon_vertices(region_points)
        perimeter = calculate_polygon_perimeter(x, y)
        return perimeter

class CroppedVoronoi:
    """
    Generates extra boundary points to give realistic cells to real points
    """

    def __init__(self, boundary):
        if len(np.shape(boundary)) == 1:
            self.boundary_points = generate_circular_boundary_points(
                boundary[0],
                boundary[1],
                boundary[2],
                1000)
            self.edge_points = generate_circular_boundary_points(
                boundary[0],
                boundary[1],
                boundary[2]+20,
                100)
        else:
            self.boundary_points = generate_polygonal_boundary_points(
                boundary, 1000, add_dist=20)
            self.edge_points = generate_polygonal_boundary_points(
                boundary, 100, add_dist=40)
        self.tree = sp.cKDTree(self.boundary_points)

    def add_points(self, points):
        input_points = np.concatenate((points, self.edge_points))
        vor = sp.Voronoi(input_points)
        vertices_to_move = []
        for region in vor.regions:
            if -1 in region:
                vertices_to_move += region
        vertices_to_move = [vertex for vertex in vertices_to_move
                            if vertex != -1]
        unique_vertices_to_move = np.unique(vertices_to_move)
        new_indices = self.tree.query(vor.vertices[unique_vertices_to_move])
        new_vertices = self.boundary_points[new_indices[1]]
        vor.vertices[unique_vertices_to_move] = new_vertices
        return vor


if __name__ == "__main__":
    dataframe = dataframes.DataStore(
            "/home/ppxjd3/Videos/packed_data.hdf5",
            load=True)
    PC = PropertyCalculator(dataframe)
    PC.calculate_shape_factor()
    # PC.calculate_pair_correlation(1)
    # PC.calculate_hexatic_order_parameter()
    # PC.calculate_local_rotational_invarient()
    # PC.calculate_orientational_correlation(1)
    # print(dataframe.particle_data.head())
    # # print(dataframe.dataframe['local density'].values.mean())
    # print(dataframe.particle_data['loc_rot_invar'].mean())
    # PC.find_edge_points(check=True)
    # print(dataframe.dataframe['on_edge'].mean())
    # PC.calculate_local_density()
    # print(dataframe.dataframe.head())