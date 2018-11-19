import numpy as np
import ParticleTracking.dataframes as df
import scipy.spatial as sp
import matplotlib.pyplot as plt
import matplotlib.path as mpath


class PropertyCalculator:
    """Class to calculate the properties associated with tracking"""
    def __init__(self, tracking_dataframe):
        self.td = tracking_dataframe
        self.num_frames = int(np.max(self.td.dataframe['frame']))

    def calculate_local_density(self):
        """Calculate local density from the voronoi network"""
        local_density_all = np.array([])
        for n in range(self.num_frames+1):
            info = self.td.extract_points_for_frame(n, include_size=True)
            particle_area = info[:,2].mean()**2*np.pi
            vor = sp.Voronoi(info[:, 0:2])
            local_density = []
            for p, r in enumerate(vor.point_region):
                region = vor.regions[r]
                region_points = vor.vertices[region]
                if -1 in region:
                    density = None
                else:
                    area = self.hull_area(region_points)
                    density = particle_area / area

                local_density.append(density)
        local_density_all = np.append(local_density_all, local_density)
        self.show_property_with_condition(info, local_density, '==', None)

    @staticmethod
    def show_property_with_condition(points, prop, cond, val):
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
        plt.show()


    @staticmethod
    def polygon_area(points):
        """Return the area of the polygon whose vertices are given by the
        sequence points.

        """
        area = 0
        q = points[-1]
        for p in points:
            area += p[0] * q[1] - p[1] * q[0]
            q = p
        return area / 2

    @staticmethod
    def hull_area(points):

        hull = sp.ConvexHull(points)
        area = hull.area
        return area


    def calculate_hexatic_order_parameter(self):
        order_params = np.array([])
        for n in range(self.num_frames+1):
            points = self.td.extract_points_for_frame(n)
            list_indices, point_indices = self._find_delaunay_indices(points)
            # The indices of neighbouring vertices of vertex k are
            # point_indices[list_indices[k]:list_indices[k+1]].
            vectors = self._find_vectors(points, list_indices, point_indices)
            angles = self._calculate_angles(vectors)
            orders = self._calculate_orders(angles, list_indices)
            order_params = np.append(order_params, orders)
        self.td.add_property_to_dataframe('order', order_params)

    def find_edge_points(self, check=False):
        edges_array = np.array([], dtype=bool)
        for f in range(self.num_frames+1):
            points = self.td.extract_points_for_frame(f)
            boundary = self.td.extract_boundary_for_frame(f)
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
                    circle = plt.Circle((boundary[0], boundary[1]), boundary[2], color='r', fill=False)
                    plt.gcf().gca().add_artist(circle)
                plt.show()
        self.td.add_property_to_dataframe('on_edge', edges_array)

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
            vertices_outside = np.linalg.norm(vertices_from_centre, axis=1) > boundary[2]
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
            points_to_subtract[list_indices[p]:list_indices[p+1], :] = points[p]
        vectors = neighbors - points_to_subtract
        return vectors

    @staticmethod
    def _calculate_orders(angles, list_indices):
        step = np.exp(6j * angles)/6
        orders = []
        for p in range(len(list_indices)-1):
            part = step[list_indices[p]:list_indices[p+1]]
            total = sum(part)
            orders.append(abs(total)**2)
        return orders

    @staticmethod
    def _find_delaunay_indices(points):
        tess = sp.Delaunay(points)
        return tess.vertex_neighbor_vertices

    @staticmethod
    def _calculate_angles(vectors):
        angles = np.angle(vectors[:, 0] + 1j*vectors[:, 1])
        return angles


if __name__ == "__main__":
    dataframe = df.TrackingDataframe(
            "/home/ppxjd3/Videos/test_data.hdf5",
            load=True)
    PC = PropertyCalculator(dataframe)
    #PC.calculate_hexatic_order_parameter()
    # print(dataframe.dataframe.head())
    #print(dataframe.dataframe['order'].mean())
    # PC.find_edge_points(check=True)
    # print(dataframe.dataframe['on_edge'].mean())
    PC.calculate_local_density()