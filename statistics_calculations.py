import numpy as np
import dataframes
import scipy.spatial as sp
import matplotlib.pyplot as plt


class PropertyCalculator:
    """Class to calculate the properties associated with tracking"""
    def __init__(self, tracking_dataframe):
        self.td = tracking_dataframe

    def calculate_hexatic_order_parameter(self):
        num_frames = int(np.max(self.td.dataframe['frame']))
        for n in range(num_frames+1):
            points = self.td.extract_points_for_frame(n)

            tess = sp.Delaunay(points)
            list_indices, point_indices = tess.vertex_neighbor_vertices
            # The indices of neighbouring vertices of vertex k are
            # point_indices[list_indices[k]:list_indices[k+1]].

            if n == 0:
                plt.figure()
                plt.triplot(points[:, 0], points[:, 1], tess.simplices.copy())
                plt.plot(points[:, 0], points[:, 1], 'o')
                plt.axis('equal')
                plt.show()

            orders = []
            no_of_neighbors = []
            for p in range(len(points)):
                point = points[p, :]
                neighbors_indices = point_indices[list_indices[p]:
                                                  list_indices[p+1]]
                neighbors = points[neighbors_indices]
                vectors = neighbors - point
                angles = self._calculate_angles(vectors)
                orders.append(self._calc_order_parameter_from_angles(angles))
                no_of_neighbors.append(len(neighbors))

            if n == 0:
                ordered_indices = np.nonzero(np.array(orders) > .3)
                ordered_points = points[ordered_indices, :].squeeze()
                plt.figure()
                plt.plot(points[:, 0], points[:, 1], 'x')
                plt.plot(ordered_points[:, 0], ordered_points[:, 1], 'o')
                plt.show()

    @staticmethod
    def _calculate_angles(vectors):
        j = (0, 1)
        coses = np.dot(vectors, (0, 1))/np.linalg.norm(vectors, axis=1)
        return np.arccos(coses)

    @staticmethod
    def _calc_order_parameter_from_angles(angles):
        angles = np.array(angles)
        angles = np.exp(6j*angles)
        angles = 1/6 * np.sum(angles)
        return np.real(angles * np.conj(angles))


if __name__ == "__main__":
    dataframe = dataframes.TrackingDataframe(
            "/home/ppxjd3/Videos/12240002_data.hdf5",
            load=True)
    PC = PropertyCalculator(dataframe)
    PC.calculate_hexatic_order_parameter()
