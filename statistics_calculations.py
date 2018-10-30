import numpy as np
import dataframes
import scipy.spatial as sp
import matplotlib.pyplot as plt
import math as m

class PropertyCalculator:
    """Class to calculate the properties associated with tracking"""
    def __init__(self, tracking_dataframe):
        self.td = tracking_dataframe

    def calculate_hexatic_order_parameter(self):
        num_frames = int(np.max(self.td.dataframe['frame']))
        count = 0
        order_param_array = np.zeros(len(self.td.dataframe['x']))

        for n in range(num_frames+1):
            points = self.td.extract_points_for_frame(n)
            list_indices, point_indices = self._find_delaunay_indices(points)
            # The indices of neighbouring vertices of vertex k are
            # point_indices[list_indices[k]:list_indices[k+1]].

            for p in range(len(points)):
                point = points[p, :]
                neighbors_indices = point_indices[list_indices[p]:
                                                  list_indices[p+1]]
                neighbors = points[neighbors_indices]
                vectors = neighbors - point
                angles = self._calculate_angles(vectors)
                order_param_array[count] = \
                    self._calc_order_parameter_from_angles(angles)
                count += 1
        self.td.add_property_to_dataframe('order2', order_param_array)

    @staticmethod
    def _find_delaunay_indices(points):
        tess = sp.Delaunay(points)
        return tess.vertex_neighbor_vertices

    @staticmethod
    def _calculate_angles(vectors):
        coses = np.dot(vectors, (0, 1))/np.linalg.norm(vectors, axis=1)
        return np.arccos(coses)

    @staticmethod
    def _calc_order_parameter_from_angles(angles):
        tot = 0
        for ang in angles:
            part = np.exp(6j * ang)
            tot += part
        tot /= 6
        return abs(tot) ** 2


if __name__ == "__main__":
    dataframe = dataframes.TrackingDataframe(
            "/home/ppxjd3/Videos/hex_data.hdf5",
            load=True)
    PC = PropertyCalculator(dataframe)
    PC.calculate_hexatic_order_parameter()
    print(dataframe.dataframe.head())