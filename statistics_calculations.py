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
        order_params = np.array([])
        for n in range(num_frames+1):
            points = self.td.extract_points_for_frame(n)
            list_indices, point_indices = self._find_delaunay_indices(points)
            # The indices of neighbouring vertices of vertex k are
            # point_indices[list_indices[k]:list_indices[k+1]].
            vectors = self._find_vectors(points, list_indices, point_indices)
            angles = self._calculate_angles(vectors)
            orders = self._calculate_orders(angles, list_indices)
            order_params = np.append(order_params, orders)
        self.td.add_property_to_dataframe('order2', order_params)

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
        coses = vectors[:, 1]/ np.linalg.norm(vectors, axis=1)
        return np.arccos(coses)

def calculate_angles(vectors):
    coses = vectors[:, 1] / np.linalg.norm(vectors, axis=1)
    return np.arccos(coses)


if __name__ == "__main__":
    dataframe = dataframes.TrackingDataframe(
            "/home/ppxjd3/Videos/12240002_data.hdf5",
            load=True)
    PC = PropertyCalculator(dataframe)
    PC.calculate_hexatic_order_parameter()
    print(dataframe.dataframe.head())
    print(dataframe.dataframe['order'].mean())
    print(dataframe.dataframe['order2'].mean())