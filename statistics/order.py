import numpy as np
import scipy.spatial as sp
import pandas as pd

def order_process(features, rad_t=3):
    # features = features.copy()
    points = features[['x', 'y', 'r']].values
    orders, neighbors = order_and_neighbors(points[:, :2], np.mean(points[:, 2]) * rad_t)
    features['order_r'] = np.real(orders).astype('float32')
    features['order_i'] = np.imag(orders).astype('float32')
    features['neighbors'] = neighbors
    return features


def order_and_neighbors(points, threshold):
    list_indices, point_indices = find_delaunay_indices(points)
    vectors = find_vectors(points, list_indices, point_indices)
    filtered = filter_vectors(vectors, threshold)
    angles = calculate_angles(vectors)
    orders, neighbors = calculate_orders(angles, list_indices, filtered)
    neighbors = np.real(neighbors).astype('uint8')
    return orders, neighbors


def find_delaunay_indices(points):
    tess = sp.Delaunay(points)
    return tess.vertex_neighbor_vertices


def find_vectors(points, list_indices, point_indices):
    repeat = list_indices[1:] - list_indices[:-1]
    return points[point_indices] - np.repeat(points, repeat, axis=0)


def filter_vectors(vectors, threshold):
    length = np.linalg.norm(vectors, axis=1)
    return length < threshold


def calculate_angles(vectors):
    angles = np.angle(vectors[:, 0] + 1j * vectors[:, 1])
    return angles


def calculate_orders(angles, list_indices, filtered):
    step = np.exp(6j * angles) * filtered
    list_indices -= 1
    stacked = np.cumsum(np.vstack((step, filtered)), axis=1)[:, list_indices[1:]]
    stacked[:, 1:] = np.diff(stacked, axis=1)
    neighbors = stacked[1, :]
    orders = np.zeros_like(neighbors)
    orders[neighbors != 0] = stacked[0, neighbors != 0] / neighbors[neighbors != 0]
    return orders, neighbors


if __name__ == "__main__":
    from Generic import filedialogs
    from ParticleTracking import dataframes, statistics
    file = filedialogs.load_filename()
    data = dataframes.DataStore(file, load=True)
    calc = statistics.PropertyCalculator(data)
    calc.order()
    print(data.df.head())
    # print(data.df.dtypes)