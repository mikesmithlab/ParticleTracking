import numpy as np
import scipy.spatial as sp


def order_process(features, threshold=2.3):
    points = features[['x', 'y', 'r']].values
    threshold *= np.mean(points[:, 2])
    orders, neighbors = order_and_neighbors(points[:, :2], threshold)
    features['order_r'] = np.real(orders).astype('float32')
    features['order_i'] = np.imag(orders).astype('float32')
    features['neighbors'] = neighbors
    return features


def order_process_mean(features, threshold=2.3):
    points = features[['x_mean', 'y_mean', 'r']].values
    threshold *= np.mean(points[:, 2])
    orders, neighbors = order_and_neighbors(points[:, :2], threshold)
    features['order_r_mean'] = np.real(orders).astype('float32')
    features['order_i_mean'] = np.imag(orders).astype('float32')
    features['neighbors_mean'] = neighbors
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
    # calculate summand for every angle
    step = np.exp(6j * angles)
    # set summand to zero if bond length > threshold
    step *= filtered
    list_indices -= 1
    # sum the angles and count neighbours for each particle
    stacked = np.cumsum((step, filtered), axis=1)[:, list_indices[1:]]
    stacked[:, 1:] = np.diff(stacked, axis=1)
    neighbors = stacked[1, :]
    indxs = neighbors != 0
    orders = np.zeros_like(neighbors)
    orders[indxs] = stacked[0, indxs] / neighbors[indxs]
    return orders, neighbors


if __name__ == "__main__":
    from Generic import filedialogs
    from ParticleTracking import statistics
    from ParticleTracking.general import dataframes

    file = filedialogs.load_filename()
    data = dataframes.DataStore(file, load=True)
    calc = statistics.PropertyCalculator(data)
    calc.order()
    print(data.df.head())
    # print(data.df.dtypes)