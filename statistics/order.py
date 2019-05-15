import scipy.spatial as sp
import numpy as np
from math import exp
from numba import jit


def order_and_neighbors(points, threshold):
    list_indices, point_indices = find_delaunay_indices(points)
    vectors = find_vectors(points, list_indices, point_indices)
    inside_threshold = filter_vectors_length(vectors, threshold)
    angles = calculate_angles(vectors)
    orders, neighbors = calculate_orders(angles, list_indices, inside_threshold)
    neighbors = np.uint8(neighbors)
    orders = np.complex64(orders)
    orders_abs = np.abs(orders)
    mean = np.mean(orders_abs)
    sus = np.var(orders_abs)
    return orders, neighbors, orders_abs, mean, sus


def find_delaunay_indices(points):
    tess = sp.Delaunay(points)
    return tess.vertex_neighbor_vertices


def find_vectors(points, list_indices, point_indices):
    neighbors = points[point_indices]
    points_to_subtract = np.zeros(np.shape(neighbors))
    for p in range(len(points)):
        points_to_subtract[list_indices[p]:list_indices[p + 1], :] = \
            points[p]
    vectors = neighbors - points_to_subtract
    return vectors


def filter_vectors_length(vectors, threshold):
    length = np.linalg.norm(vectors, axis=1)
    return length < threshold


def calculate_angles(vectors):
    angles = np.angle(vectors[:, 0] + 1j * vectors[:, 1])
    return angles


def calculate_orders(angles, list_indices, inside):
    step = list(np.exp(6j * angles))
    parts = [step[list_indices[p]:list_indices[p+1]]
             for p in range(len(list_indices)-1)]
    inside = [inside[list_indices[p]:list_indices[p+1]]
              for p in range(len(list_indices)-1)]
    parts_sum = [sum([i*j for i, j in zip(k, l)])
                 for k, l in zip(parts, inside)]
    neighbors = [sum([item for item in sublist])
                 for sublist in inside]
    orders = [a/b if b != 0 else 0 for a, b in zip(parts_sum, neighbors)]
    return orders, neighbors
