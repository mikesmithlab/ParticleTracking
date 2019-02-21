import scipy.spatial as sp
import numpy as np
from numba import jit


def corr(data, boundary, r_min, r_max, dr):
    diameter = np.mean(np.real(data[:, 2])) * 2

    area = calculate_area_from_boundary(boundary) / diameter
    density = len(data) / area

    dists = sp.distance.pdist(np.real(data[:, :2])/diameter)
    dists = sp.distance.squareform(dists)

    r_values = np.arange(r_min, r_max, dr)
    g = np.zeros(len(r_values))
    g6 = np.zeros(len(r_values))
    for i, r in enumerate(r_values):
        # Find indices for points which are greater than (r+dr) from the
        # boundary.
        j_indices = np.squeeze(np.argwhere(data[:, 4] > r + dr))

        # Calculate divisor for each value of r, describing the expected
        # number of pairs at this separation
        n = len(j_indices)
        divisor = 2 * np.pi * r * dr * density * (n - 1)

        # Remove points nearer than (r+dr) from one axis of dists
        dists_include = dists[j_indices, :]

        # Count the number of pairs with distances in the bin
        counted = np.argwhere(abs(dists_include - r - dr / 2) <= dr / 2)
        g[i] = len(counted) / divisor

        # Multiply and sum pairs of order parameters using vector
        # dot product
        order1s = data[counted[:, 0], 3]
        order2s = data[counted[:, 1], 3]
        g6[i] = np.abs(np.vdot(order1s, order2s)) / divisor
    return r_values, g, g6


def calculate_area_from_boundary(boundary):
    if len(np.shape(boundary)) == 1:
        area = np.pi * boundary[2]**2
    else:
        x, y = sort_polygon_vertices(boundary)
        area = calculate_polygon_area(x, y)
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
def sort_polygon_vertices(points):
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    angles = np.arctan2((points[:, 1] - cy), (points[:, 0] - cx))
    sort_indices = np.argsort(angles)
    x = points[sort_indices, 0]
    y = points[sort_indices, 1]
    return x, y