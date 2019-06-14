import scipy.spatial as sp
import numpy as np


def corr(features, boundary, r_min, r_max, dr):
    radius = features.r.mean()  # pixels
    diameter = radius * 2
    area = calculate_area_from_boundary(boundary)  # pixels squared
    N = features.x.count()
    density = N / area  # pixels^-2

    dists = sp.distance.pdist(features[['x', 'y']].values)  # pixels
    dists = sp.distance.squareform(dists)  # pixels
    np.fill_diagonal(dists, 0)

    orders = features[['order_r']].values + 1j * features[['order_i']].values
    order_grid = orders @ np.conj(orders).transpose()

    r_values = np.arange(r_min, r_max, dr) * radius  # pixels

    g, bins = np.histogram(dists, bins=r_values)
    g6, bins = np.histogram(dists, bins=r_values, weights=order_grid)

    bin_centres = bins[1:] - (bins[1] - bins[0]) / 2
    divisor = 2 * np.pi * r_values[:-1] * dr * density * (N - 1)  # unitless

    g = g / divisor
    g6 = g / divisor
    return bin_centres / diameter, g, g6


def calculate_area_from_boundary(boundary):
    if len(np.shape(boundary)) == 1:
        area = np.pi * boundary[2]**2
    else:
        x, y = sort_polygon_vertices(boundary)
        area = calculate_polygon_area(x, y)
    return area


def calculate_polygon_area(x, y):
    p1 = 0
    p2 = 0
    for i in range(len(x)):
        p1 += x[i] * y[i-1]
        p2 += y[i] * x[i-1]
    area = 0.5 * abs(p1-p2)
    return area


def sort_polygon_vertices(points):
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    angles = np.arctan2((points[:, 1] - cy), (points[:, 0] - cx))
    sort_indices = np.argsort(angles)
    x = points[sort_indices, 0]
    y = points[sort_indices, 1]
    return x, y