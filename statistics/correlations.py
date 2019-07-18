import numpy as np
import scipy.spatial as sp


def corr(features, boundary, r_min, r_max, dr):
    radius = features.r.mean()  # pixels
    area = calculate_area_from_boundary(boundary)  # pixels squared
    N = features.x.count()
    density = N / area  # pixels^-2

    dists = sp.distance.pdist(features[['x', 'y']].values)  # pixels
    dists = sp.distance.squareform(dists)  # pixels

    orders = features[['order_r']].values + 1j * features[['order_i']].values
    order_grid = orders @ np.conj(orders).transpose()

    r_values = np.arange(r_min, r_max, dr) * radius  # pixels

    g, bins = np.histogram(dists, bins=r_values)
    g6, bins = np.histogram(dists, bins=r_values, weights=order_grid)

    bin_centres = bins[1:] - (bins[1] - bins[0]) / 2
    divisor = 2 * np.pi * r_values[:-1] * dr * density * (N - 1)  # unitless

    g = g / divisor
    g6 = g6 / divisor
    return bin_centres, g, g6


def corr_multiple_frames(features, boundary, r_min, r_max, dr):
    area = calculate_area_from_boundary(boundary)
    radius = features.r.mean()
    N = round(features.groupby('frame').x.count().mean())
    density = N / area

    frames_in_features = np.unique(features.index.values)

    dists_all = []
    order_all = []
    for frame in frames_in_features:
        features_frame = features.loc[frame]
        dists = sp.distance.pdist(features_frame[['x', 'y']].values)
        dists = sp.distance.squareform(dists)
        orders = features_frame[['order_r']].values + 1j * features_frame[
            ['order_i']].values
        order_grid = orders @ np.conj(orders).transpose()
        dists_all.append(dists)
        order_all.append(order_grid)

    r_values = np.arange(r_min, r_max, dr) * radius
    divisor = 2 * np.pi * r_values[:-1] * dr * density * (N - 1) * len(
        frames_in_features)
    g, bins = np.histogram(dists, bins=r_values)
    g6, bins = np.histogram(dists, bins=r_values, weights=order_grid)
    bin_centers = bins[1:] - (bins[1] - bins[0]) / 2

    g = g / divisor
    g6 = g6 / divisor

    return bin_centers, g, g6


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


if __name__ == "__main__":
    from ParticleTracking import dataframes
    import matplotlib.pyplot as plt

    file = "/media/data/Data/July2019/RampsN29/15790009.hdf5"
    data = dataframes.DataStore(file)
    df = data.df.loc[:50]
    boundary = data.metadata['boundary']
    r, g, g6 = corr_multiple_frames(df, boundary, 1, 10, 0.01)
    plt.figure()
    plt.plot(r, g)
    plt.show()
