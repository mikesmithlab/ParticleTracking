import numpy as np
from scipy import spatial


def corr(features, boundary, r_min, r_max, dr):
    radius = features.r.mean()
    area = calculate_area_from_boundary(boundary)
    N = features.x.count()
    density = N / area

    r_values = np.arange(r_min, r_max, dr) * radius

    dists, orders = dists_and_orders(features, r_max * radius)
    g, bins = np.histogram(dists, bins=r_values)
    g6, bins = np.histogram(dists, bins=r_values, weights=orders)

    bin_centres = bins[1:] - (bins[1] - bins[0]) / 2
    divisor = 2 * np.pi * r_values[:-1] * (bins[1] - bins[0]) * density * len(
        dists)

    g = g / divisor
    g6 = g6 / divisor
    return bin_centres, g, g6


def corr_multiple_frames(features, boundary, r_min, r_max, dr):
    area = calculate_area_from_boundary(boundary)
    radius = features.r.mean()
    N = round(features.groupby('frame').x.count().mean())
    density = N / area

    frames_in_features = np.unique(features.index.values)

    r_values = np.arange(r_min, r_max, dr) * radius

    dists_all = []
    order_all = []
    N_queried = 0
    for frame in frames_in_features:
        features_frame = features.loc[frame]
        dists, orders = dists_and_orders(features_frame, r_max * radius)
        N_queried += len(dists)
        dists_all.append(dists)
        order_all.append(orders)

    dists_all = flat_array(dists_all)
    order_all = flat_array(order_all)

    divisor = 2 * np.pi * r_values[:-1] * (dr * radius) * density * N_queried
    g, bins = np.histogram(dists_all, bins=r_values)
    g6, bins = np.histogram(dists_all, bins=r_values, weights=order_all)
    bin_centers = bins[1:] - (bins[1] - bins[0]) / 2

    g = g / divisor
    g6 = g6 / divisor

    return bin_centers, g, g6


def dists_and_orders(f, t):
    tree = spatial.cKDTree(f[['x', 'y']].values)
    f_to_query = f.loc[f.edge_distance > t]
    dists, idx = tree.query(f_to_query[['x', 'y']].values,
                            k=len(f))

    orders = f[['order_r']].values + 1j * f[['order_i']].values
    orders2 = f_to_query[['order_r']].values + 1j * f_to_query[
        ['order_i']].values
    order_grid = orders2 @ np.conj(orders).transpose()

    # re-sort each row of order_grid to match dists
    order_grid = order_grid[np.arange(len(idx))[:, np.newaxis], idx]
    return dists, order_grid


def flat_array(x):
    return np.concatenate([item.ravel() for item in x])


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
    r, g, g6 = corr_multiple_frames(df, boundary, 1, 20, 0.01)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(r, g)
    plt.subplot(1, 2, 2)
    plt.plot(r, g6 / g)
    plt.show()

# %%
# boundary = data.metadata['boundary']
# r, g, g6 = corr_multiple_frames(df, boundary, 1, 10, 0.01)
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.plot(r, g - 1)
# plt.subplot(1, 2, 2)
# plt.plot(r, g6 / g)
# plt.show()
