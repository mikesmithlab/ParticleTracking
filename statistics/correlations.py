import numpy as np
import pandas as pd
from fast_histogram import histogram1d
from scipy import spatial


def corr(features, boundary, r_min, r_max, dr):
    radius = features.r.mean()
    area = calculate_area_from_boundary(boundary)
    N = features.x.count()
    density = N / area

    r_values = np.arange(r_min, r_max, dr) * radius

    dists, orders, N = dists_and_orders(features, r_max * radius)
    g, bins = np.histogram(dists, bins=r_values)
    g6, bins = np.histogram(dists, bins=r_values, weights=orders)

    bin_centres = bins[1:] - (bins[1] - bins[0]) / 2
    divisor = 2 * np.pi * r_values[:-1] * (bins[1] - bins[0]) * density * len(
        dists)

    g = g / divisor
    g6 = g6 / divisor
    return bin_centres, g, g6


def corr_multiple_frames(features, boundary=None, r_min=None, r_max=None,
                         dr=None):
    d = features.Duty.values[0]
    area = calculate_area_from_boundary(boundary)
    radius = features.r.mean()
    group = features.groupby('frame')
    N = group.x.count().mean()
    density = N / area

    res = group.apply(dists_and_orders, t=r_max * radius).values
    dists, orders, N_queried = list(zip(*res))
    dists = np.concatenate(dists)
    orders = np.concatenate(orders)
    N_queried = np.sum(N_queried)

    r_values = np.arange(r_min, r_max, dr) * radius

    divisor = 2 * np.pi * r_values * (dr * radius) * density * N_queried

    g = histogram1d(dists, len(r_values),
                    (np.min(r_values), np.max(r_values)))
    g6 = histogram1d(dists, len(r_values),
                     (np.min(r_values), np.max(r_values)),
                     weights=orders)
    g = g / divisor
    g6 = g6 / divisor
    res = pd.DataFrame({'r': r_values, 'g': g, 'g6': g6})
    return res


def dists_and_orders(f, t=1000):
    idx = get_idx(f, t)
    dists = get_dists(f, idx)
    orders = get_orders(f, idx)
    return dists.ravel(), orders.ravel(), len(dists)


def get_idx(f, t):
    return f.edge_distance.values > t


def get_dists(f, idx):
    x = f[['x', 'y']].values
    return spatial.distance.cdist(x[idx, :], x)


def get_orders(f, idx):
    orders = make_complex(f)
    order_grid = make_order_grid(orders, idx)
    return np.abs(order_grid)


def make_order_grid(orders, idx):
    return orders[idx] @ np.conj(orders).transpose()


def make_complex(f):
    return f[['order_r']].values + 1j * f[['order_i']].values


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

    # df = data.df.loc[0]
    # for i in range(1000):
    #     dists_and_orders(df, 600)

# %%
# boundary = data.metadata['boundary']
# r, g, g6 = corr_multiple_frames(df, boundary, 1, 10, 0.01)
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.plot(r, g - 1)
# plt.subplot(1, 2, 2)
# plt.plot(r, g6 / g)
# plt.show()
