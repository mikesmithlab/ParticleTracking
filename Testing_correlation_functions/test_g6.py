import numpy as np
from math import sqrt, cos, sin, pi
from ParticleTracking import statistics
import matplotlib.pyplot as plt
import scipy.spatial as sp

L = 30
x = np.arange(0, L)
y = np.arange(0, L*sqrt(3), sqrt(3))
X, Y = np.meshgrid(x, y)

points = np.array([X.flatten(), Y.flatten()])
vector = np.array((cos(pi/3), sin(pi/3)))
points = np.append(points, points + vector[:, np.newaxis], axis=1).transpose()
points += np.random.random(np.shape(points))*0.1
boundary_points = np.array([[0, 0], [0, (L-0.5)*sqrt(3)],
                            [(L-0.5), (L-0.5)*sqrt(3)], [(L-0.5), 0], [0, 0]])

# plt.figure()
# plt.plot(points[0], points[1], 'o')
# plt.plot(boundary_points[:, 0], boundary_points[:, 1])
# plt.show()


def order_parameter(points):
    list_indices, points_indices = statistics.PropertyCalculator._find_delaunay_indices(points)
    vectors = statistics.PropertyCalculator._find_vectors(points, list_indices, points_indices)
    angles = statistics.PropertyCalculator._calculate_angles(vectors)
    orders = statistics.PropertyCalculator._calculate_orders(angles, list_indices)
    return orders

orders = np.array(order_parameter(points))
boundary_points_more = statistics.generate_polygonal_boundary_points(boundary_points, L)


def edge_distance(points, boundary_points):
    distance = sp.distance.cdist(points, boundary_points)
    distance = np.sort(distance, axis=1)
    distance_to_edge = distance[:, 0]
    return distance_to_edge

distance = edge_distance(points, boundary_points_more)

density = 2 / (3*sqrt(3)/2)


def correlations(points, orders, edge_dist):
    dists = sp.distance.pdist(points)
    dists = sp.distance.squareform(dists)

    global density

    dr = 0.01
    r_values = np.arange(1, 10, dr)
    g = np.zeros(len(r_values))
    g_all = np.zeros(len(r_values))
    g6 = np.zeros(len(r_values))
    g6_abs = np.zeros(len(r_values))

    for i, r in enumerate(r_values):
        j_indices = np.squeeze(np.argwhere(edge_dist > r+dr))
        n = len(j_indices)
        divisor = 2 * np.pi * r * dr * density * (n-1)
        dists_include = dists[j_indices, :]
        counted = np.argwhere(abs(dists_include-r-dr/2) <= dr/2)
        g[i] = len(counted) / divisor

        order1s = orders[counted[:, 0]]
        order2s = orders[counted[:, 1]]
        summand = np.real(np.vdot(order1s, order2s))
        summand_abs = np.abs(np.vdot(order1s, order2s))
        #
        if len(counted) >> 0:
            g6[i] = summand / len(counted)
            g6_abs[i] = summand_abs / len(counted)

    # plt.figure()
    # plt.loglog(r_values, g-1, 'r-')
    # plt.loglog(r_values, g_all-1, 'b-')
    # plt.legend(['Excluding Edges', 'All Particles'])
    # plt.xlabel('r')
    # plt.ylabel('g(r) - 1')
    # plt.show()

    plt.figure()
    plt.loglog(r_values, g6, 'r-')
    plt.loglog(r_values, g6_abs, 'b-')
    plt.legend(['Real', 'Abs'])
    plt.xlabel('r')
    plt.ylabel('g6/g')
    plt.show()

correlations(points, orders, distance)