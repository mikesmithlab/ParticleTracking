import numpy as np


def distance(points, boundary):
    line_ends = [[boundary[i-1, :], boundary[i, :]] for i in range(6)]
    a, b, c = zip(
        *[[(yb - ya) / (xa - xb), 1, xa * (ya - yb) / (xa - xb) - ya] for
          (xa, ya), (xb, yb) in line_ends])
    a, b, c = np.array(a, ndmin=2), np.array(b, ndmin=2), np.array(c, ndmin=2)
    x, y = points[:, 0, np.newaxis], points[:, 1, np.newaxis]
    # Find distance from each point to each line
    d = np.abs(x @ a + y @ b + c) / (np.sqrt(a ** 2 + b ** 2))
    # Find distance to closest line for each point
    d = np.min(d, axis=1)
    return d.astype('float32')
