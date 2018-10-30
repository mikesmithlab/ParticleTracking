import numpy as np
import matplotlib.pyplot as plt

x1 = np.arange(-19*np.sqrt(3)/2, 19*np.sqrt(3)/2, np.sqrt(3))
y1 = np.arange(-20.5, 20.5, 1)

[X1,Y1] = np.meshgrid(x1,y1)

x2 = np.arange(-20*np.sqrt(3)/2, 20*np.sqrt(3)/2, np.sqrt(3))
y2 = np.arange(-20, 20, 1)

[X2, Y2] = np.meshgrid(x2, y2)

points1 = np.array([X1, Y1])
points2 = np.array([X2, Y2])

points = np.zeros((np.prod(np.shape(points1)), 2))
k = 0
for i in range(np.shape(points1)[1]):
    for j in range(np.shape(points1)[2]):
        points[k, 0] = points1[0, i, j]
        points[k, 1] = points1[1, i, j]
        k += 1

points_2 = np.zeros((np.prod(np.shape(points2)), 2))
k = 0
for i in range(np.shape(points2)[1]):
    for j in range(np.shape(points2)[2]):
        points_2[k, 0] = points2[0, i, j]
        points_2[k, 1] = points2[1, i, j]
        k += 1

all_points = np.concatenate((points, points_2))

plt.figure()
plt.plot(all_points[:, 0], all_points[:, 1], 'x')
plt.show()

import pandas as pd

x = all_points[:, 0]
y = all_points[:, 1]
frame = np.zeros(np.shape(x))
data = pd.DataFrame({"x": x, "y": y, "frame": frame})
store = pd.HDFStore("/home/ppxjd3/Videos/hex_data.hdf5")
store['data'] = data
store['boundary'] = data
store.close()
