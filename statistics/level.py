import numpy as np


def check_level(points, boundary):
    center_of_tray = np.mean(boundary, axis=0)
    max_dist = np.linalg.norm(boundary[0, :] - center_of_tray)
    vectors = points - center_of_tray
