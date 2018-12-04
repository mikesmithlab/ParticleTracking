import ParticleTracking.tracking as tracking
import numpy as np

file = "/home/ppxjd3/Videos/packed_4.mp4"
crop_points = np.array([(269, 25), (547, 26), (685, 266), (545, 507), (269, 506), (131, 265)])
methods = ['flip', 'threshold tozero', 'opening']
options = {
    'grayscale threshold': None,
    'number of tray sides': 6,
    'min_dist': 6,
    'p_1': 200,
    'p_2': 3,
    'min_rad': 4,
    'max_rad': 7,
    'max frame displacement': 6,
    'min frame life': 10,
    'memory': 8,
    'opening kernel': 6
    }
pt = tracking.ParticleTracker(file, options, methods, False, True, True, crop_points=crop_points)
pt.track()