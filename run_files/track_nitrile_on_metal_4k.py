import ParticleTracking.tracking as tracking
import numpy as np


# file = filedialogs.load_filename('Load a video')
# file = "/home/ppxjd3/Videos/15410003.MP4"
file = "/home/ppxjd3/Videos/packed.mp4"
# outfile = "/home/ppxjd3/Videos/packed_4.mp4"
crop_points = np.array([(1074, 99), (2186, 108), (2743, 1067), (2187, 2026), (1080, 2024), (520, 1064)])
# crop_points = None
methods = ['flip', 'threshold tozero', 'opening']
options = {
    'grayscale threshold': None,
    'number of tray sides': 6,
    'min_dist': 30,
    'p_1': 200,
    'p_2': 3,
    'min_rad': 15,
    'max_rad': 19,
    'max frame displacement': 25,
    'min frame life': 10,
    'memory': 8,
    'opening kernel': 23
    }
pt = tracking.ParticleTracker(file, options, methods, False, True, True, crop_points=crop_points)
pt.track()
