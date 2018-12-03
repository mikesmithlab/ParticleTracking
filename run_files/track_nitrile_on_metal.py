import Generic.filedialogs as filedialogs
import Generic.video as video
import Generic.images as images
import ParticleTracking.preprocessing as preprocessing
import ParticleTracking.tracking as tracking
import numpy as np


# file = filedialogs.load_filename('Load a video')
# file = "/home/ppxjd3/Videos/15410003.MP4"
file = "/home/ppxjd3/Videos/packed.mp4"
crop_points = np.array([(1208, 518), (1796, 526), (2416, 1238), (2078, 1637), (1247, 1641), (918, 1159)])
print(np.shape(crop_points))
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
pt = tracking.ParticleTracker(file, options, methods, True, False, False, crop_points)
pt.track()
