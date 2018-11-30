import Generic.filedialogs as filedialogs
import Generic.video as video
import Generic.images as images
import ParticleTracking.preprocessing as preprocessing
import ParticleTracking.tracking as tracking

file = filedialogs.load_filename('Load a video')
# file = "/home/ppxjd3/Videos/15410003.MP4"
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
pt = tracking.ParticleTracker(file, options, methods, True, True, True)
pt.track()
