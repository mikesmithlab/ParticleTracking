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
    'min_dist': 29,
    'p_1': 200,
    'p_2': 5,
    'min_rad': 14,
    'max_rad': 18,
    'max frame displacement': 25,
    'min frame life': 10,
    'memory': 3,
    'opening kernel': 29
    }
pt = tracking.ParticleTracker(file, options, methods, False, True, True)
pt.track()
