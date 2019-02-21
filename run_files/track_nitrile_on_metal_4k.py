from ParticleTracking import tracking, dataframes, statistics, graphs, annotation
from Generic import filedialogs
import numpy as np
import warnings
warnings.filterwarnings("ignore")

### Load a file ###
###################
file = filedialogs.load_filename('Load a video', remove_ext=False)

### Tracking ###
###############
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
# crop_points = np.array([[1095, 56], [2228, 67], [2792, 1049], [2230, 2023], [1095, 2025], [527, 1048]])
pt = tracking.ParticleTracker(file, methods, options, False, crop_points=None)
# pt.track()


data_store = dataframes.DataStore(file, load=True)


### Statistics ###
##################
calculator = statistics.PropertyCalculator(data_store)
calculator.distance()
# calculator.edge_distance()
# calculator.level_checks()
# calculator.order()
# calculator.susceptibility()
# calculator.average_order_parameter()
# calculator.density()
# calculator.average_density()
# calculator.correlations(300, r_min=1, r_max=20, dr=0.04)
calculator.correlations(10)

# data_store.inspect_dataframes()

### Annotations ###
###################
# annotator = annotation.VideoAnnotator(data_store, file)
# annotator.add_coloured_circles()
annotation.neighbors(data_store, 0)

### Graphs ###
##############

graphs.order_quiver(data_store, 0)
