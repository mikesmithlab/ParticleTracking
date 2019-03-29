from ParticleTracking import tracking, dataframes, statistics, graphs, annotation
from Generic import filedialogs
import numpy as np
import warnings
warnings.filterwarnings("ignore")

### Load a file ###
###################
file = filedialogs.load_filename('Load a video', remove_ext=False, directory='/home/ppxjd3/Videos/')

### Tracking ###
###############
methods = ['flip']
options = {
    'number of tray sides': 6,
    'min_dist': 23,
    'p_1': 105,
    'p_2': 2,
    'min_rad': 13,
    'max_rad': 14,
    'max frame displacement': 10,
    'min frame life': 50,
    'memory': 3
    }

pt = tracking.ParticleTracker(file, methods, options, True, auto_crop=True, debug=False)

pt.track()


data_store = dataframes.DataStore(file, load=True)


### Statistics ###
##################
# calculator = statistics.PropertyCalculator(data_store)
# calculator.distance()
# calculator.edge_distance()
# calculator.level_checks()
# calculator.order()
# calculator.susceptibility()
# calculator.average_order_parameter()
# calculator.density()
# calculator.average_density()
# calculator.correlations(300, r_min=1, r_max=20, dr=0.04)
# calculator.correlations(10)

# data_store.inspect_dataframes()

### Annotations ###
###################
# annotator = annotation.VideoAnnotator(data_store, file)
# annotator.add_coloured_circles()
# # annotation.neighbors(data_store, 0)

### Graphs ###
##############

# graphs.order_quiver(data_store, 0)
