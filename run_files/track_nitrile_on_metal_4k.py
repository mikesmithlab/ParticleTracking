from ParticleTracking import tracking, dataframes, statistics, graphs, annotation
from Generic import filedialogs
import numpy as np
import warnings
warnings.filterwarnings("ignore")

### Load a file ###
###################
file = filedialogs.load_filename('Load a video', remove_ext=False)

### Tracking ###
################
# methods = ['flip', 'threshold tozero', 'opening']
# options = {
#     'grayscale threshold': None,
#     'number of tray sides': 6,
#     'min_dist': 30,
#     'p_1': 200,
#     'p_2': 3,
#     'min_rad': 15,
#     'max_rad': 19,
#     'max frame displacement': 25,
#     'min frame life': 10,
#     'memory': 8,
#     'opening kernel': 23
#     }
# # crop_points = np.array([[1095, 56], [2228, 67], [2792, 1049], [2230, 2023], [1095, 2025], [527, 1048]])
# pt = tracking.ParticleTracker(file, methods, options, True, crop_points=None)
# pt.track()


data_store = dataframes.DataStore(file, load=True)


### Statistics ###
##################
calculator = statistics.PropertyCalculator(data_store)
# # # calculator.calculate_level_checks()
calculator.order_parameter()
# calculator.calculate_susceptibility()
# calculator.average_order_parameter()
# calculator.calculate_local_density()
# calculator.calculate_average_local_density()
calculator.correlations(1)

### Annotations ###
###################
# annotator = annotation.VideoAnnotator(data_store, file)
# annotator.add_coloured_circles()