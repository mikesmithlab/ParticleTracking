from ParticleTracking import tracking, dataframes, statistics, graphs
from Generic import filedialogs

file = filedialogs.load_filename('Load a video')
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
import numpy as np
# crop_points = np.array([[1095, 56], [2228, 67], [2792, 1049], [2230, 2023], [1095, 2025], [527, 1048]])
pt = tracking.ParticleTracker(file, methods, options, True, True, True, crop_points=None)
import time
# s = time.time()
# pt.track()
# print(time.time() - s)

data_store = dataframes.DataStore(pt.data_store_filename, load=True)
calculator = statistics.PropertyCalculator(data_store)
# calculator.calculate_level_checks()
calculator.calculate_hexatic_order_parameter()
calculator.calculate_order_magnitude()
calculator.calculate_susceptibility()
# calculator.calculate_pair_correlation(1)
# calculator.calculate_orientational_correlation(1)
# calculator.average_order_parameter()