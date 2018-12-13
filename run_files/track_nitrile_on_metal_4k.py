from ParticleTracking import tracking, dataframes, statistics
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
pt = tracking.ParticleTracker(file, options, methods, False, True, True)
# pt.track()

data_store = dataframes.DataStore(pt.data_store_filename, load=True)
calculator = statistics.PropertyCalculator(data_store)
calculator.calculate_hexatic_order_parameter()
calculator.calculate_local_rotational_invarient()
calculator.calculate_pair_correlation(1)
calculator.calculate_orientational_correlation(1)
