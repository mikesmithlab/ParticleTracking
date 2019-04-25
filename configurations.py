"""
Add methods and parameters for different systems here

Dictionary items with parameters than can be controlled in a gui should
be lists with items [initial, start, stop, step]

method should be a tuple so it isn't parsed by the gui
"""



NITRILE_BEADS_PARAMETERS = {
    'crop method': 'blue hex',
    'method': ('flip',),
    'number of tray sides': 6,
    'min_dist': [23, 3, 51, 1],
    'p_1': [105, 0, 255, 1],
    'p_2': [2, 0, 20, 1],
    'min_rad': [13, 1, 101, 1],
    'max_rad': [14, 1, 101, 1],
    'max frame displacement': 10,
    'min frame life': 5,
    'memory': 3
    }

EXAMPLE_CHILD_PARAMETERS = {
    'crop method': 'blue hex',
    'method': ('flip', 'simple threshold'),
    'number of tray sides': 6,
    'min_dist': [23, 3, 51, 1],
    'p_1': [105, 0, 255, 1],
    'p_2': [2, 0, 20, 1],
    'min_rad': [13, 1, 101, 1],
    'max_rad': [14, 1, 101, 1],
    'max frame displacement': 10,
    'min frame life': 5,
    'memory': 3,
    'grayscale threshold': [50, 0, 255, 1]
}