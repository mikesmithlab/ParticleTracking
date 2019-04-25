import cv2
"""
Add configuration dictionaries for your methodology here.

Each dictionary MUST contain the following keys:
    'crop method': 'blue hex' or 'manual' or None
    'method' : tuple of keys from the printout of this file
    'number of tray sides': if crop method is manual
    'max frame displacement' : trackpy
    'min frame life' : trackpy
    'memory' : trackpy

The trackpy keys can be ignored if the _link_trajectories method is overwritten

Dictionary items with parameters than can be controlled in a gui should
be lists with items [initial, start, stop, step]

Run this file to print out the possible methods in preprocessing.
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
    'method': ('flip', 'threshold'),
    'threshold': [200, 0, 255, 1],
    'threshold mode': cv2.THRESH_BINARY,
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


if __name__ == "__main__":
    from ParticleTracking import preprocessing_methods as pm
    all_dir = dir(pm)
    all_functions = [a for a in all_dir if a[0] != '_']
    print(all_functions)
