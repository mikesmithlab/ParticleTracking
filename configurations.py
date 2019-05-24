import cv2
"""
Add configuration dictionaries for your methodology here.

Each dictionary MUST contain the following keys:
    'crop method': method in preprocessing_crops
    'method' : tuple of keys from the printout of this file
    'number of tray sides': if crop method is manual
    'max frame displacement' : trackpy
    'min frame life' : trackpy
    'memory' : trackpy

For many parameter sets we have [start value, min value, max value, ?]

The trackpy keys can be ignored if the _link_trajectories method is overwritten

Dictionary items with parameters than can be controlled in a gui should
be lists with items [initial, start, stop, step]

Run this file to print out the possible methods in preprocessing.
"""

NITRILE_BEADS_PARAMETERS = {
    'crop method': 'find_blue_hex_crop_and_mask',
    'method': ('flip', 'crop_and_mask', 'grayscale'),
    'number of tray sides': 6,
    'min_dist': [23, 3, 51, 1],
    'p_1': [105, 0, 255, 1],
    'p_2': [2, 1, 20, 1],
    'min_rad': [13, 1, 101, 1],
    'max_rad': [14, 1, 101, 1],
    'max frame displacement': 10,
    'min frame life': 5,
    'memory': 3
    }

EXAMPLE_CHILD_PARAMETERS = {
    'crop method': 'no_crop',
    'method': ('grayscale', 'flip', 'threshold'),
    'threshold': [200, 0, 255, 1],
    'threshold mode': cv2.THRESH_BINARY,
    'number of tray sides': 6,
    'min_dist': [23, 3, 51, 1],
    'p_1': [105, 0, 255, 1],
    'p_2': [2, 1, 20, 1],
    'min_rad': [13, 1, 101, 1],
    'max_rad': [14, 1, 101, 1],
    'max frame displacement': 10,
    'min frame life': 5,
    'memory': 3
}


'''
min area is a threshold that is slightly larger than a single bacterium.
The aim is to be able to identify when a bacterium might be dividing.

colors 1 = Green = single bacteria
       2 = Blue  = dividing bacteria
       3 = Red = Sticking bacteria
       4 = turquoise = not classified
'''
BACTERIA_PARAMETERS = {
    'crop method': 'no_crop',
    'method': ('grayscale', 'adaptive_threshold'),
    'adaptive threshold block size': [53, 3, 101, 2],
    'adaptive threshold C': [-26, -30, 30, 1],
    'adaptive threshold mode': [0, 0, 1, 1],
    'area bacterium': [114, 0, 500, 1],
    'width bacterium': [8, 0, 50, 1],
    'max frame displacement': 20,
    'min frame life': 5,
    'memory': 3,
    'trajectory smoothing': 1,
    'outside cutoff': 2,
    'colors': {1:(0,255,0),2:(255,0,0),3:(0,0,255),4:(255,255,0)}
    }


if __name__ == "__main__":
    from ParticleTracking.preprocessing import preprocessing_methods as pm
    from ParticleTracking.preprocessing import preprocessing_crops as pc

    all_dir = dir(pc)
    all_functions = [a for a in all_dir if a[0] != '_']
    print('preprocessing crops')
    print(all_functions)

    print('')

    all_dir = dir(pm)
    all_functions = [a for a in all_dir if a[0] != '_']
    print('preprocessing methods')
    print(all_functions)
