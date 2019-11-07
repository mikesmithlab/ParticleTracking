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
    'method': ('flip', 'crop_and_mask', 'grayscale', 'flip', 'threshold'),
    'number of tray sides': 6,
    'min_dist': [23, 3, 51, 1],
    'p_1': [105, 1, 255, 1],
    'p_2': [4, 1, 20, 1],
    'dp': [1, 1, 5, 1],
    'min_rad': [13, 1, 101, 1],
    'threshold': [170, 1, 255, 1],
    'threshold mode': cv2.THRESH_TOZERO_INV,
    'max_rad': [14, 1, 101, 1],
    'max frame displacement': 10,
    'min frame life': 5,
    'memory': 3
    }

TRACKPY_NITRILE_PARAMETERS = {
    'crop method': 'find_blue_hex_crop_and_mask',
    'method': ('flip', 'crop_and_mask', 'grayscale', 'flip', 'threshold', 'distance'),
    'number of tray sides': 6,
    'threshold': [170, 1, 255, 1],
    'threshold mode': cv2.THRESH_TOZERO_INV,
    'search_range': 15,
    'memory': 3,
    'min frame life': 3
}

EXAMPLE_CHILD_PARAMETERS = {
    'crop method': 'find_blue_hex_crop_and_mask',
    'method': ('flip', 'crop_and_mask', 'grayscale', 'threshold'),
    'threshold': [50, 1, 255, 1],
    'threshold mode': cv2.THRESH_BINARY_INV,
    'number of tray sides': 6,
    'min_dist': [23, 3, 51, 1],
    'p_1': [105, 1, 255, 1],
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

colors 0 = White = too small
       1 = Green = single bacteria
       2 = Red  = dividing bacteria
       3 = Blue = Sticking bacteria
       4 = Black = not classified
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
    'noise cutoff': [0.3, 0, 1, 1],
    'noise floor': 10,
    'single bacterium cutoff': [1.7, 1, 3, 1],
    'outside cutoff': 2,
    'colors': {0:(255,255,255),1:(0,255,0),2:(0,0,255),3:(255,0,0),4:(0,0,0)}
    }

BACTERIA2_PARAMETERS = {
    'crop method': 'no_crop',
    'method': ('grayscale', 'adjust_gamma','adaptive_threshold'),
    'gamma': [30, 0, 1000, 1],
    'adaptive threshold block size': [61, 3, 101, 2],
    'adaptive threshold C': [-17, -30, 30, 1],
    'adaptive threshold mode': [0, 0, 1, 1],
    'area bacterium': [183, 0, 500, 1],
    'aspect bacterium': [147, 2, 1000, 1],
    'max frame displacement': 50,
    'min frame life': 1,
    'memory': 3,
    'trajectory smoothing': 3,
    'noise cutoff': [5, 0, 100, 1],
    'single bacterium cutoff': [237, 100, 300, 1],
    'colors': {0:(0,0,0),
           1:(255,0,0),
           2:(0,255,0),
           3:(0,0,255),
           4:(255,255,0),
           5:(0,255,255),
           6:(255,0,255),
           7:(128,0,0),
           8:(0,128,0),
           9:(0,0,128),
           10:(128,128,0),
           11:(128,0,128),
           12:(0,128,128),
           13:(165,42,42),
           14:(255,69,0),
           15:(0,250,154),
           16:(32,178,170),
           17:(30,144,255),
           18:(139,0,128),
           19:(128,128,128)
           },
    'contour thickness': 2,
    'trajectory thickness': 2,
    'font size': 2,
    'fps': 5,
    'scale':1
    }

TRACKPY_PARAMETERS = {
    'crop method': 'no_crop',
    'method': ('grayscale', 'adjust_gamma','adaptive_threshold'),
    'gamma': [30, 0, 1000, 1],
    'adaptive threshold block size': [61, 3, 101, 2],
    'adaptive threshold C': [-17, -30, 30, 1],
    'adaptive threshold mode': [0, 0, 1, 1],
    'diameter': [10, 0, 500, 2],
    'separation': [1, 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         , 1000, 1],
    'max frame displacement': 50,
    'min frame life': 1,
    'memory': 3,
    'trajectory smoothing': 3,
    'noise cutoff': [5, 0, 100, 1],
    'single bacterium cutoff': [237, 100, 300, 1],
    'colors': {0:(0,0,0),
           1:(255,0,0),
           2:(0,255,0),
           3:(0,0,255),
           4:(255,255,0),
           5:(0,255,255),
           6:(255,0,255),
           7:(128,0,0),
           8:(0,128,0),
           9:(0,0,128),
           10:(128,128,0),
           11:(128,0,128),
           12:(0,128,128),
           13:(165,42,42),
           14:(255,69,0),
           15:(0,250,154),
           16:(32,178,170),
           17:(30,144,255),
           18:(139,0,128),
           19:(128,128,128)
           },
    'contour thickness': 2,
    'trajectory thickness': 2,
    'font size': 2,
    'fps': 5,
    'scale':1
    }

HYDROGEL_PARAMETERS = {
    'crop method': 'no_crop',
    'method': ('grayscale','adaptive_threshold'),
    'adaptive threshold block size': [61, 3, 101, 2],
    'adaptive threshold C': [-17, -30, 30, 1],
    'adaptive threshold mode': [0, 0, 1, 1],
    'max frame displacement': 50,
    'min frame life': 1,
    'memory': 3,
    'trajectory smoothing': 3,
    'noise cutoff': [5, 0, 100, 1],
    'single bacterium cutoff': [237, 100, 300, 1],
    'colors': {0:(0,0,0),
           1:(255,0,0),
           2:(0,255,0),
           3:(0,0,255),
           4:(255,255,0),
           5:(0,255,255),
           6:(255,0,255),
           7:(128,0,0),
           8:(0,128,0),
           9:(0,0,128),
           10:(128,128,0),
           11:(128,0,128),
           12:(0,128,128),
           13:(165,42,42),
           14:(255,69,0),
           15:(0,250,154),
           16:(32,178,170),
           17:(30,144,255),
           18:(139,0,128),
           19:(128,128,128)
           },
    'contour thickness': 2,
    'trajectory thickness': 2,
    'font size': 2,
    'fps': 5,
    'scale':1
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
