import pandas as pd
import numpy as np

GLASS_BEAD_PROCESS_LIST = ['grayscale',
                             'simple threshold',
                             'adaptive threshold',
                             'gaussian blur']

GLASS_BEAD_OPTIONS_DICT = {'config': 1,
                           'title': 'Glass_Bead',
                           'grayscale threshold': 100,
                           'adaptive threshold block size': 11,
                           'adaptive threshold C': 2,
                           'blur kernel': 5,
                           'min_dist': 20,
                           'p_1': 200,
                           'p_2': 10,
                           'min_rad': 18,
                           'max_rad': 20,
                           'number of tray sides': 1,
                           'max frame displacement': 10,
                           'min frame life': 5}

RED_BEAD_OPTIONS_DICT = {'config': 0,
                         'title': 'Red_Bead',
                         'grayscale threshold': 50,
                         'adaptive threshold block size': 5,
                         'adaptive threshold C': 7,
                         'blur kernel': 3,
                         'min_dist': 10,
                         'p_1': 100,
                         'p_2': 20,
                         'min_rad': 15,
                         'max_rad': 23,
                         'number of tray sides': 6,
                         'max frame displacement': 5,
                         'min frame life': 2}

RED_BEAD_PROCESS_LIST = ['grayscale','gaussian blur',
                             'simple threshold',
                             'adaptive threshold']

PARTICLE_LIST = ['Glass_Bead', 'Red_Bead']


class ConfigDataframe:

    def __init__(self):
        self.dataframe = pd.read_csv('configs.csv')

    def get_options(self, config_name):
        options = self.dataframe.loc[
            self.dataframe['title'] == config_name].to_dict('records')
        return options[0]

    def replace_row(self, new_row, title):
        self.dataframe.loc[self.dataframe['title'] == title] = new_row.values()
        print(self.dataframe.head())
        self.save_dataframe()

    def save_dataframe(self):
        self.dataframe.to_csv('configs.csv')

    def print_head(self):
        print(self.dataframe.head())

if __name__=="__main__":
    cd = ConfigDataframe()
    options = cd.get_options('Red_Bead')
    print(options)
    #methods = cd.get_methods('Red_Bead')
    #print(methods)