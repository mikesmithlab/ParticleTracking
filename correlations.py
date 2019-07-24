import os

import pandas as pd
import seaborn as sns

from Generic import filedialogs
from ParticleTracking import statistics, dataframes

sns.set()


def calculate_corr_data(file=None, rmin=1, rmax=20, dr=0.02):
    if file is None:
        file = filedialogs.load_filename()
    new_file = file[:-5] + '_corr.hdf5'
    if not os.path.exists(new_file):
        data = dataframes.DataStore(file)
        calc = statistics.PropertyCalculator(data)
        res = calc.correlations_all_duties(rmin, rmax, dr)
        res = res.reset_index()
        res.to_hdf(new_file, 'df')
    else:
        print('file already exists')


class CorrData:

    def __init__(self, filename):
        self.filename = filename
        self.data = pd.DataFrame(columns=['d', 'r', 'g', 'g6'])

    def add_data(self, d, r, g, g6):
        self.data = pd.concat((
            self.data, pd.DataFrame({'d': d, 'r': [r], 'g': [g], 'g6': [g6]})))

    def save(self):
        self.data.to_hdf(self.filename, 'df')


def load_corr_data(filename):
    return pd.read_hdf(filename)


if __name__ == "__main__":
    calculate_corr_data()
