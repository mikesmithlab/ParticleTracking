import os

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from Generic import filedialogs
from ParticleTracking import statistics, dataframes

sns.set()


def calculate_corr_data(file=None, rmin=1, rmax=20, dr=0.2):
    if file is None:
        file = filedialogs.load_filename()
    new_file = file[:-5] + '_corr.hdf5'
    if not os.path.exists(new_file):
        data = dataframes.DataStore(file)

        calc = statistics.PropertyCalculator(data)
        CD = CorrData(new_file)
        duty = calc.duty()
        duty = np.unique(duty)
        for d in tqdm(duty):
            r, g, g6 = calc.correlations_duty(d, rmin, rmax, dr)
            CD.add_data(d, r, g, g6)
        CD.save()
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
    return pd.read_hdf(filename, 'df')


if __name__ == "__main__":
    # calculate_corr_data()