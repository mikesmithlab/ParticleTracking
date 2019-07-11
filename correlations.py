import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider
from tqdm import tqdm

from Generic import filedialogs
from ParticleTracking import statistics, dataframes


def calculate_corr_data(file=None):
    if file is None:
        file = filedialogs.load_filename()
    data = dataframes.DataStore(file)

    calc = statistics.PropertyCalculator(data)
    CD = CorrData(file[:-5] + '_corr.hdf5')
    duty = calc.duty()
    duty = np.unique(duty)
    for d in tqdm(duty):
        r, g, g6 = calc.correlations_duty(d)
        CD.add_data(d, r, g, g6)
    CD.save()


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


class CorrelationViewer:

    def __init__(self):
        file = filedialogs.load_filename()
        self.data = load_corr_data(file)
        self.duty = self.data.d.values
        self.setup_figure()
        print('data loaded')
        plt.show()

    def setup_figure(self):
        self.fig, self.ax = plt.subplots(1, 2)
        self.fig.subplots_adjust(bottom=0.25)
        ax = self.fig.add_axes([0.1, 0.1, 0.8, 0.05])
        self.duty_slider = Slider(ax, 'Duty', 400, 1000, valinit=0, valstep=1)
        self.duty_slider.on_changed(self.update)
        r, g, g6 = self.get_data(self.duty[0])
        self.plot0, = self.ax[0].plot(r, g - 1)
        self.plot1, = self.ax[1].plot(r, g6 / g)

    def get_data(self, val):
        data = self.data.loc[self.data.d == val, ['r', 'g', 'g6']].values
        r, g, g6 = data.T
        return r[0], g[0], g6[0]

    def update(self, val):
        val = self.duty_slider.val
        if val in self.duty:
            r, g, g6 = self.get_data(val)
            self.plot0.set_ydata(g - 1)
            self.plot1.set_ydata(g6 / g)
            self.ax[0].set_title(str(val))
            self.fig.canvas.draw_idle()


if __name__ == "__main__":
    CorrelationViewer()
