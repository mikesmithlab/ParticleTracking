import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.widgets import Slider, RadioButtons, Button
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


class CorrelationViewer:

    def __init__(self):
        self.open(None)
        self.setup_figure()
        print('data loaded')
        plt.show()

    def open(self, event):
        self.file = filedialogs.load_filename()
        self.data = load_corr_data(self.file)
        self.duty = self.data.d.values

    def setup_figure(self):
        self.fig = plt.figure()
        gs = self.fig.add_gridspec(5, 3, height_ratios=(4, 4, 1, 1, 1),
                              width_ratios=(5, 5, 1), wspace=0.3, hspace=0.3)
        self.g_ax = self.fig.add_subplot(gs[:2, 0])
        self.g6_ax = self.fig.add_subplot(gs[:2, 1])
        duty_ax = self.fig.add_subplot(gs[2, :2])
        self.duty_slider = Slider(duty_ax, 'Duty', 400, 1000, valinit=0,
                                  valstep=1)
        self.duty_slider.on_changed(self.get_vals)
        #
        g_fit_ax = self.fig.add_subplot(gs[3, 0])
        self.g_fit_slider = Slider(g_fit_ax, 'g_fit_offset', -1, 2, valinit=0,
                                   valstep=0.01)
        self.g_fit_slider.on_changed(self.get_vals)
        #
        g6_fit_ax = self.fig.add_subplot(gs[3, 1])
        self.g6_fit_slider = Slider(g6_fit_ax, 'g6 fit offset', -1, 5,
                                    valinit=0, valstep=0.01)
        self.g6_fit_slider.on_changed(self.get_vals)

        radio_ax = self.fig.add_subplot(gs[0, 2])
        self.radio = RadioButtons(radio_ax, ('xy', 'logx', 'logy', 'loglog'))
        self.radio.on_clicked(self.get_vals)

        open_ax = self.fig.add_subplot(gs[1, 2])
        self.open_button = Button(open_ax, 'Open')
        self.open_button.on_clicked(self.open)

        g6_power_ax = self.fig.add_subplot(gs[4, 1])
        self.g6_power_slider = Slider(g6_power_ax, 'g6 power (1/)', 3, 5,
                                      valinit=4, valstep=0.1)
        self.g6_power_slider.on_changed(self.get_vals)

        r, g, g6 = self.get_data(self.duty[0])
        self.plot_g, = self.g_ax.plot(r, g - 1)
        self.plot_g_line, = self.g_ax.plot(r, r ** (-1 / 3))
        self.plot_g6, = self.g6_ax.plot(r, g6 / g)
        self.plot_g6_line, = self.g6_ax.plot(r, r ** (-1 / 4))

        self.g_ax.set_xlabel('r / pixels')
        self.g_ax.set_ylabel('$G(r)$')
        self.g6_ax.set_xlabel('r / pixels')
        self.g6_ax.set_ylabel('$G_6(r)$')

    def get_data(self, val):
        data = self.data.loc[self.data.d == val, ['r', 'g', 'g6']].values
        r, g, g6 = data.T
        return r[0], g[0], g6[0]

    def get_vals(self, val):
        duty_val = self.duty_slider.val
        g_fit_val = self.g_fit_slider.val
        g6_fit_val = self.g6_fit_slider.val
        g6_power_val = self.g6_power_slider.val
        radio_label = self.radio.value_selected
        plot_dict = {'xy': ['linear', 'linear'], 'logx': ['log', 'linear'],
                     'logy': ['linear', 'log'], 'loglog': ['log', 'log']}
        proj = plot_dict[radio_label]
        self.g_ax.set_xscale(proj[0])
        self.g_ax.set_yscale(proj[1])
        self.g6_ax.set_xscale(proj[0])
        self.g6_ax.set_yscale(proj[1])
        if duty_val in self.duty:
            r, g, g6 = self.get_data(duty_val)
            self.plot_g.set_ydata(g - 1)
            self.plot_g6.set_ydata(g6 / g)
            self.plot_g_line.set_ydata(r ** (-1 / 3) + g_fit_val)
            self.plot_g6_line.set_ydata(r ** (-1 / g6_power_val) + g6_fit_val)
            self.fig.canvas.draw_idle()


if __name__ == "__main__":
    # calculate_corr_data()
    CorrelationViewer()
