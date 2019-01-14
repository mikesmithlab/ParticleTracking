from ParticleTracking import dataframes
import numpy as np
import matplotlib.pyplot as plt

def plot_shape_factor_histogram(filename, frame):
    datastore = dataframes.DataStore(filename, load=True)
    shape_factors = datastore.get_info(frame, ['shape factor'])
    n, bins = np.histogram(shape_factors, bins=100)
    plt.figure()
    plt.plot(bins[:-1], n, 'x')
    plt.show()

def polar_histogram(bins, data, y_err=None, filename=None):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    width = bins[1] - bins[0]
    bars = ax.bar(bins[:-1], data, width=width, bottom=0.0, align='edge',
                  yerr=y_err)
    if filename is not None:
        plt.savefig(filename)

class Plotter:

    def __init__(self, nrows=1, ncols=1, figsize=(8, 4)):
        self.nrows = nrows
        self.ncols = ncols
        self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols,
                                         figsize=figsize)
        if (nrows*ncols) == 1:
            self.ax = [self.ax]

        if (nrows > 1) and (ncols > 1):
            self.ax = self.ax.flatten()

    def add_scatter(self, subplot, xdata, ydata):
        self.ax[subplot].errorbar(xdata, ydata)

    def add_bar(self, subplot, xdata, ydata, yerr=None):
        self.ax[subplot].bar(xdata, ydata, yerr=yerr,
                             width=xdata[1]-xdata[0], align='edge')

    def add_polar_bar(self, subplot, xdata, ydata, yerr=None):
        self.ax[subplot] = plt.subplot(self.nrows, self.ncols, subplot+1, polar=True)
        self.ax[subplot].bar(xdata, ydata, yerr=yerr,
                             width=xdata[1]-xdata[0], align='edge')

    def add_hexbin(self, subplot, xdata, ydata, gridsize=100, mincnt=0):
        self.ax[subplot].hexbin(xdata, ydata, gridsize=gridsize, mincnt=mincnt)

    def show(self):
        plt.show()

    def config_axes(self, subplot=0, xlabel=None, ylabel=None, legend=None):
        self.ax[subplot].set_xlabel(xlabel)
        self.ax[subplot].set_ylabel(ylabel)
        if legend is not None:
            self.ax[subplot].legend(legend)

    def save(self, filename):
        self.fig.savefig(filename)

def scatter(xdata, ydata, xerr=None, yerr=None, xlabel=None, ylabel=None,
            filename=None, fmt=None, errorevery=1, capsize=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt=fmt, errorevery=errorevery,
                capsize=capsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if filename is not None:
        plt.savefig(filename)



if __name__ == "__main__":
    plot_shape_factor_histogram("/home/ppxjd3/Videos/liquid_data.hdf5", 0)
