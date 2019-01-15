from ParticleTracking import dataframes
import numpy as np
import matplotlib.pyplot as plt

class FigureMaker:

    def __init__(self, filename):
        self.plot_data = dataframes.PlotData(filename)

    def plot_level_checks(self):
        plotter = Plotter(2, 2, (8, 8))

        # data
        mean_x = self.plot_data.read_column('mean x pos')
        # mean_x_err = self.plot_data.read_column('mean x pos err')
        mean_y = self.plot_data.read_column('mean y pos')
        # mean_y_err = self.plot_data.read_column('mean y pos err')
        frames = self.plot_data.read_column('mean pos frames')

        plotter.add_scatter(0, frames, mean_x)
        plotter.add_scatter(0, frames, mean_y)
        plotter.config_axes(0, xlabel='frame', ylabel='Distance / $\sigma$',
                            legend=['x', 'y'], title='a')

        dist_bins = self.plot_data.read_column('mean dist bins')
        dist_N = self.plot_data.read_column('mean dist hist')
        dist_N_err = self.plot_data.read_column('mean dist hist err')
        plotter.add_bar(1, xdata=dist_bins[:-1], ydata=dist_N, yerr=dist_N_err)
        plotter.config_axes(1, xlabel='Distance / $\sigma$',
                            ylabel='Frequency', title='b')

        angle_bins = self.plot_data.read_column('angle dist bins')
        angle_means = self.plot_data.read_column('angle dist hist')
        angle_err = self.plot_data.read_column('angle dist hist err')
        plotter.config_axes(2, title='c')
        plotter.add_polar_bar(2, angle_bins[:-1], angle_means, angle_err)

        hex_x = self.plot_data.read_column('hexbin x')
        hex_y = self.plot_data.read_column('hexbin y')

        plotter.add_hexbin(3, hex_x, hex_y)
        plotter.config_axes(3, xlabel='x/$\sigma$', ylabel='y/$\sigma$', title='d')

        plotter.config_subplots(wspace=0.4, hspace=0.3)
        plotter.show()

    def plot_orientational_correlation(self):
        r = self.plot_data.read_column('orientation_correlation_1_r')
        g = self.plot_data.read_column('orientation correlation_1_g')

        plotter = Plotter()
        plotter.add_scatter(0, r, g)
        plotter.add_scatter(0, r, max(g)*r**(-1/4))
        plotter.config_axes(0, xlabel='r/$\sigma$', ylabel='$G_6(r)$', xscale='log', yscale='log')
        plotter.show()

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

    def config_axes(self, subplot=0, xlabel=None, ylabel=None, legend=None, title=None, xscale='linear', yscale='linear'):
        self.ax[subplot].set_xlabel(xlabel)
        self.ax[subplot].set_ylabel(ylabel)
        if legend is not None:
            self.ax[subplot].legend(legend)
        self.ax[subplot].set_title(title)
        self.ax[subplot].set_xscale(xscale)
        self.ax[subplot].set_yscale(yscale)

    def config_subplots(self, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None):
        self.fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

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
    # plot_shape_factor_histogram("/home/ppxjd3/Videos/liquid_data.hdf5", 0)
    filename = "/home/ppxjd3/Videos/grid/grid_plot_data.hdf5"
    fig_maker = FigureMaker(filename)
    # fig_maker.plot_level_checks()
    fig_maker.plot_orientational_correlation()