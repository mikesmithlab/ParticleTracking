from ParticleTracking import dataframes
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.optimize as optimize
from Generic import filedialogs, fitting


## MOVE REMAINING METHODS INTO EXPERIMENT SCRIPTS

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
        plotter.add_scatter(0, r, (max(g)/0.6)*np.exp(-r/2))
        plotter.config_axes(0, xlabel='r/$\sigma$', ylabel='$G_6(r)$', xscale='log', yscale='log')
        plotter.show()


def plot_shape_factor_histogram(filename, frame):
    datastore = dataframes.DataStore(filename, load=True)
    shape_factors = datastore.get_info(frame, ['shape factor'])
    n, bins = np.histogram(shape_factors, bins=100)
    plt.figure()
    plt.plot(bins[:-1], n, 'x')
    plt.show()


def exponential(x, b):
    return (max(x)/np.exp(1))*np.exp(b*x)

if __name__ == "__main__":
    # plot_shape_factor_histogram("/home/ppxjd3/Videos/liquid_data.hdf5", 0)
    # filename = filedialogs.load_filename('Load a plotting dataframe', remove_ext=True)
    # filename = "/home/ppxjd3/Videos/down/down.MP4"
    # filename = "/home/ppxjd3/Videos/Solid/grid"
    # # filename = "/home/ppxjd3/Videos/grid/grid_plot_data.hdf5"
    # fig_maker = FigureMaker(filename)
    # # fig_maker.plot_level_checks()
    # fig_maker.plot_orientational_correlation()
    # correlation_fitter(filename)
    # plot_correlations(filename, 1)
    graph = Plotter(2, 2)
    x = np.arange(1, 10)
    y = x * 2
    graph.add_scatter(0, x, y)
    graph.add_bar(1, x, y)
    graph.show()