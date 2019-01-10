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



if __name__ == "__main__":
    plot_shape_factor_histogram("/home/ppxjd3/Videos/liquid_data.hdf5", 0)
