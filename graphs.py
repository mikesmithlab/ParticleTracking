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


if __name__ == "__main__":
    plot_shape_factor_histogram("/home/ppxjd3/Videos/liquid_data.hdf5", 0)
