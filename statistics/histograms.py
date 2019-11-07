import numpy as np


def histogram(data, frames, column, bins):
    data = data.loc[frames][column]
    counts, bins = zip(*data.groupby('frame')
                       .apply(np.histogram, bins=bins)
                       .values)
    counts, bins = np.array(counts), np.array(bins)
    return counts, bins

if __name__ == "__main__":
    from Generic import filedialogs
    from ParticleTracking import statistics
    from ParticleTracking.general import dataframes
    import matplotlib.pyplot as plt
    file = filedialogs.load_filename()
    data = dataframes.DataStore(file, load=True)
    calc = statistics.PropertyCalculator(data)

    #%%
    counts, bins = calc.histogram([1, 2, 3], 'order_r', np.arange(-1, 1.01, 0.01))

    #%%
    plt.figure()
    plt.plot(bins[:, :-1].transpose(), counts.transpose(), '.')
