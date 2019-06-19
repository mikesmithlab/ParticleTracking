import numpy as np


def histogram(data, frames, column, bins):
    data = data.loc[frames][column]
    counts, bins = zip(*data.groupby('frame')
                       .apply(np.histogram, bins=bins)
                       .values)
    counts, bins = np.array(counts), np.array(bins)
    return counts, bins
