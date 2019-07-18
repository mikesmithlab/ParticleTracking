import os

import matplotlib.pyplot as plt
import numpy as np

from Generic import filedialogs
from ParticleTracking import dataframes


def create_report():
    directory = filedialogs.open_directory()
    files = filedialogs.get_files_directory(directory+'/*.hdf5')
    total_balls = {}
    for file in files:
        file_short = os.path.splitext(os.path.split(file)[1])[0]
        meta = dataframes.MetaStore(file)
        total_balls[file_short] = meta.metadata['n']
    plt.figure()
    plt.bar(range(len(total_balls)), list(total_balls.values()), align='center')
    plt.xticks(range(len(total_balls)), list(total_balls.keys()))
    plt.ylabel('Total Balls')

if __name__ == "__main__":
    create_report()