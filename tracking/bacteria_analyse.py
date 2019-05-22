import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import trackpy as tp









if __name__ == '__main__':

    filename = '/media/ppzmis/data/ActiveMatter/bacteria_plastic/bacteria.hdf5'

    df = pd.read_hdf(filename)

    smoothing = 25



    traj = df[df['particle'] == 22]

    plt.figure()
    plt.plot(traj['x'], traj['y'], 'rx')

    traj = traj.rolling(smoothing, win_type='bartlett').mean()

    plt.plot(traj['x'], traj['y'], 'b-')
    plt.show()





