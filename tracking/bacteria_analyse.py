import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import trackpy as tp
from ParticleTracking import dataframes









if __name__ == '__main__':

    filename = '/media/ppzmis/data/ActiveMatter/bacteria_plastic/bacteria.hdf5'

    data_store = dataframes.DataStore(filename)
    df1 = data_store.df

    print(df1.columns.values)


    print(df1.head())

    traj = df1[df1['particle'] == 22]

    print(traj.head())

    plt.figure()
    plt.plot(traj['x raw'],traj['y raw'],'rx')
    plt.plot(traj['x'],traj['y'],'b.')
    plt.show()






