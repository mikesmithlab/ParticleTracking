import pandas as pd
import numpy as np
from ParticleTracking.general import dataframes
import trackpy as tp

def subtract_drift(df, classifier=None):
    print('Subtracting mean drift')
    df.index.name = 'None'
    df['frame'] = df.index
    df_subset=select_subset(df, classifier=classifier)
    df_subset.index.name = None
    df_subset['frame'] = df_subset.index
    drift = tp.motion.compute_drift(df_subset)
    corrected_traj = tp.motion.subtract_drift(
    df[['frame', 'x', 'y', 'particle']].copy(), drift)

    df['x drift'] = df['x'].copy()
    df['y drift'] = df['y'].copy()
    df = df.drop(columns=['x', 'y'])
    corrected_traj.index.name = None
    df = pd.merge(df, corrected_traj,
                   on=['particle', 'frame'])
    df.set_index('frame', drop=True, inplace=True)
    df.sort_values(by=['frame', 'particle'], inplace=True)
    num_particles = np.size(df_subset['particle'].unique())
    print('This is based on', num_particles, 'particle trajectories')
    return df

def select_subset(df, classifier=None):
    if classifier is None:
        df_subset = df.copy()
    else:
        df_subset = df[df['classifier'] == classifier].copy()
    df_subset.name=None
    df_subset['frame']=df_subset.index
    return df_subset

def median_classifier(df):
    print('calculating median classification')
    df.index.name = None
    df['frame'] = df.index
    meanvals = df.groupby(by='particle').median()[
        'classifier'].copy().rename('classifier').astype(int)
    df['classifier old'] = df['classifier'].copy()
    df = df.drop(columns='classifier')
    df = pd.merge(df, pd.DataFrame(meanvals),
                             on='particle')
    df.set_index('frame', drop=True, inplace=True)
    df.sort_values(by = ['frame', 'particle'], inplace=True)
    return df

def single_traj(df, particle_num):
    traj = df[df['particle'] == particle_num].copy()
    return traj[['x','y','classifier']]

def traj_jump_classifier(traj, frame_gap, threshold):

    if np.size(traj['x'].values) <=frame_gap:
        #This case only arises when the trajectory has less than frame_gap points
        #and the trajectory has the particle disappearing for a few frames.
        trajx_diffs = traj['x'].values[0] - traj['x'].values[
                                                     :-1]
        trajy_diffs = traj['y'].values[0] - traj['y'].values[
                                                     :-1]
    else:
        trajx_diffs = traj['x'].values[frame_gap:] - traj['x'].values[:-frame_gap]
        trajy_diffs = traj['y'].values[frame_gap:] - traj['y'].values[:-frame_gap]

    displacements = (trajx_diffs**2 + trajy_diffs**2)**0.5

    if np.max(displacements) > threshold:
        classifier = 3
        return classifier
    else:
        classifier = 2
        return classifier

def trajectories_reclassifier(df, frame_gap=8, jump_threshold=15):
    '''
    Assumes, all the current classifiers associated with a trajectory are the same
    Calculate distribution of jump vectors along each trajectory with specified lag
    Reclassify category 2

    :param df: dataframe of all particle trajectories
    :param frame_gap: time between samples in frames
    :param jump_threshold: distance in pixels moved over frame_gap
    :return: dataframe with reclassifications
    '''

    for particle in np.unique(df['particle'].values):
        if np.mean(df[df['particle']==particle]['classifier']) == 2:
            traj = single_traj(df, particle)
            classifier = traj_jump_classifier(traj, frame_gap, jump_threshold)
            if classifier != 2:
                print('changing classification')
                df[df['particle'] == particle]=df[df['particle'] == particle].assign(classifier=classifier)
    return df


if __name__ == '__main__':
    filename = '/media/ppzmis/data/ActiveMatter/Microscopy/190820bacteriaand500nmparticles/videos/joined/StreamDIC003.hdf5'
    data = dataframes.DataStore(filename)
    data.df = trajectories_reclassifier(data.df)
    data.save(filename[:-5] + '_trajclassified.hdf5')