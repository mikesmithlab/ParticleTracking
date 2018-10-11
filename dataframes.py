import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import trackpy as tp

class TrackingDataframe:
    """Class to manage dataframes associated with tracking"""

    def __init__(self, filename):
        self.dataframe = pd.DataFrame()
        self.filename = filename

    def add_tracking_data(self, frame, circles):
        frame_list = np.ones((np.shape(circles)[1], 1)) * frame
        dataframe_to_append = pd.DataFrame({
                "x": circles[0, :, 0],
                "y": circles[0, :, 1],
                "size": circles[0, :, 2],
                "frame": frame_list[:, 0]},
                index=np.arange(1, np.shape(circles)[1] + 1))
        self.dataframe = pd.concat([self.dataframe, dataframe_to_append])

    def save_dataframe(self):
        self.dataframe.to_hdf(self.filename, 'w')

    def filter_trajectories(self):
        max_frame_displacement = 10
        min_frame_life = 5
        self.dataframe = tp.link_df(self.dataframe, max_frame_displacement)
        self.dataframe = tp.filter_stubs(self.dataframe, min_frame_life)

if __name__=="__main__":
    dataframe = pd.read_hdf("/home/ppxjd3/Code/ParticleTracking/test_data/test_video.hdf5")
    print(dataframe.head())