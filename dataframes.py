import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import trackpy as tp


class TrackingDataframe:
    """Class to manage dataframes associated with tracking"""

    def __init__(self, filename, load=False):
        self.dataframe = pd.DataFrame()
        self.boundary_df = pd.DataFrame()
        self.filename = filename
        if load:
            self._load_dataframe()

    def add_tracking_data(self, frame, circles, boundary):
        frame_list = np.ones((np.shape(circles)[1], 1)) * frame
        dataframe_to_append = pd.DataFrame({
                "x": circles[0, :, 0],
                "y": circles[0, :, 1],
                "size": circles[0, :, 2],
                "frame": frame_list[:, 0]},
                index=np.arange(1, np.shape(circles)[1] + 1))
        self.dataframe = pd.concat([self.dataframe, dataframe_to_append])
        boundary_df_to_append = pd.DataFrame({
                "frame": frame,
                "boundary": [boundary]})
        self.boundary_df = pd.concat([self.boundary_df, boundary_df_to_append])

    def save_dataframe(self):
        store = pd.HDFStore(self.filename)
        store['data'] = self.dataframe
        store['boundary'] = self.boundary_df

    def _load_dataframe(self):
        store = pd.HDFStore(self.filename)
        self.dataframe = store['data']
        self.boundary_df = store['boundary']

    def return_circles_for_frame(self, frame):
        """
        Returns the 'x', 'y', and 'size' data for a certain frame

        Parameters
        ----------
        frame: int

        Returns
        -------
        circles: ndarray
            Array of shape (p, 3) where for each particle p:
             circles[p, 0] contains x coordinates
             circles[p, 1] contains y coordinate
             circles[p, size] contains size
        """
        circles = self.dataframe.loc[self.dataframe['frame'] == frame,
                                     ['x', 'y', 'size']].as_matrix()

        return circles


if __name__=="__main__":
    store = pd.HDFStore("/home/ppxjd3/Code/ParticleTracking/test_data/test_video.hdf5")
    dataframe = store['data']
    circles = dataframe.loc[dataframe['frame'] == 1,
                                 ['x', 'y', 'size']].as_matrix()
    a =1