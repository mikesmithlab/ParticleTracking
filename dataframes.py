import pandas as pd
import numpy as np
import os


class TrackingDataframe:
    """Class to manage dataframes associated with tracking"""

    def __init__(self, filename, load=False):
        self.dataframe = pd.DataFrame()
        self.boundary_df = pd.DataFrame()
        self.filename = filename
        if load:
            self._load_dataframe()

    def add_tracking_data(self, frame, circles, boundary):
        frame_list = np.ones((np.shape(circles)[0], 1)) * frame
        dataframe_to_append = pd.DataFrame({
                "x": circles[:, 0],
                "y": circles[:, 1],
                "size": circles[:, 2],
                "frame": frame_list[:, 0]},
                index=np.arange(1, np.shape(circles)[0] + 1))
        self.dataframe = pd.concat([self.dataframe, dataframe_to_append])
        boundary_df_to_append = pd.DataFrame({
                "frame": frame,
                "boundary": [boundary]})
        self.boundary_df = pd.concat([self.boundary_df, boundary_df_to_append])

    def extract_points_for_frame(self, frame_no, include_size=False):
        if include_size:
            points = self.dataframe.loc[
                self.dataframe['frame'] == frame_no, ['x', 'y', 'size']].values
        else:
            points = self.dataframe.loc[
                self.dataframe['frame'] == frame_no, ['x', 'y']].values
        return points

    def add_property_to_dataframe(self, property_string, property):
        self.dataframe[property_string] = property
        self.save_dataframe()

    def save_dataframe(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)
        store = pd.HDFStore(self.filename)
        store['data'] = self.dataframe
        store['boundary'] = self.boundary_df
        store.close()

    def _load_dataframe(self):

        store = pd.HDFStore(self.filename)
        self.dataframe = store['data']
        self.boundary_df = store['boundary']
        store.close()

    def extract_boundary_for_frame(self, frame):
        boundary = self.boundary_df.loc[self.boundary_df['frame'] == frame, 'boundary'].values
        return boundary[0]

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
                                     ['x', 'y', 'size']].values
        return circles

    def return_property_and_circles_for_frame(self, frame, dataframe_columns):
        out = self.dataframe.loc[self.dataframe['frame'] == frame,
                                               dataframe_columns].values
        return out


def concatenate_dataframe(dataframe_list, new_filename):
    data_save = pd.DataFrame()
    boundaries_save = pd.DataFrame()
    for file in dataframe_list:
        store = pd.HDFStore(file)
        data = store['data']
        boundaries = store['boundary']
        data_save = pd.concat([data_save, data])
        boundaries_save = pd.concat([boundaries_save, boundaries])
        store.close()

    if os.path.exists(new_filename):
        os.remove(new_filename)
    store_out = pd.HDFStore(new_filename)
    store_out['data'] = data_save
    store_out['boundary'] = boundaries_save
    store_out.close()


if __name__=="__main__":
    filename = "/home/ppxjd3/Videos/12240002_data.hdf5"
    td = TrackingDataframe(filename, load=True)
    circles = td.return_circles_for_frame(5)
    print(np.shape(circles))
