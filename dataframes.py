import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)


class DataStore:
    """Class to manage dataframes associated with tracking"""

    def __init__(self, filename, load=False):
        self.dataframe = pd.DataFrame()
        self.boundary_df = pd.DataFrame()
        self.filename = filename
        if load:
            self._load_dataframe()
            self._find_properties()

    def _find_properties(self):
        self.num_frames = self.dataframe['frame'].max()

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

    def get_info(self, frame_no, include_size=False, prop=None):
        if include_size and prop is None:
            points = self.dataframe.loc[
                self.dataframe['frame'] == frame_no, ['x', 'y', 'size']].values
        elif include_size and prop is not None:
            points = self.dataframe.loc[
                self.dataframe['frame'] == frame_no, ['x', 'y', 'size',
                                                      prop]].values
        elif not include_size and prop is None:
            points = self.dataframe.loc[
                self.dataframe['frame'] == frame_no, ['x', 'y']].values
        else:
            points = self.dataframe.loc[
                self.dataframe['frame'] == frame_no, ['x', 'y',
                                                      prop]].values
        return points

    def add_property(self, prop_string, prop):
        self.dataframe[prop_string] = prop
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

    def get_boundary(self, frame):
        boundary = self.boundary_df.loc[self.boundary_df['frame'] == frame,
                                        'boundary'].values
        return boundary[0]


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
    from Generic import filedialogs
    filename = filedialogs.load_filename('Select a dataframe', directory="/home/ppxjd3/Videos", file_filter='*.hdf5')
    data = DataStore(filename, load=True)
    info = data.get_info(1, include_size=True, prop='order')

