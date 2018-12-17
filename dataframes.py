import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)


class DataStore:
    """Class to manage dataframes associated with tracking"""

    def __init__(self, filename, load=False):
        self.particle_data = pd.DataFrame()
        self.boundary_data = pd.DataFrame()
        self.filename = filename
        if load:
            self._load()
            self._find_properties()

    def get_headings(self):
        headings = self.particle_data.columns.values.tolist()
        return headings

    def _find_properties(self):
        self.num_frames = self.particle_data['frame'].max()

    def add_tracking_data(self, frame, circles, boundary):
        frame_list = np.ones((np.shape(circles)[0], 1)) * frame
        new_particles = pd.DataFrame({
                "x": circles[:, 0],
                "y": circles[:, 1],
                "size": circles[:, 2],
                "frame": frame_list[:, 0]},
                index=np.arange(1, np.shape(circles)[0] + 1))
        self.particle_data = pd.concat([self.particle_data, new_particles])
        new_boundary = pd.DataFrame({
                "frame": frame, "boundary": [boundary]})
        self.boundary_data = pd.concat([self.boundary_data, new_boundary])

    def get_info(self, frame_no, headings):
        info = self.particle_data.loc[
            self.particle_data['frame'] == frame_no, headings].values
        return info

    def get_column(self, column_name):
        column = self.particle_data[column_name].values
        return column

    def add_property(self, prop_string, prop):
        self.particle_data[prop_string] = prop
        self.save()

    def save(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)
        store = pd.HDFStore(self.filename)
        store['data'] = self.particle_data
        store['boundary'] = self.boundary_data
        store.close()

    def _load(self):
        store = pd.HDFStore(self.filename)
        self.particle_data = store['data']
        self.boundary_data = store['boundary']
        store.close()

    def get_boundary(self, frame):
        boundary = self.boundary_data.loc[
            self.boundary_data['frame'] == frame, 'boundary'].values
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
    filename = filedialogs.load_filename(
        'Select a dataframe',
        directory="/home/ppxjd3/Videos",
        file_filter='*.hdf5')
    data = DataStore(filename, load=True)
    info = data.get_info(1, include_size=True, prop='order')

