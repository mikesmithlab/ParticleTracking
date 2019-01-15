import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)


class DataStore:
    """
    Manages datastore containing particle and boundary info

    Attributes
    ----------
    particle_data : pandas dataframe
        Contains info on particles positions and properties

    boundary_data : pandas dataframe
        Contains info on the boundaries for each frame

    filename : str
        Location of the datastore

    num_frames: int
        The number of frames in `particle_data`

    Methods
    -------
    add_tracking_data(frame, circles, boundary)
        Adds the initial tracking information for each frame

    get_info(frame_no, headings)
        Gets the info defined by headings for frame_no

    get_column(column)
        Gets the column defined by the string `column` from `particle_data`

    add_property(heading, values)
        Adds `values` to column `heading`

    save()
        Saves the datastore

    get_boundary(frame)
        Gets the boundary info for a given frame
    """

    def __init__(self, filename, load=False):
        self.particle_data = pd.DataFrame()
        self.boundary_data = pd.DataFrame()
        self.filename = filename
        if load:
            self._load()
            self._find_properties()

    def get_headings(self):
        """Returns the headings of `particle_data` as a list"""
        headings = self.particle_data.columns.values.tolist()
        return headings

    def _find_properties(self):
        self.num_frames = self.particle_data['frame'].max()

    def add_tracking_data(self, frame, circles, boundary):
        """
        Adds initial tracking information to the dataframe

        Parameters
        ----------
        frame : int
            Frame number

        circles : ndarray of shape (N, 3)
            Contains 'x', 'y', and 'size' in each of the 3 columns

        boundary : ndarray
            Either shape (3,) or (N, 2) depending on shape of tray
        """
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

    def get_info(self, frame, headings):
        """
        Returns info as an ndarray for a frame

        Parameters
        ----------
        frame : int

        headings : list of str
            List containing headings of desired columns

        Returns
        -------
        info : ndarray
            Contains information specified by headings for all the
            points in frame
        """
        info = self.particle_data.loc[
            self.particle_data['frame'] == frame, headings].values
        return info

    def get_column(self, name):
        """
        Returns all the values from a column

        Parameters
        ----------
        name: str
            Heading of desired column

        Returns
        -------
        column : ndarray
            All the values in the column
        """
        column = self.particle_data[name].values
        return column

    def add_property(self, heading, values):
        """
        Adds a new column to the dataframe

        Parameters
        ----------
        heading : str
            Name of the new column

        values : ndarray
            Must be the same shape as other columns in the dataframe
        """
        self.particle_data[heading] = values
        self.save()

    def save(self):
        """Saves the dataframe, overwrites if exists"""
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
        """
        Returns the boundary for a frame

        Parameters
        ----------
        frame : int

        Returns
        -------
        boundary[0] : array_like
            Either (3,) shape array containing x, y, and r for circle
            or (N, 2) shape array containing (x, y) for N vertices
        """
        boundary = self.boundary_data.loc[
            self.boundary_data['frame'] == frame, 'boundary'].values
        return boundary[0]


def concatenate_datastore(datastore_list, new_filename):
    """
    Concatenates datastores in a list

    Parameters
    ----------
    datastore_list : list of str
        Contains the filenames of the datastores to be concatenated

    new_filename : str
        Destination of new datastore
    """
    data_save = pd.DataFrame()
    boundaries_save = pd.DataFrame()
    for file in datastore_list:
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


class PlotData:

    def __init__(self, filename):
        self.filename = filename
        if os.path.exists(filename):
            self.load()
        else:
            self.df = pd.DataFrame()
            self.df.to_hdf(self.filename, 'df')

    def load(self):
        self.df = pd.read_hdf(self.filename, index_col=0)

    def add_column(self, name, data):
        if name in self.df:
            temp_df = self.df.drop(name, axis=1)
        else:
            temp_df = self.df
        new_df = pd.DataFrame({name: data})
        self.df = pd.concat([temp_df, new_df], axis=1)
        self.df.to_hdf(self.filename, 'df')

    def read_column(self, name):
        data = self.df[name].values
        data = data[~np.isnan(data)]
        return data

if __name__ == "__main__":
    # from Generic import filedialogs
    # filename = filedialogs.load_filename(
    #     'Select a dataframe',
    #     directory="/home/ppxjd3/Videos",
    #     file_filter='*.hdf5')
    # data = DataStore(filename, load=True)
    # info = data.get_info(1, include_size=True, prop='order')

    filename = 'test2.csv'
    plot_data = PlotData(filename)
    # data1 = np.arange(5, 50, 10)
    # plot_data.add_column('col1', data1)
    data2 = np.arange(30, 39)
    plot_data.add_column('col2', data2)
    print(plot_data.df.head())

