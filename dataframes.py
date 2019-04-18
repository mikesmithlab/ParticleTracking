import pandas as pd
import numpy as np
import os
import itertools
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

    add_particle_property(heading, values)
        Adds `values` to column `heading`

    save()
        Saves the datastore

    get_boundary(frame)
        Gets the boundary info for a given frame
    """

    def __init__(self, filename, load=False):
        self.particle_data = pd.DataFrame()
        self.boundary_data = pd.DataFrame()
        self.frame_data = pd.DataFrame()
        self.filename = os.path.splitext(filename)[0] + '.hdf5'
        if load:
            self._load()
            self._find_properties()

    def inspect_dataframes(self):
        print('Particle data columns : ')
        print(self.particle_data.columns.values.tolist())
        print('')
        print('Frame data columns : ')
        print(self.frame_data.columns.values.tolist())
        print('')
        print('Particle data head : ')
        print(self.particle_data.head())
        print('')
        print('Frame data head : ')
        print(self.frame_data.head())
        print('')

    def get_headings(self):
        """Returns the headings of `particle_data` as a list"""
        headings = self.particle_data.columns.values.tolist()
        return headings

    def fill_frame_data(self):
        if 'frame' in self.frame_data.columns.values.tolist():
            print('frames already initialised')
        else:
            frame_list = np.arange(0, self.num_frames)
            self.frame_data['frame'] = frame_list
            print('frames initialised')
            self.save()

    def _find_properties(self):
        self.num_frames = int(self.particle_data['frame'].max()+1)

    def add_tracking_data(self, frame, tracked_data, col_names=None):
        """
        Adds initial tracking information to the dataframe

        Parameters
        ----------
        frame : int
            Frame number

        tracked_data : ndarray of shape (N, 3)
            Contains 'x', 'y', and 'size' in each of the 3 columns

        col_names : list of str or None
            List containing titles for each column of tracked data.
            If None will
        """
        col_names = ['x', 'y', 'r'] if col_names is None else col_names
        frame_list = np.ones((np.shape(tracked_data)[0], 1)) * frame
        tracked_data = np.concatenate((tracked_data, frame_list), axis=1)
        col_names.append('frame')

        data_dict = {name: tracked_data[:, i]
                     for i, name in enumerate(col_names)}
        new_particles = pd.DataFrame(
            data_dict, index=np.arange(1, np.shape(tracked_data)[0] + 1))

        self.particle_data = pd.concat([self.particle_data, new_particles])

    def add_boundary_data(self, frame, boundary):
        """
        Adds boundary information to the boundary dataframe.

        Parameters
        ----------
        frame: int
            Frame number

        boundary : ndarray
            Either shape (3,) for circular tray (xc, yc, r)
            Or shape (N, 2) for polygon with N vertices
        """
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

    def add_particle_property(self, heading, values):
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

    def add_frame_property(self, heading, values):
        """
        Adds a new column to the frame dataframe

        Parameters
        ----------
        heading : str
            Name of the new column

        values : ndarray
            Must be the same shape as other columns in the dataframe
        """
        self.frame_data[heading] = values
        self.save()

    def save(self):
        """Saves the dataframe, overwrites if exists"""
        if os.path.exists(self.filename):
            os.remove(self.filename)
        store = pd.HDFStore(self.filename)
        store['data'] = self.particle_data
        store['boundary'] = self.boundary_data
        store['frame'] = self.frame_data
        store.close()

    def _load(self):
        store = pd.HDFStore(self.filename)
        self.particle_data = store['data']
        self.boundary_data = store['boundary']
        try:
            self.frame_data = store['frame']
            print('Using stored frame_data')
        except KeyError as error:
            print(error)
            print('Using empty frame_data')
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
    frame_save = pd.DataFrame()
    for file in datastore_list:
        store = pd.HDFStore(file)
        data = store['data']
        boundaries = store['boundary']
        frame_data = store['frame']
        data_save = pd.concat([data_save, data])
        boundaries_save = pd.concat([boundaries_save, boundaries])
        frame_save = pd.concat([frame_save, frame_data])
        store.close()

    if os.path.exists(new_filename):
        os.remove(new_filename)
    store_out = pd.HDFStore(new_filename)
    store_out['data'] = data_save
    store_out['boundary'] = boundaries_save
    store_out['frame'] = frame_save
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


class CorrData:
    """Dataframe to save data from spatial and orientational correlations"""

    def __init__(self, filename):
        self.filename = os.path.splitext(filename)[0]+'_corr.hdf5'
        if os.path.exists(self.filename):
            self.exists = True
            self.load()
        else:
            self.exists = False
            self.df = None

    def load(self):
        self.df = pd.read_hdf(self.filename)

    def save(self):
        self.df.to_hdf(self.filename, 'df')

    def head(self):
        return self.df.head()

    def tail(self):
        return self.df.tail()

    def add_row(self, data, frame, label):
        if self.exists is False:
            self.df = self._add_row(data, frame, label)
            self.exists = True
        elif self.exists is True:
            if (frame, label) in self.df.index.tolist():
                self.df.loc[frame, label] = data
            else:
                self.df = self.df.append(self._add_row(data, frame, label))
        self.save()

    def _add_row(self, data, frame, label):
        index = [(frame, label)]
        headr = list(np.arange(0, len(data)))
        indx = pd.MultiIndex.from_tuples(index, names=['frame', 'data'])
        cols = pd.Index(headr)
        df = pd.DataFrame([data], indx, cols)
        return df

    def get_row(self, frame, label):
        return self.df.loc[frame, label].values

if __name__ == "__main__":
    # from Generic import filedialogs
    # filename = filedialogs.load_filename(
    #     'Select a dataframe',
    #     directory="/home/ppxjd3/Videos",
    #     file_filter='*.hdf5')
    # # data = DataStore(filename, load=True)
    # # data.inspect_dataframes()
    data1 = np.arange(0, 10)
    corr = CorrData('test')
    corr.add_row(data1, 0, 'a')
    corr.add_row(data1 ** 2, 0, 'b')
    corr.add_row(data1 * 3 + 1, 1, 'a')
    print(corr.df.head())