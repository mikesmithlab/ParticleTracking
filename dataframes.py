import os

import numpy as np
import pandas as pd


class DataStore:
    """
    Manages HDFStore containing particle data and metadata

    Attributes
    ----------
    df : pandas dataframe
        Contains info on particle positions and properties.
        Index of dataframe is the video frame.

    metadata : dict
        Dictionary containing any metadata values.
    """
    def __init__(self, filename, load=True):
        self.df = pd.DataFrame()
        self.metadata = {}
        self.filename = os.path.splitext(filename)[0] + '.hdf5'
        if load:
            self.load()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()
        del self.df, self.metadata

    def add_frame_property(self, heading, values):
        """
        Add data for each frame.

        Parameters
        ----------
        heading: str
            title of dataframe column

        values: arraylike
            array of values with length = num_frames
        """
        prop = pd.Series(values,
                         index=pd.Index(np.arange(len(values)), name='frame'))
        self.df[heading] = prop

    def add_metadata(self, name, data):
        """
        Add metadata to store.

        Parameters
        ----------
        name: str
            string key for dictionary

        data: Any
            Anything that can be saved as a dictionary item
        """
        self.metadata[name] = data

    def add_particle_property(self, heading, values):
        """
        Add properties for each particle in the dataframe

        Parameters
        ----------
        heading: str
            Title of dataframe column

        values: arraylike
            Array of values with same length as dataframe

        """
        self.df[heading] = values

    def add_tracking_data(self, frame, tracked_data, col_names=None):
        """
        Add tracked data for each frame.

        Parameters
        ----------
        frame: int
            Frame number

        tracked_data: arraylike
            (N, D) shape array of N particles with D properties

        col_names: list of str
            Titles of each D properties for dataframe columns
        """
        if isinstance(tracked_data, pd.DataFrame):
            tracked_data['frame'] = frame
            print(tracked_data.dtypes)
            self.df = self.df.append(tracked_data.set_index('frame'))
        else:
            if isinstance(tracked_data, np.ndarray):
                col_names = ['x', 'y', 'r'] if col_names is None else col_names
                data_dict = {name: tracked_data[:, i]
                             for i, name in enumerate(col_names)}

            elif isinstance(tracked_data, list):
                data_dict = {name: tracked_data[i]
                             for i, name in enumerate(col_names)}

            else:
                print('type wrong')
            data_dict['frame'] = frame
            new_df = pd.DataFrame(data_dict).set_index('frame')
            self.df = self.df.append(new_df)

    def append_store(self, store):
        """
        Append an instance of this class to itself.

        Parameters
        ----------
        store: seperate instance of this class
        """
        self.df = self.df.append(store.df)
        self.metadata = {**self.metadata, **store.metadata}

    def get_column(self, name):
        return self.df[name].values

    def get_headings(self):
        """
        Get dataframe headings

        Returns
        -------
        list of dataframe column titles
        """
        return self.df.columns.values.tolist()

    def get_info(self, frame, headings):
        """
        Get information on particles in a particular frame.

        Parameters
        ----------
        frame: int

        headings: list of str
            Titles of dataframe columns to be returned
        """
        return self.df.loc[frame, headings].values

    def get_info_all_frames(self, headings):
        """
        Get info from all frames stacked into list of lists.

        Parameters
        ----------
        headings : list of str
            Dataframe columns
        """
        all_headings = ['frame'] + headings
        data = self.df.reset_index()[all_headings].values
        info = self.stack_info(data)
        return info

    def get_info_all_frames_generator(self, headings):
        for f in range(self.num_frames):
            yield self.df.loc[f, headings].values

    def get_metadata(self, name):
        """
        Return item from the metadata dictionary
        Parameters
        ----------
        name : str
            metadata dictionary key

        Returns
        -------
        dictionary item for given key
        """
        return self.metadata[name]

    def load(self):
        """Load HDFStore"""
        with pd.HDFStore(self.filename) as store:
            self.df = store.get('df')
            self.metadata = store.get_storer('df').attrs.metadata
        self.num_frames = max(self.df.index.values)+1

    def reset_index(self):
        """Move frame index to column"""
        self.df = self.df.reset_index()

    def save(self):
        """Save HDFStore"""
        with pd.HDFStore(self.filename) as store:
            store.put('df', self.df)
            store.get_storer('df').attrs.metadata = self.metadata

    def set_frame_index(self):
        """Move frame column to index"""
        if 'frame' in self.df.columns.values.tolist():
            if self.df.index.name == 'frame':
               self.df = self.df.drop('frame', 1)
            else:
                self.df = self.df.set_index('frame')

    @staticmethod
    def stack_info(arr):
        f = arr[:, 0]
        _, c = np.unique(f, return_counts=True)
        indices = np.insert(np.cumsum(c), 0, 0)
        info = [arr[indices[i]:indices[i + 1], 1:]
             for i in range(len(c))]
        return info


class MetaStore:
    def __init__(self, filename):
        self.filename = filename
        self.metadata = self.load(filename)

    @staticmethod
    def load(filename):
        with pd.HDFStore(filename) as store:
            metadata = store.get_storer('df').attrs.metadata
        return metadata

    def add_metadata(self, metadata):
        self.metadata.update(metadata)

    def save(self):
        with pd.HDFStore(self.filename) as store:
            store.get_storer('df').attrs.metadata = self.metadata




def concatenate_datastore(datastore_list, new_filename):
    DS_out = DataStore(new_filename, load=False)
    for file in datastore_list:
        DS = DataStore(file, load=True)
        DS_out.append_store(DS)
    DS_out.save()






if __name__ == "__main__":
    from Generic import filedialogs
    file = filedialogs.load_filename()
    DS = DataStore(file)
    print(DS.df.head())
    print(DS.df.tail())
    print(DS.metadata)
    print(DS.df.dtypes)