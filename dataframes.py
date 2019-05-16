import os

import numpy as np
import pandas as pd


class DataStore:
    def __init__(self, filename, load=True):
        self.df = pd.DataFrame()
        self.metadata = {}
        self.filename = os.path.splitext(filename)[0] + '.hdf5'
        if load:
            self.load()

    def get_headings(self):
        return self.df.columns.values.tolist()

    def get_column(self, name):
        return self.df[name].values

    def add_particle_property(self, heading, values):
        self.df[heading] = values

    def add_particle_properties(self, headings, values):
        for heading, value in zip(headings, values):
            self.add_particle_property(heading, value)

    def get_info_all_frames(self, headings):
        all_headings = ['frame'] + headings
        data = self.df.reset_index()[all_headings].values
        info = self.stack_info(data)
        return info

    @staticmethod
    def stack_info(arr):
        f = arr[:, 0]
        _, c = np.unique(f, return_counts=True)
        indices = np.insert(np.cumsum(c), 0, 0)
        info = [arr[indices[i]:indices[i + 1], 1:]
             for i in range(len(c))]
        return info

    def add_tracking_data(self, frame, tracked_data, col_names=None):
        col_names = ['x', 'y', 'r'] if col_names is None else col_names
        data_dict = {name: tracked_data[:, i]
                     for i, name in enumerate(col_names)}
        data_dict['frame'] = frame
        new_df = pd.DataFrame(data_dict).set_index('frame')
        self.df = self.df.append(new_df)

    def add_metadata(self, name, data):
        self.metadata[name] = data

    def get_metadata(self, name):
        return self.metadata[name]

    def get_info(self, frame, headings):
        return self.df.loc[frame, headings].values

    def reset_index(self):
        self.df = self.df.reset_index()

    def save(self):
        with pd.HDFStore(self.filename) as store:
            store.put('df', self.df)
            store.get_storer('df').attrs.metadata = self.metadata

    def load(self):
        with pd.HDFStore(self.filename) as store:
            self.df = store.get('df')
            self.metadata = store.get_storer('df').attrs.metadata

    def get_crop(self):
        return self.metadata['crop']

    def set_frame_index(self):
        if 'frame' in self.df.columns.values.tolist():
            if self.df.index.name == 'frame':
               self.df = self.df.drop('frame', 1)
            else:
                self.df = self.df.set_index('frame')

    def add_frame_property(self, heading, values):
        prop = pd.Series(values,
                         index=pd.Index(np.arange(len(values)), name='frame'))
        self.df[heading] = prop

    def add_frame_properties(self, headings, values):
        for heading, value in zip(headings, values):
            self.add_frame_property(heading, value)

    def append_store(self, store):
        self.df = self.df.append(store.df)
        self.metadata = {**self.metadata, **store.metadata}


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