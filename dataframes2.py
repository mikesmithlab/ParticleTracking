import os

import numpy as np
import pandas as pd


class DataStore:
    def __init__(self, filename, load=True):
        self.particle_data = pd.DataFrame()
        self.crop = []
        self.boundary = []
        self.filename = os.path.splitext(filename)[0] + '.hdf5'
        if load:
            self.load()

    def get_headings(self):
        return self.particle_data.columns.values.tolist()

    def get_column(self, name):
        return self.particle_data[name].values

    def add_particle_property(self, heading, values):
        self.particle_data[heading] = values

    def add_particle_properties(self, headings, values):
        for heading, value in zip(headings, values):
            self.add_particle_property(heading, value)

    def get_info_all_frames(self, headings):
        all_headings = ['frame'] + headings
        data = self.particle_data.reset_index()[all_headings].values
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
        self.particle_data = self.particle_data.append(new_df)

    def add_boundary_data(self, frame, boundary):
            self.boundary = boundary

    def get_info(self, frame, headings):
        return self.particle_data.loc[frame, headings].values

    def reset_index(self):
        self.particle_data = self.particle_data.reset_index()
        print(self.particle_data.head())

    def save(self):
        with pd.HDFStore(self.filename) as store:
            store.put('df', self.particle_data)
            store.get_storer('df').attrs.crop = self.crop
            store.get_storer('df').attrs.boundary = self.boundary

    def load(self):
        with pd.HDFStore(self.filename) as store:
            self.particle_data = store.get('df')
            self.crop = store.get_storer('df').attrs.crop
            self.boundary = store.get_storer('df').attrs.boundary

    def add_crop(self, crop):
        self.crop = crop

    def set_frame_index(self):
        if 'frame' in self.particle_data.columns.values.tolist():
            if self.particle_data.index.name == 'frame':
               self.particle_data = self.particle_data.drop('frame', 1)
            else:
                self.particle_data = self.particle_data.set_index('frame')

    def add_frame_property(self, heading, values):
        prop = pd.Series(values,
                         index=pd.Index(np.arange(len(values)), name='frame'))
        self.particle_data[heading] = prop

    def add_frame_properties(self, headings, values):
        for heading, value in zip(headings, values):
            self.add_frame_property(heading, value)

    def append_store(self, store):
        self.particle_data = self.particle_data.append(store.particle_data)
        self.crop = store.crop
        self.boundary = store.boundary

    def get_boundary(self):
        return self.boundary


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
    print(DS.particle_data.head())
    print(DS.particle_data.tail())