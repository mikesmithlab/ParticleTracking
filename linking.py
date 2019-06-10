from ParticleTracking import dataframes
import trackpy as tp


class Linker:

    def __init__(self, filename):
        self.filename = filename
        self.data = dataframes.DataStore(self.filename)
        self.data.reset_index()

    def link(self, search_range, memory=None):
        self.data.df = tp.link(self.data.df, search_range, memory=memory)

    def filter(self, min_frame_life):
        self.data.df = tp.filter_stubs(min_frame_life)

    def quit(self):
        self.data.set_frame_index()
        self.data.save()