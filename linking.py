import pandas as pd
import trackpy as tp
import os

class Linker:

    def __init__(self, filename):
        self.filename = filename
        file, ext = os.path.splitext(filename)
        self.new_filename = file + '_new' + ext

    def link(self, search_range, memory=None):
        with pd.HDFStore(self.filename) as old, pd.HDFStore(self.new_filename, 'w') as new:
            data = old.get('df')
            meta = old.get_storer('df').attrs.metadata
            frames = (data.loc[f].reset_index() for f in range(100))
            for linked in tp.link_df_iter(frames, search_range, memory=memory):
                new.append('df', linked.set_index('frame'))
            new.get_storer('df').attrs.metadata = meta


if __name__ == "__main__":
    from Generic import filedialogs
    import time

    file = filedialogs.load_filename()
    linker = Linker(file)
    t = time.time()
    linker.link(15, 3)
    print(time.time() - t)