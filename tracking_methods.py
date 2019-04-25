from ParticleTracking.tracking import ParticleTracker
from ParticleTracking import configurations
from ParticleTracking import preprocessing
import os


class JamesPT(ParticleTracker):

    def __init__(self, filename, tracking=False, multiprocess=False):
        self.filename = os.path.splitext(filename)[0]
        self.parameters = configurations.NITRILE_BEADS_PARAMETERS
        self.ip = preprocessing.Preprocessor(self.parameters)
        self.exp = 'James'
        self.input_filename = filename
        if tracking:
            ParticleTracker.__init__(self, multiprocess=multiprocess)
