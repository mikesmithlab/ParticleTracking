import cv2
import numpy as np
from ParticleTracking.general.dataframes import DataStore
from ParticleTracking.postprocessing import postprocessing_methods as pm

class PostProcessor:
    def __init__(self, parameters=None, data_filename=None):
        self.parameters = parameters
        self.data_filename = data_filename
        self.data = DataStore(self.data_filename, load=True)

    def process(self):
        for method in self.parameters['postprocessor method']:
            self.data = getattr(pm, method)(self.data, parameters=self.parameters)
        self.data.save()
