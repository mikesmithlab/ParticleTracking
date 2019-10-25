from Generic import images
import os

import cv2
from ParticleTracking.configurations import BACTERIA2_PARAMETERS as params
from ParticleTracking.annotation import TrackingAnnotator
import numpy as np



class BacteriaAnnotator(TrackingAnnotator):

    def __init__(self, filename, data, parameter, crop=False):
        self.data = data
        self.crop = crop
        self.parameter = parameter
        in_name = filename
        out_name = os.path.splitext(filename)[0]+'_annotated.mp4'
        TrackingAnnotator.__init__(self, in_name, out_name, params)

    def process_frame(self, frame, f):
        print(f)
        annotated_frame = self._draw_boxes(frame, f)
        annotated_frame = self._add_number(annotated_frame, f, colx='x drift', coly='y drift')
        annotated_frame = self._draw_trajs(annotated_frame, f, colx='x drift', coly='y drift')
        return annotated_frame

    def check_crop(self, filename):
        return filename


if __name__ == "__main__":
    from ParticleTracking import dataframes

    file = '/media/ppzmis/data/ActiveMatter/Microscopy/190820bacteriaand500nmparticles/videos/joined/StreamDIC003_trajclassified.hdf5'

    dataframe = dataframes.DataStore(file, load=True)
    file, ext = file.split('_')
    input_video = file + '.mp4'
    annotator = BacteriaAnnotator(input_video, dataframe, parameter='box')
    annotator.annotate(bitrate='HIGH4K', framerate=20.0)