from Generic import video, images
import os
from cv2 import FONT_HERSHEY_PLAIN as font
import cv2
from ParticleTracking.configurations import BACTERIA2_PARAMETERS as params
from ParticleTracking.preprocessing.preprocessing_methods import adjust_gamma
import pygame
import numpy as np
import trackpy as tp
import matplotlib.pyplot as plt


class BacteriaAnnotator(video.Annotator):

    def __init__(self, filename, data, parameter, crop=False):
        self.data = data
        self.crop = crop
        self.parameter = parameter
        in_name = filename
        out_name = os.path.splitext(filename)[0]+'_annotated.mp4'
        # frames_as_surface - if True returns pygames_surface else returns numpy array.
        video.Annotator.__init__(self, in_name, out_name, frames_as_surface=False)

    def process_frame(self, frame, f):
        annotated_frame = self._draw_boxes(frame, f)
        annotated_frame = self._add_number(annotated_frame, f)
        annotated_frame = self._draw_trajs(annotated_frame, f)
        return annotated_frame

    def _draw_boxes(self, frame, f):
        info = self.data.get_info(f, ['box', 'classifier'])
        colors = params['colors']

        for bacterium in info:
            annotated_frame = images.draw_contours(frame, [
                    bacterium[0]], col=colors[2])#bacterium[1]])
        return annotated_frame

    def _add_number(self, frame, f):
        colors = params['colors']
        x = self.data.get_info(f, 'x')
        y = self.data.get_info(f, 'y')
        particles = self.data.get_info(f, 'particle')
        classifier = self.data.get_info(f, 'classifier')
        for index, particle in enumerate(particles):
            if classifier[index] != 0:
                frame = cv2.putText(frame, str(int(particles[index])), (int(x[index]), int(y[index])), font, 2, colors[int(classifier[index])], 1, cv2.LINE_AA)
        return frame

    def _draw_trajs(self, frame, f):
        particle_ids = self.data.df[self.data.df.index == f]['particle'].unique()
        for index, particle in enumerate(particle_ids):
            single_traj = self.data.df[self.data.df['particle'] == particle]
            traj_pts = single_traj[single_traj.index <= f][['x','y']].values
            if np.shape(traj_pts)[0] > 3:
                frame = cv2.polylines(frame, np.int32([traj_pts]), False, (0, 255, 0), thickness=params.thickness)
        return frame





    def check_crop(self, filename):
        return filename


if __name__ == "__main__":
    from ParticleTracking import dataframes
    dataframe = dataframes.DataStore('/media/ppzmis/data/ActiveMatter/Microscopy/190709MRaggregates/videos/test2.hdf5', load=True)
    input_video = '/media/ppzmis/data/ActiveMatter/Microscopy/190709MRaggregates/videos/test2.mp4'
    annotator = BacteriaAnnotator(input_video, dataframe, parameter='box')
    annotator.annotate()