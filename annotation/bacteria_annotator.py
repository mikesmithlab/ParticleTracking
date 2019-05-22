from Generic import video, images
import os
from cv2 import FONT_HERSHEY_PLAIN as font
import cv2
import pygame
import numpy as np


class BacteriaAnnotator(video.Annotator):

    def __init__(self, filename, data, parameter, crop=False):
        self.data = data
        self.parameter = parameter
        self.crop = crop
        in_name = filename
        out_name = os.path.splitext(filename)[0]+'_'+parameter+'.mp4'
        # frames_as_surface - if True returns pygames_surface else returns numpy array.
        video.Annotator.__init__(self, in_name, out_name, frames_as_surface=False)

    def process_frame(self, frame, f):
        annotated_frame = self._draw_boxes(frame, f)
        annotated_frame = self._add_number(annotated_frame, f)

        return annotated_frame

    def _draw_boxes(self, frame, f):
        if self.parameter == 'box':
            info = self.data.get_info(f, ['box', 'classifier'])
            colors = {1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 255, 0)}
            for bacterium in info:
                if bacterium[1] != 0:
                    annotated_frame = images.draw_contours(frame, [
                        bacterium[0]], col=colors[bacterium[1]])
            # for bacterium in info:
            #     if bacterium[1] == 1:
            #         annotated_frame = images.draw_contours(frame, [
            #             bacterium[0]], col=(0, 0, 255))
            #     elif bacterium[1] == 2:
            #         annotated_frame = images.draw_contours(frame, [
            #             bacterium[0]], col=(255, 0, 0))
            #     elif bacterium[1] == 3:
            #         annotated_frame = images.draw_contours(frame, [
            #             bacterium[0]], col=(0, 255, 0))
        return annotated_frame

    def _add_number(self, frame, f):
        colors = {1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 255, 0)}
        x = self.data.get_info(f, 'x')
        y = self.data.get_info(f, 'y')
        particles = self.data.get_info(f, 'particle')
        classifier = self.data.get_info(f, 'classifier')
        for index, particle in enumerate(particles):
            if classifier[index] != 0:
                frame = cv2.putText(frame, str(int(particle)), (int(x[index]), int(y[index])), font, 1, colors[int(classifier[index])], 1, cv2.LINE_AA)
        return frame

    def check_crop(self, filename):
        return filename


if __name__ == "__main__":
    from ParticleTracking import dataframes
    dataframe = dataframes.DataStore(
        '/media/ppzmis/data/ActiveMatter/bacteria_plastic/bacteria.hdf5', load=True)
    input_video = '/media/ppzmis/data/ActiveMatter/bacteria_plastic/bacteria.mp4'
    annotator = BacteriaAnnotator(input_video, dataframe, 'box')
    annotator.annotate()