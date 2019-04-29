import os
from Generic import video, images
from tqdm import tqdm
import pygame
import numpy as np
from matplotlib import cm


class VideoAnnotator:

    def __init__(self, filename, use_crop=True, frame_type='array'):
        self.filename = os.path.split(filename)[0]
        if use_crop:
            self.input_video_filename = self.filename + '_crop.mp4'
            self.check_crop_exists()
        else:
            self.input_video_filename = self.filename
        self.frame_type = frame_type

    def check_crop_exists(self):
        if not os.path.exists(self.input_video_filename):
            crop = self.td.crop
            video.crop_video(self.filename + '.MP4',
                             crop[0][0],
                             crop[1][0],
                             crop[0][1],
                             crop[1][1])

    def annotate(self):
        cap = video.ReadVideoFFMPEG(self.input_video_filename)
        out = video.WriteVideoFFMPEG(self.output_video_filename,
                                     bitrate='HIGH1080')
        for f in tqdm(range(cap.num_frames), 'Annotate'):
            if self.frame_type == 'array':
                frame = cap.read_frame()
            elif self.frame_type == 'surface':
                frame = cap.read_frame_bytes()
                frame = pygame.image.fromstring(
                    frame, (cap.width, cap.height), 'RGB')

            frame = self.process_frame(frame, f)

            if self.frame_type == 'array':
                out.add_frame(frame)
            elif self.frame_type == 'surface':
                frame = pygame.image.tostring(frame, 'RGB')
                out.add_frame_bytes(
                    frame,
                    cap.width,
                    cap.height)
        out.close()

    def process_frame(self, frame, f):
        return frame


class JamesAnnotateCircle(VideoAnnotator):

    def __init__(self, filename, data, parameter):
        self.data = data
        self.parameter = parameter
        VideoAnnotator.__init__(filename, frame_type='surface')
        self.output_filename = self.filename + '_particles.mp4'

    def process_frame(self, frame, f):
        info = self.data.get_info(f, ['x', 'y', 'r', self.parameter])
        col = (255, 0, 0)
        for xi, yi, r, param in info:
            if self.parameter == 'particle':
                pygame.draw.circle(frame, col, (int(xi), int(yi), int(r), 3))
            else:
                col = np.multiply(cm.viridis(param), 255)
                pygame.draw.circle(
                    frame, col, (int(xi), int(yi)), int(r))
        return frame