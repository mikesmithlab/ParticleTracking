import cv2
from ParticleTracking import dataframes
from Generic import video, images
from matplotlib import cm
import numpy as np
import os
import pygame


class VideoAnnotator:

    def __init__(
            self,
            dataframe_inst,
            filename,
            shrink_factor=1
            ):
        self.td = dataframe_inst
        self.filename = os.path.splitext(filename)[0]
        self.input_video_filename = self.filename+'_crop.mp4'
        self.check_crop_exists()

        self.shrink_factor = shrink_factor

    def check_crop_exists(self):
        if not os.path.exists(self.input_video_filename):
            crop = np.loadtxt(self.filename+'.txt')
            video.crop_video(self.filename+'.MP4',
                             crop[0][0],
                             crop[1][0],
                             crop[0][1],
                             crop[1][1])

    def add_annotations(self, voronoi=False, delaunay=False):
        cap = video.ReadVideoFFMPEG(self.input_video_filename)
        output_video_filename = (
                self.filename +
                '_network' +
                self.extension
                )
        out = video.WriteVideoFFMPEG(output_video_filename,
                                     bitrate='MEDIUM4K')
        for f in range(cap.num_frames):
            print(f)
            points = self.td.get_info(f, ['x', 'y'])
            frame = cap.read_frame()
            if delaunay:
                frame = images.draw_delaunay_tess(frame, points)
            if voronoi:
                frame = images.draw_voronoi_cells(frame, points)
            out.add_frame(frame)
        out.close()

    def add_coloured_circles(self, parameter=None):
        if parameter is not None:
            output_video_filename = \
                self.filename + '_' + parameter + '.mp4'
        else:
            parameter = 'particle'
            output_video_filename = \
                self.filename + '_circles' + '.mp4'

        cap = video.ReadVideoFFMPEG(self.input_video_filename)
        out = video.WriteVideoFFMPEG(
            output_video_filename, bitrate='HIGH1080')
        col = (255, 0, 0)
        for f in range(cap.num_frames):
            # print('Annotating frame ', f+1, ' of ', cap.num_frames)
            frame = cap.read_frame_bytes()
            surface = pygame.image.fromstring(
                frame, (cap.width, cap.height), 'RGB')
            info = self.td.get_info(f, ['x', 'y', 'size', parameter])

            for xi, yi, r, param in info:
                if parameter == 'particle':
                    pygame.draw.circle(
                        surface, col, (int(xi), int(yi)), int(r), 3)
                else:
                    col = np.multiply(cm.viridis(param), 255)
                    pygame.draw.circle(
                        surface, col, (int(xi), int(yi)), int(r))
            if self.shrink_factor != 1:
                surface = pygame.transform.scale(
                    surface,
                    (cap.width//self.shrink_factor,
                     cap.height//self.shrink_factor))
            frame = pygame.image.tostring(surface, 'RGB')
            out.add_frame_bytes(
                frame,
                cap.width//self.shrink_factor,
                cap.height//self.shrink_factor)

        out.close()


if __name__ == "__main__":

    dataframe = dataframes.DataStore(
            "/home/ppxjd3/Videos/solid_data.hdf5",
            load=True)
    input_video = "/home/ppxjd3/Videos/solid_crop.mp4"
    VA = VideoAnnotator(
            dataframe,
            input_video,
            shrink_factor=1)
    VA.add_coloured_circles('loc_rot_invar')
    # VA.add_annotations(voronoi=True, delaunay=True)
