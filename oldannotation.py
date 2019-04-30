import cv2
from ParticleTracking import dataframes
from Generic import video, images
from matplotlib import cm
import numpy as np
import os
import pygame
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
import scipy.spatial as sp
from tqdm import tqdm


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
            crop = self.td.crop
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
        for f in tqdm(range(cap.num_frames), 'network annotations'):
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
        for f in tqdm(range(cap.num_frames), 'Circle annotations'):
            frame = cap.read_frame_bytes()
            surface = pygame.image.fromstring(
                frame, (cap.width, cap.height), 'RGB')
            info = self.td.get_info(f, ['x', 'y', 'r', parameter])

            surface = images.pygame_draw_circles(surface, info)
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


def neighbors(data_store, frame):
    data = data_store.get_info(frame, ['x', 'y', 'size', 'neighbors'])
    rad = data[:, 2].max()
    fig, ax = plt.subplots()
    patches = []
    colors = []
    for x, y, r, n in data:
        patches.append(Circle((x, y), r, linewidth=0))
        colors.append(n)
    voro = sp.Voronoi(data[:, :2])
    ridge_vertices = voro.ridge_vertices
    new_ridge_vertices = []
    for ridge in ridge_vertices:
        if -1 not in ridge:
            new_ridge_vertices.append(ridge)
    polygons = voro.vertices[new_ridge_vertices]
    for line in polygons:
        poly = Polygon(np.array(line), closed=False)
        patches.append(poly)
        colors.append(0)

    p = PatchCollection(patches, alpha=1, cmap=matplotlib.cm.Set1, match_original=True)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    fig.colorbar(p, ax=ax)
    # ax.set_xlim([data[:, 0].min()-rad, data[:, 0].max()+rad])
    # ax.set_ylim([data[:, 1].min()-rad, data[:, 1].max()+rad])
    ax.set_aspect('equal')
    ax.set_axis_off()
    plt.show()


if __name__ == "__main__":
    dataframe = dataframes.DataStore(
        "/home/ppxjd3/Videos/test_video.hdf5", load=True)
    input_video = "/home/ppxjd3/Videos/test_video.mp4"
    # VA = VideoAnnotator(
    #         dataframe,
    #         input_video,
    #         shrink_factor=1)
    # VA.add_coloured_circles('particle')
    # # VA.add_annotations(voronoi=True, delaunay=True)
    neighbors(dataframe, 0)
