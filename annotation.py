import cv2
import ParticleTracking.dataframes as df
import Generic.video as vid
import Generic.images as im
from matplotlib import cm
import numpy as np
import multiprocessing as mp
import subprocess as sub
import os
from PIL import ImageDraw, Image, ImageOps

class VideoAnnotator:

    def __init__(
            self,
            dataframe_inst,
            input_video_filename,
            shrink_factor=1
            ):
        self.td = dataframe_inst
        self.input_video_filename = input_video_filename
        self.core_filename, self.extension = (
                os.path.splitext(self.input_video_filename))
        self.shrink_factor = shrink_factor

    def add_annotations(self, voronoi=False, delaunay=False):
        cap = vid.ReadVideoFFMPEG(self.input_video_filename)
        output_video_filename = (
                self.core_filename +
                '_network' +
                self.extension
                )
        out = vid.WriteVideoFFMPEG(output_video_filename,
                                   bitrate='MEDIUM4K')
        for f in range(cap.num_frames):
            print(f)
            points = self.td.get_info(f)
            frame = cap.read_frame()
            if delaunay:
                frame = im.draw_delaunay_tess(frame, points)
            if voronoi:
                frame = im.draw_voronoi_cells(frame, points)
            out.add_frame(frame)
        out.close()

    def add_coloured_circles(self, parameter=None):
        if parameter is not None:
            output_video_filename = \
                self.core_filename + '_' + self.parameter + self.extension
        else:
            parameter = 'particle'
            output_video_filename = \
                self.core_filename + '_circles' + self.extension

        cap = vid.ReadVideoFFMPEG(self.input_video_filename)
        out = vid.WriteVideoFFMPEG(output_video_filename)
        col = (255, 0, 0)
        for f in range(cap.num_frames):
            print('Annotating frame ', f+1, ' of ', cap.num_frames)
            frame = cap.read_frame_PIL()
            info = self.td.get_info(f, include_size=True, prop=parameter)
            draw = ImageDraw.Draw(frame)
            for xi, yi, r, param in info:
                if parameter == 'particle':
                    draw.ellipse([xi-r, yi-r, xi+r, yi+r],
                                 outline=col, width=5)
                else:
                    col = np.multiply(cm.viridis(param), 255)
                    draw.ellipse([xi-r, yi-r, xi+r, yi+r],
                                 fill=(int(col[0]), int(col[1]), int(col[2])))
            if self.shrink_factor > 1:
                new = frame.resize((((frame.width//self.shrink_factor)//2)*2,
                                    ((frame.height//self.shrink_factor)//2)*2),
                                   resample=Image.BILINEAR)
            else:
                new = frame
            out.add_frame_PIL(new)
        out.close()


def init_circle(r, col=(255, 0, 0), fill=False):
    im = Image.new('RGB', [int(r) * 2] * 2, (255, 255, 255))
    draw = ImageDraw.Draw(im)
    if fill:
        draw.ellipse((1, 1, r * 2 - 1, r * 2 - 1), fill=col)
    else:
        draw.ellipse((1, 1, r * 2 - 1, r * 2 - 1), outline=col, width=5)
    im = im.crop((0, 0) + (r * 2, r * 2))
    mask = ImageOps.invert(im.convert('L'))
    return im, mask


if __name__ == "__main__":

    dataframe = df.DataStore(
            "/home/ppxjd3/Videos/packed_data.hdf5",
            load=True)
    input_video = "/home/ppxjd3/Videos/packed_crop.mp4"
    VA = VideoAnnotator(
            dataframe,
            input_video,
            shrink_factor=1)
    VA.add_coloured_circles()
    # VA.add_annotations(voronoi=True, delaunay=True)
    # image = Image.new('RGB', [1000, 1000], (255, 0, 255))
    # circle, mask = init_circle(20)
    # image.paste(circle.convert('RGB', (0, 255, 0, 0)), (100, 100, 140, 140), mask)
    # image.show()