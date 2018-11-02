import cv2
import ParticleTracking.dataframes as dataframes
import Generic.video as video
import Generic.images as im
from matplotlib import cm
import numpy as np
import multiprocessing as mp
import subprocess as sp
import os
import scipy.spatial as spatial
import time

class VideoAnnotator:

    def __init__(self,
                 dataframe_inst,
                 input_video_filename,
                 output_video_filename,
                 shrink_factor=1,
                 multiprocess=True):
        self.td = dataframe_inst
        self.input_video_filename = input_video_filename
        self.output_video_filename = output_video_filename
        self.shrink_factor = shrink_factor
        self.multiprocess = multiprocess

    def add_delaunay_network(self):
        cap = video.ReadVideo(self.input_video_filename)
        out = video.WriteVideo(self.output_video_filename,
                               (cap.height, cap.width, 3),
                               codec='mp4v')
        for f in range(cap.num_frames):
            print(f)
            points = self.td.extract_points_for_frame(f)
            frame = cap.read_next_frame()
            tess = spatial.Delaunay(points)
            frame = im.draw_triangles(frame, points[tess.simplices])
            out.add_frame(frame)
        cap.close()
        out.close()

    def add_coloured_circles(self, parameter=None):
        self.parameter = parameter
        if not parameter:
            self.dataframe_columns = ['x', 'y', 'size', 'particle']
            self.circle_type = 2
        else:
            self.dataframe_columns = ['x', 'y', 'size', parameter]
            self.circle_type = -1

        if self.multiprocess:
            self._add_coloured_circles_multi()
        else:
            self._add_coloured_circles_process(0)

    def _add_coloured_circles_multi(self):
        self.num_processes = mp.cpu_count()
        self.extension = "mp4"
        self.fourcc = "mp4v"

        p = mp.Pool(self.num_processes)
        p.map(self._add_coloured_circles_process, range(self.num_processes))

        self._cleanup_intermediate_videos()

    def _add_coloured_circles_process(self, group_number):
        cap = video.ReadVideo(self.input_video_filename)
        self._find_video_info()
        if self.multiprocess:
            frame_no_start = self.frame_jump_unit * group_number
            cap.set_frame(frame_no_start)
            write_name = "{}.{}".format(group_number, self.extension)
            if group_number + 1 == self.num_processes:
                self.frame_jump_unit += self.remainder
        else:
            write_name = self.output_video_filename

        out = video.WriteVideo(write_name,
                               frame_size=(self.height, self.width, 3),
                               codec=self.fourcc,
                               fps=self.fps)
        col = (0, 255, 255)
        proc_frames = 0
        for f in range(int(self.frame_jump_unit)):
            frame = cap.read_next_frame()
            info = self.td.return_property_and_circles_for_frame(
                f, self.dataframe_columns)
            for xi, yi, r, param in info:
                if self.parameter:
                    col = np.multiply(cm.jet(param)[0:3], 255)
                cv2.circle(frame,
                           (int(xi), int(yi)),
                           int(r),
                           (col[0], col[1], col[2]),
                           self.circle_type)
            if self.shrink_factor is not 1:
                frame = cv2.resize(frame,
                                   None,
                                   fx=1 / self.shrink_factor,
                                   fy=1 / self.shrink_factor,
                                   interpolation=cv2.INTER_CUBIC)
            proc_frames += 1
            out.add_frame(frame)
        cap.close()
        out.close()

    def _cleanup_intermediate_videos(self):
        intermediate_files = ["{}.{}".format(i, self.extension)
                              for i in range(self.num_processes)]
        with open("intermediate_files.txt", "w") as f:
            for t in intermediate_files:
                f.write("file {} \n".format(t))
        ffmepg_command = "ffmpeg -y -loglevel error -f concat " \
                         "-safe 0 -i intermediate_files.txt "
        ffmepg_command += " -vcodec copy "
        ffmepg_command += self.output_video_filename
        sp.Popen(ffmepg_command, shell=True).wait()

        for f in intermediate_files:
            os.remove(f)
        os.remove("intermediate_files.txt")

    def _find_video_info(self):
        input_video = video.ReadVideo(self.input_video_filename)
        if self.multiprocess:
            self.frame_jump_unit = input_video.num_frames // self.num_processes
            self.remainder = input_video.num_frames % self.num_processes
        else:
            self.frame_jump_unit = input_video.num_frames
        self.fps = input_video.fps
        self.width = int(input_video.width/self.shrink_factor)
        self.height = int(input_video.height/self.shrink_factor)


if __name__ == "__main__":

    dataframe = dataframes.TrackingDataframe(
            "/home/ppxjd3/Videos/12240002_data.hdf5",
            load=True)
    input_video = "/home/ppxjd3/Videos/12240002_crop.mp4"
    output_video = "/home/ppxjd3/Videos/12240002_crop_tri.mp4"
    VA = VideoAnnotator(
            dataframe,
            input_video,
            output_video,
            shrink_factor=1,
            multiprocess=True)
    # VA.add_coloured_circles('order')
    VA.add_delaunay_network()
