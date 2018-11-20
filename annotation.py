import cv2
import ParticleTracking.dataframes as df
import Generic.video as vid
import Generic.images as im
from matplotlib import cm
import numpy as np
import multiprocessing as mp
import subprocess as sub
import os

class VideoAnnotator:

    def __init__(
            self,
            dataframe_inst,
            input_video_filename,
            shrink_factor=1,
            multiprocess=True
            ):
        self.td = dataframe_inst
        self.input_video_filename = input_video_filename
        self.core_filename, self.extension = (
                os.path.splitext(self.input_video_filename))
        self.shrink_factor = shrink_factor
        self.multiprocess = multiprocess

    def add_annotations(self, voronoi=False, delaunay=False):
        cap = vid.ReadVideo(self.input_video_filename)
        output_video_filename = (
                self.core_filename +
                '_network' +
                self.extension
                )
        out = vid.WriteVideo(output_video_filename,
                             (cap.height, cap.width, 3))
        for f in range(cap.num_frames):
            print(f)
            points = self.td.extract_points_for_frame(f)
            frame = cap.read_next_frame()
            if delaunay:
                frame = im.draw_delaunay_tess(frame, points)
            if voronoi:
                frame = im.draw_voronoi_cells(frame, points)
            out.add_frame(frame)
        cap.close()
        out.close()

    def add_coloured_circles(self, parameter=None):
        self.parameter = parameter
        if parameter is not None:
            self.circle_type = -1
            self.output_video_filename = self.core_filename + '_' + \
                                         self.parameter + self.extension
        else:
            self.parameter = 'particle'
            self.circle_type = 2
            self.output_video_filename = self.core_filename + '_' + \
                                         '_circles' + self.extension

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
        cap = vid.ReadVideo(self.input_video_filename)
        self.fourcc = "mp4v"
        self._find_video_info()
        if self.multiprocess:
            frame_no_start = self.frame_jump_unit * group_number
            cap.set_frame(frame_no_start)
            write_name = "{}.{}".format(group_number, self.extension)
            if group_number + 1 == self.num_processes:
                self.frame_jump_unit += self.remainder
        else:
            frame_no_start = 0
            write_name = self.output_video_filename

        out = vid.WriteVideo(write_name,
                             frame_size=(self.height, self.width, 3),
                             codec=self.fourcc,
                             fps=self.fps)
        col = (0, 255, 255)
        proc_frames = 0
        for f in range(int(self.frame_jump_unit)):
            f += frame_no_start
            frame = cap.read_next_frame()
            try:
                info = self.td.return_property_and_circles_for_frame(
                    f, self.parameter)
            except AssertionError as error:
                print(error)
                self.output_video_filename = ''
                break
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
        sub.Popen(ffmepg_command, shell=True).wait()

        for f in intermediate_files:
            os.remove(f)
        os.remove("intermediate_files.txt")

    def _find_video_info(self):
        cap = vid.ReadVideo(self.input_video_filename)
        if self.multiprocess:
            self.frame_jump_unit = cap.num_frames // self.num_processes
            self.remainder = cap.num_frames % self.num_processes
        else:
            self.frame_jump_unit = cap.num_frames
        self.fps = cap.fps
        self.width = int(cap.width/self.shrink_factor)
        self.height = int(cap.height/self.shrink_factor)


if __name__ == "__main__":

    dataframe = df.TrackingDataframe(
            "/home/ppxjd3/Videos/test_data.hdf5",
            load=True)
    input_video = "/home/ppxjd3/Videos/test_crop.mp4"
    VA = VideoAnnotator(
            dataframe,
            input_video,
            shrink_factor=1,
            multiprocess=True)
    VA.add_coloured_circles('order')
    # VA.add_annotations(voronoi=True, delaunay=True)
