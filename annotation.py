import cv2
import ParticleTracking.dataframes as dataframes
import Generic.video as video
from matplotlib import cm
import numpy as np
import multiprocessing as mp
import subprocess as sp
import os


class VideoAnnotator:
    """Class to annotate videos with information"""

    def __init__(self,
                 dataframe_inst,
                 input_video_filename,
                 output_video_filename,
                 shrink_factor=1,
                 multiprocess=True):
        """
        Initialise VideoAnnotator

        Parameters
        ----------
        dataframe_inst: Class instance
            Instance of the class dataframes.TrackingDataframe

        input_video_filename: string
            string containing the full filepath for a cropped video

        output_video_filename: string
            string containing the full filepath where the annotated video
            will be saved

        shrink_factor: int
            Size of video will be shrunk by this factor

        multiprocess: Bool
            If True will use multiprocessing for annotation
        """
        self.td = dataframe_inst
        self.input_video_filename = input_video_filename
        self.output_video_filename = output_video_filename
        self.shrink_factor = shrink_factor
        self.multiprocess = multiprocess

    def add_coloured_circles(self, parameter=None):
        self.parameter = parameter
        if not parameter:
            self.dataframe_columns = ['x', 'y', 'size', 'particle']
            self.circle_type = 2
        else:
            self.dataframe_columns = ['x', 'y', 'size', parameter]
            self.circle_type = -1

        if self.multiprocess:
            self.add_coloured_circles_multi(parameter)
        else:
            self.add_coloured_circles_single(parameter)

    def add_coloured_circles_single(self, parameter):
        self._find_video_info()
        input_video = video.ReadVideo(self.input_video_filename)
        out = video.WriteVideo(self.output_video_filename,
                               frame_size=(self.height, self.width, 3),
                               codec='mp4v')
        col = (0, 255, 255)
        for f in range(int(input_video.num_frames)):
            frame = input_video.read_next_frame()
            info = self.td.return_property_and_circles_for_frame(f, self.dataframe_columns)
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
                                   fx=1/self.shrink_factor,
                                   fy=1/self.shrink_factor,
                                   interpolation=cv2.INTER_CUBIC)
            out.add_frame(frame)
        out.close()
        input_video.close()

    def add_coloured_circles_multi(self, parameter='order'):
        self.num_processes = mp.cpu_count()
        self.extension = "mp4"
        self.fourcc = "mp4v"
        self.parameter = parameter
        self._find_video_info()

        p = mp.Pool(self.num_processes)
        p.map(self._add_coloured_circles_process, range(self.num_processes))

        self._cleanup_intermediate_videos()

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

    def _add_coloured_circles_process(self, group_number):
        cap = video.ReadVideo(self.input_video_filename)
        frame_no_start = self.frame_jump_unit * group_number
        cap.set_frame(frame_no_start)
        out = video.WriteVideo("{}.{}".format(group_number, self.extension),
                               frame_size=(self.height, self.width, 3),
                               codec=self.fourcc,
                               fps=self.fps)
        proc_frames = 0
        col = (0, 255, 255)
        while proc_frames < self.frame_jump_unit:
            frame_no = frame_no_start + proc_frames
            info = self.td.return_property_and_circles_for_frame(
                    frame_no, self.dataframe_columns)
            frame = cap.read_next_frame()

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
                                   fx=1/self.shrink_factor,
                                   fy=1/self.shrink_factor,
                                   interpolation=cv2.INTER_CUBIC)
            proc_frames += 1
            out.add_frame(frame)
        cap.close()
        out.close()

    def _find_video_info(self):
        input_video = video.ReadVideo(self.input_video_filename)
        if self.multiprocess:
            self.frame_jump_unit = input_video.num_frames // self.num_processes
        self.fps = input_video.fps
        self.width = int(input_video.width/self.shrink_factor)
        self.height = int(input_video.height/self.shrink_factor)


if __name__ == "__main__":

    dataframe = dataframes.TrackingDataframe(
            "/home/ppxjd3/Videos/12240002_data.hdf5",
            load=True)
    VA = VideoAnnotator(
            dataframe,
            "/home/ppxjd3/Videos/12240002_crop.mp4",
            "/home/ppxjd3/Videos/12240002_crop_annotate.mp4",
            shrink_factor=1,
            multiprocess=True)
    VA.add_coloured_circles()
