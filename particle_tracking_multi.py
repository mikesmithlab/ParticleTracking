import cv2
import numpy as np
import subprocess as sp
import multiprocessing as mp
import os
import ParticleTracking.preprocessing as prepro
import ParticleTracking.dataframes as dataframes
import ParticleTracking.configuration as config
from operator import methodcaller

class ParticleTrackerMulti:

    def __init__(self,
                 input_video_filename,
                 options,
                 method_order):
        self.video_filename = input_video_filename
        self.options = options
        self.method_order = method_order
        self.ip = prepro.ImagePreprocessor(self.method_order, self.options)

    def track_process(self, group_number):
        cap = cv2.VideoCapture(self.video_filename)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_jump_unit * group_number)

        out = cv2.VideoWriter("{}.{}".format(group_number, self.extension),
                              cv2.VideoWriter_fourcc(*self.fourcc),
                              self.fps,
                              (self.width, self.height))

        proc_frames = 0
        while proc_frames < self.frame_jump_unit:
            ret, frame = cap.read()
            if ret == False:
                break
            new_frame, cropped_frame, _ = self.ip.process_image(frame)
            circles = self.find_circles(new_frame)
            anno_frame = self.annotate_frame_with_circles(cropped_frame, circles)
            out.write(anno_frame)
            proc_frames += 1
        cap.release()
        out.release()

    @staticmethod
    def annotate_frame_with_circles(frame, circles):
        if len(circles) > 0:
            for i in range(np.shape(circles)[1]):
                x = circles[:, i, 0]
                y = circles[:, i, 1]
                size = circles[:, i, 2]
                cv2.circle(frame, (int(x), int(y)),
                           int(size), (0, 255, 255), 2)
        return frame

    def find_circles(self, frame):
        """
        Uses cv2.HoughCircles to detect circles in a image

        Parameters
        ----------
        frame: numpy array

        Class Options
        -------------
        Uses the following keys from self.options:
            'min_dist'
            'p_1'
            'p_2'
            'min_rad'
            'max_rad'

        Returns
        -------
        circles: numpy array
            Contains the x, y, and radius of each detected circle
        """
        circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1,
                                   self.options['min_dist'],
                                   param1=self.options['p_1'],
                                   param2=self.options['p_2'],
                                   minRadius=self.options['min_rad'],
                                   maxRadius=self.options['max_rad'])
        return circles

    def start_multi_track(self):
        self.num_processes = mp.cpu_count()
        self.extension = "mp4"
        cap = cv2.VideoCapture(self.video_filename)
        self.frame_jump_unit = cap.get(cv2.CAP_PROP_FRAME_COUNT) // self.num_processes
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.fourcc = "mp4v"
        _, frame = cap.read()
        _, cropped_frame, _ = self.ip.process_image(frame)
        self.width = int(np.shape(cropped_frame)[1])
        self.height = int(np.shape(cropped_frame)[0])
        p = mp.Pool(self.num_processes)
        p.map(self.track_process, range(self.num_processes))
        self.cleanup_intermediate_files()

    def cleanup_intermediate_files(self):
        intermediate_files = ["{}.{}".format(i, self.extension) for i in range(self.num_processes)]
        with open("intermediate_files.txt", "w") as f:
            for t in intermediate_files:
                f.write("file {} \n".format(t))

        ffmepg_command =  "ffmpeg -y -loglevel error -f concat -safe 0 -i intermediate_files.txt "
        ffmepg_command += " -vcodec copy"
        ffmepg_command += "/home/ppxjd3/Videos/12240002_out.{}".format(self.extension)
        sp.Popen(ffmepg_command, shell=True).wait()

        for f in intermediate_files:
            os.remove(f)
        os.remove("intermediate_files.txt")


if __name__=="__main__":
    vid_name = "/home/ppxjd3/Videos/12240002.MP4"
    process_config = config.RUBBER_BEAD_PROCESS_LIST
    config_df = config.ConfigDataframe()
    options = config_df.get_options('Rubber_Bead')
    PT = ParticleTrackerMulti(vid_name, options, process_config)
    import time
    start = time.time()
    PT.start_multi_track()
    print(time.time()-start)