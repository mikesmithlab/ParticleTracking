import cv2
import numpy as np
import subprocess as sp
import multiprocessing as mp
import os
import ParticleTracking.preprocessing as prepro
import ParticleTracking.dataframes as dataframes
import ParticleTracking.configuration as config
import ParticleTracking.annotation as anno
from operator import methodcaller
import Generic.video as vid
import trackpy as tp

class ParticleTrackerMulti:

    def __init__(self,
                 input_video_filename,
                 options,
                 method_order):
        self.video_filename = input_video_filename
        self.video_corename = os.path.splitext(input_video_filename)[0]
        self.data_store_filename = self.video_corename + '_data.hdf5'
        self.options = options
        self.method_order = method_order
        self.ip = prepro.ImagePreprocessor(self.method_order, self.options)

    def track(self):
        self.num_processes = mp.cpu_count()
        self.extension = "mp4"
        self.fourcc = "mp4v"
        self.find_video_info()

        p = mp.Pool(self.num_processes)
        p.map(self.track_process, range(self.num_processes))

        self.cleanup_intermediate_files()
        self.cleanup_intermediate_dataframes()
        self.link_trajectories()
        self._check_video_tracking()

    def _check_video_tracking(self):
        va = anno.VideoAnnotator(
                self.data_store,
                self.video_corename + "_crop.{}".format(self.extension),
                self.video_corename + "_annotated.{}".format(self.extension))
        va.add_tracking_circles()


    def link_trajectories(self):
        self.data_store = dataframes.TrackingDataframe(self.data_store_filename,
                                                       load=True)
        self.data_store.dataframe = tp.link_df(
                self.data_store.dataframe,
                self.options['max frame displacement'])
        self.data_store.dataframe = tp.filter_stubs(
                self.data_store.dataframe,
                self.options['min frame life'])
        self.data_store.save_dataframe()

    def cleanup_intermediate_dataframes(self):
        dataframe_list = ["{}.hdf5".format(i) for i in
                          range(self.num_processes)]
        dataframes.concatenate_dataframe(dataframe_list,
                                         self.data_store_filename)
        for file in dataframe_list:
            os.remove(file)

    def find_video_info(self):
        cap = vid.ReadVideo(self.video_filename)
        self.frame_jump_unit = cap.num_frames // self.num_processes
        self.fps = cap.fps
        frame = cap.read_next_frame()
        _, cropped_frame, _ = self.ip.process_image(frame)
        self.width = int(np.shape(cropped_frame)[1])
        self.height = int(np.shape(cropped_frame)[0])

    def track_process(self, group_number):
        data = dataframes.TrackingDataframe(str(group_number)+'.hdf5')
        cap = vid.ReadVideo(self.video_filename)
        frame_no_start = self.frame_jump_unit * group_number
        cap.set_frame(frame_no_start)
        out_crop = vid.WriteVideo("{}.{}".format(group_number, self.extension),
                                  frame_size=(self.height, self.width, 3),
                                  codec=self.fourcc,
                                  fps=self.fps)

        proc_frames = 0
        while proc_frames < self.frame_jump_unit:
            frame = cap.read_next_frame()
            new_frame, cropped_frame, boundary = self.ip.process_image(frame)
            circles = self.find_circles(new_frame)
            data.add_tracking_data(frame_no_start+proc_frames, circles, boundary)
            anno_frame = self.annotate_frame_with_circles(cropped_frame.copy(),
                                                          circles)
            out_crop.add_frame(cropped_frame)
            proc_frames += 1
        data.save_dataframe()
        cap.close()
        out_crop.close()
        data

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

    def cleanup_intermediate_files(self):
        intermediate_files = ["{}.{}".format(i, self.extension) for i in range(self.num_processes)]
        with open("intermediate_files.txt", "w") as f:
            for t in intermediate_files:
                f.write("file {} \n".format(t))

        ffmepg_command =  "ffmpeg -y -loglevel error -f concat -safe 0 -i intermediate_files.txt "
        ffmepg_command += " -vcodec copy "
        ffmepg_command += self.video_corename+"_crop.{}".format(self.extension)
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
    PT.track()
    print(time.time()-start)