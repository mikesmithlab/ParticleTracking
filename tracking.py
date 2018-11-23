import cv2
import os
import time
import numpy as np
import trackpy as tp
import multiprocessing as mp
import subprocess as sub
import Generic.video as vid
import Generic.images as im
import ParticleTracking.preprocessing as pp
import ParticleTracking.dataframes as df
import ParticleTracking.configuration as config
import ParticleTracking.annotation as an


class ParticleTracker:
    """Class to track the locations of the particles in a video."""

    def __init__(self,
                 input_video_filename,
                 options,
                 method_order,
                 multiprocess=False,
                 save_crop_video=True,
                 save_check_video=False):

        self.video_filename = input_video_filename
        self.video_corename = os.path.splitext(input_video_filename)[0]
        self.data_store_filename = self.video_corename + '_data.hdf5'
        self.options = options
        self.method_order = method_order
        self.multiprocess = multiprocess
        self.save_crop_video = save_crop_video
        self.save_check_video = save_check_video
        self.ip = pp.ImagePreprocessor(self.method_order, self.options)
        self.check_options()

    def check_options(self):
        assert 'min_dist' in self.options, 'min_dist not in dictionary'
        assert 'p_1' in self.options, 'p_1 not in dictionary'
        assert 'p_2' in self.options, 'p_2 not in dictionary'
        assert 'min_rad' in self.options, 'min_rad not in dictionary'
        assert 'max_rad' in self.options, 'max rad not in dictionary'


    def track(self):
        if self.multiprocess==True:
            self.track_multiprocess()
        else:
            self.track_singleprocess()

    def track_multiprocess(self):
        """Call this to start tracking"""
        self.num_processes = mp.cpu_count()
        self.extension = "mp4"
        self.fourcc = "mp4v"
        self._find_video_info()

        p = mp.Pool(self.num_processes)
        p.map(self._track_process, range(self.num_processes))

        self._cleanup_intermediate_files()
        self._cleanup_intermediate_dataframes()
        self._link_trajectories()
        if self.save_check_video:
            self._check_video_tracking()

    def track_singleprocess(self):
        """Call this to start the tracking"""
        self.video = vid.ReadVideo(self.video_filename)
        if os.path.exists(self.data_store_filename):
            os.remove(self.data_store_filename)
        data = df.TrackingDataframe(self.data_store_filename)
        for f in range(self.video.num_frames):
            print(f+1, " of ", self.video.num_frames)
            frame = self.video.read_next_frame()
            new_frame, cropped_frame, boundary = self.ip.process_image(frame)
            circles = self.find_circles(new_frame)
            if self.save_crop_video:
                self._save_cropped_video(cropped_frame)
            data.add_tracking_data(f, circles, boundary)
        data.save_dataframe()
        self._link_trajectories()
        if self.save_check_video:
            self._check_video_tracking()

    def _save_cropped_video(self, frame):
        if self.video.frame_num == 1:
            self.crop_video = vid.WriteVideo(
                    self.video_corename + "_crop.mp4",
                    frame_size=np.shape(frame),
                    codec='mp4v')
        self.crop_video.add_frame(frame)

        if self.video.frame_num == self.video.num_frames:
            self.crop_video.close()

    def _find_video_info(self):
        """From the video reads properties for other methods"""
        cap = vid.ReadVideo(self.video_filename)
        self.frame_jump_unit = cap.num_frames // self.num_processes
        self.fps = cap.fps
        frame = cap.read_next_frame()
        _, cropped_frame, _ = self.ip.process_image(frame)
        self.width, self.height = im.get_width_and_height(cropped_frame)

    def _track_process(self, group_number):
        """
        The method which is mapped to the Pool implementing the tracking.

        Finds the circles in a percentage of the video and saves the cropped
        video and dataframe for this part to the current working directory.

        Parameters
        ----------
        group_number: int
            Describes which fraction of the video the method should act on
        """
        data = df.TrackingDataframe(str(group_number) + '.hdf5')
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
            data.add_tracking_data(frame_no_start+proc_frames,
                                   circles,
                                   boundary)
            out_crop.add_frame(cropped_frame)
            proc_frames += 1
        data.save_dataframe()
        cap.close()
        out_crop.close()

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
        circles = np.squeeze(circles)
        return circles

    def _cleanup_intermediate_files(self):
        """
        Concatenates the intermediate videos using ffmpeg.

        Concatenates the videos whose filepath are in intermediate_files.txt
        then removes the intermediate videos and the text file.
        """
        intermediate_files = ["{}.{}".format(i, self.extension)
                              for i in range(self.num_processes)]
        with open("intermediate_files.txt", "w") as f:
            for t in intermediate_files:
                f.write("file {} \n".format(t))

        ffmepg_command = "ffmpeg -y -loglevel error -f concat " \
                         "-safe 0 -i intermediate_files.txt "
        ffmepg_command += " -vcodec copy "
        ffmepg_command += self.video_corename+"_crop.{}".format(self.extension)
        sub.Popen(ffmepg_command, shell=True).wait()

        for f in intermediate_files:
            os.remove(f)
        os.remove("intermediate_files.txt")

    def _cleanup_intermediate_dataframes(self):
        """Concatanates and removes intermediate dataframes"""
        dataframe_list = ["{}.hdf5".format(i) for i in
                          range(self.num_processes)]
        df.concatenate_dataframe(dataframe_list,
                                 self.data_store_filename)
        for file in dataframe_list:
            os.remove(file)

    def _link_trajectories(self):
        """Implements the trackpy functions link_df and filter_stubs"""
        data_store = df.TrackingDataframe(self.data_store_filename,
                                          load=True)
        try:
            a = self.options['max frame displacement']
        except KeyError as error:
            print(error)
            print('max frame displacement set to 10')
            self.options['max frame displacement'] = 10
        try:
            a = self.options['min frame life']
        except KeyError as error:
            print(error)
            print('min frame life set to 10')
            self.options['min frame life'] = 10
        data_store.dataframe = tp.link_df(
                data_store.dataframe,
                self.options['max frame displacement'])
        data_store.dataframe = tp.filter_stubs(
                data_store.dataframe,
                self.options['min frame life'])
        data_store.save_dataframe()

    def _check_video_tracking(self):
        """Uses the VideoAnnotator class to draw circles on the video"""
        data_store = df.TrackingDataframe(self.data_store_filename,
                                          load=True)
        va = an.VideoAnnotator(
                data_store,
                self.video_corename + "_crop.mp4",
                multiprocess=True)
        va.add_coloured_circles()


if __name__ == "__main__":
    choice = 'glass'
    config_df = config.ConfigDataframe()
    if choice == 'rubber':
        vid_name = "/home/ppxjd3/Videos/12240002.MP4"
        ml = config.MethodsList('Rubber_Bead', load=True)
        process_config = ml.extract_methods()
        options_in = config_df.get_options('Rubber_Bead')
    else:
        vid_name = "/home/ppxjd3/Videos/test.mp4"
        ml = config.MethodsList('Glass_Bead', load=True)
        process_config = ml.extract_methods()
        options_in = config_df.get_options('Glass_Bead')

    PT = ParticleTracker(vid_name,
                         options_in,
                         process_config,
                         multiprocess=False,
                         save_crop_video=True,
                         save_check_video=True)
    start = time.time()
    PT.track()
    print(time.time()-start)
