import cv2
import Generic.video as video
import ParticleTracking.preprocessing as preprocessing
import ParticleTracking.dataframes as dataframes
import numpy as np
import ParticleTracking.configuration as config
import trackpy as tp
import annotation as anno


class ParticleTracker:
    """Class to track the locations of the particles in a video."""

    def __init__(self,
                 input_video,
                 dataframe_inst,
                 preprocessor_inst,
                 options,
                 method_order,
                 crop_vid_filename=None,
                 test_vid_filename=None):
        """
        Init

        Parameters
        ----------
        vid: class instance
            Instance of class Generic.video.ReadVideo containing the input video

        dataframe: class instance
            Instance of class ParticleTracking.dataframes.TrackingDataframe

        options: dictionary
            A dictionary containing all the parameters needed for functions

        method_order: list
            A list containing string associated with methods in the order they
            will be used

        crop_vid_filename: string
            A string containing the filepath to save the cropped video
                If None don't make video

        test_vid_filename: string
            A string containing the filepath to save the video with the tracked
            circles annotated on top.
                If None, don't make video
        """
        self.video = input_video
        self.options = options
        self.ip = preprocessor_inst
        self.ip = preprocessing.ImagePreprocessor(self.video,
                                                  method_order,
                                                  self.options)
        self.crop_vid_filename = crop_vid_filename
        self.test_vid_filename = test_vid_filename
        self.td = dataframe_inst

    def track(self):
        """Call this to start the tracking"""
        for f in range(self.video.num_frames):
            print(f+1, " of ", self.video.num_frames)
            frame = self.video.read_next_frame()
            new_frame, cropped_frame, boundary = self.ip.process_image(frame)
            circles = self.find_circles(new_frame)
            if self.crop_vid_filename:
                self._save_cropped_video(cropped_frame)
            self.td.add_tracking_data(f, circles, boundary)
        self._filter_trajectories()
        self.td.save_dataframe()
        if self.test_vid_filename:
            self._check_video_tracking()

    def _check_video_tracking(self):
        va = anno.VideoAnnotator(self.td,
                                 self.crop_vid_filename,
                                 self.test_vid_filename)
        va.add_tracking_circles()

    def annotate_frame_with_circles(self, frame, circles):
        if len(circles) > 0:
            for x, y, size in circles:
                cv2.circle(frame, (int(x), int(y)),
                           int(size), (0, 255, 255), 2)
        return frame

    def _save_cropped_video(self, frame):
        if self.video.frame_num == 1:
            self.crop_video = video.WriteVideo(
                    self.crop_vid_filename,
                    frame_size=np.shape(frame))
        self.crop_video.add_frame(frame)

        if self.video.frame_num == self.video.num_frames:
            self.crop_video.close()

    def _filter_trajectories(self):
        """
        Use trackpy to filter trajectories

        Class Options
        -------------
        Uses the following keys from self.options:
            'max frame displacement'
            'min frame life'
        """
        self.td.dataframe = tp.link_df(self.td.dataframe,
                                       self.options['max frame displacement'])
        self.td.dataframe = tp.filter_stubs(self.td.dataframe,
                                            self.options['min frame life'])

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


if __name__ == "__main__":
    in_vid = video.ReadVideo(
        "/home/ppxjd3/Code/ParticleTracking/test_data/test_video_EDIT.avi")
    options_dict = config.GLASS_BEAD_OPTIONS_DICT
    process_config = config.GLASS_BEAD_PROCESS_LIST
    out_vid = "/home/ppxjd3/Code/ParticleTracking/test_data/test_video_annotated.avi"
    crop_vid_name = "/home/ppxjd3/Code/ParticleTracking/test_data/test_video_crop.avi"
    dataframe_name = "/home/ppxjd3/Code/ParticleTracking/test_data/test_video.hdf5"
    dataframe = dataframes.TrackingDataframe(dataframe_name)
    preprocess = preprocessing.ImagePreprocessor(in_vid,
                                                process_config,
                                                options_dict)
    PT = ParticleTracker(in_vid, dataframe, preprocess, options_dict,
                         process_config, crop_vid_name,  out_vid)
    PT.track()
