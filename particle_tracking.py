import cv2
import Generic.video as video
import ParticleTracking.preprocessing as preprocessing
import ParticleTracking.dataframes as dataframes
import numpy as np
import ParticleTracking.configuration as config

class ParticleTracker:
    """Class to track the locations of the particles in a video."""

    def __init__(self, vid, dataframe, options, process_config, write_video_filename):
        self.video = vid
        self.options = options
        self.IP = preprocessing.ImagePreprocessor(self.video, process_config,
                                                  self.options)
        self.new_vid_filename = write_video_filename
        self.TD = dataframe

    def track(self):
        """Call this to start the tracking"""
        for f in range(self.video.num_frames):
            print(f+1, " of ", self.video.num_frames)
            frame = self.video.read_next_frame()
            new_frame = self.IP.process_image(frame)
            circles = self._find_circles(new_frame)
            # self._annotate_video_with_circles(new_frame, circles)
            self.TD.add_tracking_data(f, circles)
        self.TD.filter_trajectories()
        self.TD.save_dataframe()

    def _annotate_video_with_circles(self, frame, circles):

        if len(np.shape(frame)) == 2:
            frame = np.stack((frame, frame, frame), axis=2)
        for i in circles[0, :]:
            cv2.circle(frame, (int(i[0]), int(i[1])),
                       int(i[2]), (0, 255, 255), 2)

        if self.video.frame_num == 1:
            self.new_video = video.WriteVideo(
                    self.new_vid_filename,
                    frame_size=np.shape(frame))

        self.new_video.add_frame(frame)

        if self.video.frame_num == self.video.num_frames:
            self.new_video.close()

    def _find_circles(self, frame):
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
    dataframe_name = "/home/ppxjd3/Code/ParticleTracking/test_data/test_video.hdf5"
    dataframe = dataframes.TrackingDataframe(dataframe_name)
    PT = ParticleTracker(in_vid, dataframe, options_dict,
                         process_config, out_vid)
    PT.track()
