import cv2
import Generic.video as video
import ParticleTracking.preprocessing as preprocessing
import ParticleTracking.dataframes as dataframes
import numpy as np


class ParticleTracker:
    """Class to track the locations of the particles in a video."""

    def __init__(self, vid, options, write_video_filename, dataframe_filename):
        self.video = vid
        self.options = options
        self.IP = preprocessing.ImagePreprocessor(self.video, 1)
        self.new_vid_filename = write_video_filename
        self.df = dataframes.TrackingDataframe(dataframe_filename)

    def track(self):
        for f in range(self.video.num_frames):
            print(f+1, " of ", self.video.num_frames)
            frame = vid.read_next_frame()
            new_frame = self.IP.process_image(frame)
            circles = self._find_circles(new_frame)
            # self._annotate_video_with_circles(new_frame, circles)
            self.df.add_tracking_data(f, circles)
        self.df.filter_trajectories()
        self.df.save_dataframe()

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
    vid = video.ReadVideo(
        "/home/ppxjd3/Code/ParticleTracking/test_data/test_video_EDIT.avi")
    options_dict = {'min_dist': 20, 'p_1': 200, 'p_2': 10, 'min_rad': 18,
                    'max_rad': 20}
    out_vid = "/home/ppxjd3/Code/ParticleTracking/test_data/test_video_annotated.avi"
    dataframe_name = "/home/ppxjd3/Code/ParticleTracking/test_data/test_video.hdf5"
    PT = ParticleTracker(vid, options_dict, out_vid, dataframe_name)
    PT.track()
