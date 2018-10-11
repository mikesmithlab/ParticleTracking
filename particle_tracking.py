import cv2
import Generic.video as video
import ParticleTracking.preprocessing as preprocessing
import numpy as np

class ParticleTracker:
    """Class to track the locations of the particles in a video for each frame."""

    def __init__(self, vid, options, write_video_filename):
        self.video = vid
        self.options = options
        self.IP = preprocessing.ImagePreprocessor(self.video, 1)
        self.new_vid_filename = write_video_filename

    def track(self):
        for f in range(self.video.num_frames):
            print(f, " of ", self.video.num_frames)
            frame = vid.read_next_frame()
            new_frame = self.IP.process_image(frame)
            circles = self._find_circles(new_frame)
            annotated_frame = new_frame.copy()
            if len(np.shape(annotated_frame)) == 2:
                annotated_frame = np.stack((annotated_frame,
                                            annotated_frame,
                                            annotated_frame),
                                           axis=2)
            for i in circles[0, :]:
                cv2.circle(annotated_frame, (int(i[0]), int(i[1])),
                           int(i[2]), (0, 255, 255), 2)

            if self.video.frame_num == 1:
                self.new_video = video.WriteVideo(
                        self.new_vid_filename,
                        frame_size=np.shape(annotated_frame))


            self.new_video.add_frame(annotated_frame)
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
    PT = ParticleTracker(vid, options_dict, out_vid)
    PT.track()