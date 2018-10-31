import cv2
import ParticleTracking.dataframes as dataframes
import Generic.video as video
from matplotlib import cm
import numpy as np

class VideoAnnotator:
    """Class to annotate videos with information"""

    def __init__(self,
                 dataframe_inst,
                 input_video_filename,
                 output_video_filename,
                 shrink_factor=1):
        """
        Initialise VideoAnnotator

        Parameters
        ----------
        td: Class instance
            Instance of the class dataframes.TrackingDataframe

        input_video_filename: string
            string containing the full filepath for a cropped video

        output_video_filename: string
            string containing the full filepath where the annotated video
            will be saved
        """
        self.td = dataframe_inst
        self.input_video = video.ReadVideo(input_video_filename)
        self.shrink_factor = shrink_factor
        self.output_video = video.WriteVideo(
                output_video_filename,
                frame_size=(int(self.input_video.height/self.shrink_factor),
                            int(self.input_video.width/self.shrink_factor),
                            int(3)),
                codec='mp4v')

    def add_tracking_circles(self):
        """Annotates a video with the tracked circles to check tracking"""
        for f in range(self.input_video.num_frames):
            frame = self.input_video.read_next_frame()
            circles = self.td.return_circles_for_frame(f)
            if self.shrink_factor is not 1:
                frame = cv2.resize(frame,
                                   None,
                                   fx=1/self.shrink_factor,
                                   fy=1/self.shrink_factor,
                                   interpolation=cv2.INTER_CUBIC)
                circles /= self.shrink_factor
            for x, y, size in circles:
                cv2.circle(frame, (int(x), int(y)),
                           int(size), (0, 255, 255), 1)
            self.output_video.add_frame(frame)
        self.output_video.close()

    def add_coloured_circles(self, parameter='order'):
        for f in range(int(self.input_video.num_frames/18)):
            print(f, ' of ', self.input_video.num_frames)
            frame = self.input_video.read_next_frame()
            out = self.td.return_property_and_circles_for_frame(f, parameter)
            for xi, yi, r, param in out:
                col = np.multiply(cm.jet(param)[0:3], 255)
                cv2.circle(frame,
                           (int(xi), int(yi)),
                           int(r),
                           (col[0], col[1], col[2]),
                           -1)
            if self.shrink_factor is not 1:
                frame = cv2.resize(frame,
                                   None,
                                   fx=1/self.shrink_factor,
                                   fy=1/self.shrink_factor,
                                   interpolation=cv2.INTER_CUBIC)
            self.output_video.add_frame(frame)
        self.output_video.close()


if __name__=="__main__":

    dataframe = dataframes.TrackingDataframe(
            "/home/ppxjd3/Videos/12240002_data.hdf5",
            load=True)
    VA = VideoAnnotator(dataframe,
            "/home/ppxjd3/Videos/12240002_crop.mp4",
            "/home/ppxjd3/Videos/12240002_crop_order.mp4",
            shrink_factor=2)
    VA.add_coloured_circles()