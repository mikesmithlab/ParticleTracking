import cv2
import numpy as np
import ParticleTracking.preprocessing as preprocessing
import ParticleTracking.dataframes as dataframes
import ParticleTracking.particle_tracking as particle_tracking
import Generic.video as video


class VideoAnnotator:
    """Class to annotate videos with information"""

    def __init__(self,
                 dataframe_inst,
                 input_video_filename,
                 output_video_filename):
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
        print(self.td.dataframe.head())
        self.input_video = video.ReadVideo(input_video_filename)
        self.output_video = video.WriteVideo(
                output_video_filename,
                frame_size=(int(self.input_video.height),
                            int(self.input_video.width),
                            int(3)))

    def add_tracking_circles(self):
        """Annotates a video with the tracked circles to check tracking"""
        for f in range(self.input_video.num_frames):
            frame = self.input_video.read_next_frame()
            circles = self.td.return_circles_for_frame(f)
            for x, y, size in circles:
                cv2.circle(frame, (int(x), int(y)),
                           int(size), (0, 255, 255), 2)
            self.output_video.add_frame(frame)
        self.output_video.close()


if __name__=="__main__":

    dataframe = dataframes.TrackingDataframe(
            "/home/ppxjd3/Code/ParticleTracking/test_data/test_video.hdf5",
            load=True)
    VA = VideoAnnotator(dataframe,
            "/home/ppxjd3/Code/ParticleTracking/test_data/test_video_crop.avi",
            "/home/ppxjd3/Code/ParticleTracking/test_data/test_video_crop_annotated.avi")
    VA.add_tracking_circles()