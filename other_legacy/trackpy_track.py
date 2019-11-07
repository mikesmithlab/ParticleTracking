from Generic import images, video
from ParticleTracking.tracking import ParticleTracker
from ParticleTracking import preprocessing
from ParticleTracking.other_legacy import configurations
import trackpy as tp


class Trackpy_Track(ParticleTracker):
    """
    Example class for inheriting ParticleTracker
    """

    def __init__(self, filename, tracking=False, multiprocess=False):
        """
        Parameters
        ----------
        filename: str
            filepath for video.ReadVideo class

        tracking: bool
            If true, do steps specific to tracking.

        multiprocess: bool
            If true performs tracking on multiple cores
        """
        self.tracking = tracking
        self.parameters = configurations.TRACKPY_PARAMETERS
        self.ip = preprocessing.Preprocessor(self.parameters)
        self.input_filename = filename
        if self.tracking:
            ParticleTracker.__init__(self, multiprocess=multiprocess)
        else:
            self.cap = video.ReadVideo(self.input_filename)
            self.frame = self.cap.read_next_frame()

    def analyse_frame(self):
        """
        Change steps in this method.

        This is done every frame

        Returns
        -------
        info:
            (N, X) array containing X values for N tracked objects
        boundary:
            from preprocessor
        info_headings:
            List of X strings describing the values in circles

        """
        if self.tracking:
            frame = self.cap.read_next_frame()
        else:
            frame = self.cap.find_frame(self.parameters['frame'][0])
        new_frame, boundary, cropped_frame = self.ip.process(frame)

        ### ONLY EDIT BETWEEN THESE COMMENTS
        info =tp.locate(
            new_frame,
            self.parameters['min_dist'][0],
            self.parameters['p_1'][0],
            self.parameters['p_2'][0],
            self.parameters['min_rad'][0],
            self.parameters['max_rad'][0])

        info_headings = ['x', 'y', 'r']
        ### ONLY EDIT BETWEEN THESE COMMENTS
        if self.tracking:
            return info, boundary, info_headings
        else:
            # THIS NEXT LINE CAN BE EDITED TOO
            annotated_frame = images.draw_circles(cropped_frame, info)
            return new_frame, annotated_frame

    def extra_steps(self):
        """
        Add extra steps here which can be performed after tracking.

        Accepts no arguments and cannot return.

        This is done once.
        """
        pass



if __name__ == "__main__":
    from Generic import filedialogs
    file = filedialogs.load_filename('Load a video')
    tracker = Trackpy_Track(file, tracking=True, multiprocess=False)
    tracker.track()
