from Generic import images, video
from ParticleTracking.tracking import ParticleTracker
from ParticleTracking import configurations, preprocessing


class ExampleChild(ParticleTracker):
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
        self.parameters = configurations.NITRILE_BEADS_PARAMETERS
        self.ip = preprocessing.Preprocessor(self.parameters)
        self.input_filename = filename
        if self.tracking:
            ParticleTracker.__init__(self, multiprocess=multiprocess)
        else:
            self.cap = video.ReadVideo(self.input_filename)

    def _analyse_frame(self):
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
        frame = self.cap.read_next_frame()
        new_frame, boundary = self.ip.process(frame)
        info = images.find_circles(
            new_frame,
            self.parameters['min_dist'][0],
            self.parameters['p_1'][0],
            self.parameters['p_2'][0],
            self.parameters['min_rad'][0],
            self.parameters['max_rad'][0])
        info_headings = ['x', 'y', 'r']
        if self.tracking:
            return info, boundary, info_headings
        else:
            return frame, new_frame

    def extra_steps(self):
        """
        Add extra steps here which can be performed after tracking.

        Accepts no arguments and cannot return.

        This is done once.
        """
        pass

    def update_parameters(self, parameters):
        self.parameters = parameters
        self.ip.update_parameters(self.parameters)


if __name__ == "__main__":
    from Generic import filedialogs
    file = filedialogs.load_filename('Load a video')
    tracker = ExampleChild(file, tracking=True, multiprocess=True)
    tracker.track()
