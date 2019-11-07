from Generic import images, video
from ParticleTracking.tracking import ParticleTracker
from ParticleTracking import preprocessing, postprocessing
from ParticleTracking.other_legacy import configurations


class Hydrogel(ParticleTracker):
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
        self.parameters = configurations.HYDROGEL_PARAMETERS

        self.ip = preprocessing.Preprocessor(self.parameters)

        self.input_filename = filename
        if self.tracking:
            ParticleTracker.__init__(self, multiprocess=multiprocess)
        else:
            self.cap = video.ReadVideo(self.input_filename)
            self.frame = self.cap.read_next_frame()
            self.pp = postprocessing.PostProcessor(self.parameters)

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



        ### ONLY EDIT BETWEEN THESE COMMENTS
        if self.tracking:
            #return info, boundary, info_headings
            pass
        else:
            # THIS NEXT LINE CAN BE EDITED TOO
            #annotated_frame = self.annotate.process(frame)#draw_circles(new_frame, info )
            return new_frame, new_frame


    def extra_steps(self):
        """
        Add extra steps here which can be performed after tracking.

        Accepts no arguments and cannot return.

        This is done once.
        """
        pass


    def _draw_circles(self, frame, info):
        # info = info[:, :3] if self.parameter == 'particle' else info
        annotated_frame = images.pygame_draw_circles(frame, info)
        return annotated_frame


if __name__ == "__main__":
    from ParticleTracking.tracking.tracking_gui import TrackingGui
    #file = filedialogs.load_filename('Load a video')
    file='/home/mike/Documents/HydrogelTest.m4v'
    tracker = Hydrogel(file, tracking=False, multiprocess=False)
    TrackingGui(tracker)
    
    
    #tracker.track()
