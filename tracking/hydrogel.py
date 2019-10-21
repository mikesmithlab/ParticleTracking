from Generic import images, video
from ParticleTracking.tracking import ParticleTracker
from ParticleTracking import configurations, preprocessing
import trackpy as tp


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
        self.parameters = configurations.EXAMPLE_CHILD_PARAMETERS
        #If you want to use the variance method to subtract a bkg img use the following line
        #The bkg image should be stored with the movie with same name + suffix = _bkgimg.png
        #self.parameters['bkg_img'] = cv2.imread(filename[:-5] + '_bkgimg.png')]
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


        info_headings = ['x', 'y', 'r']
        ### ONLY EDIT BETWEEN THESE COMMENTS
        if self.tracking:
            #return info, boundary, info_headings
            pass
        else:
            # THIS NEXT LINE CAN BE EDITED TOO
            #annotated_frame = self._draw_circles(new_frame, info )
            #return new_frame, annotated_frame
            pass

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
    from Generic import filedialogs
    from ParticleTracking.tracking.tracking_gui import TrackingGui
    file = filedialogs.load_filename('Load a video')
    tracker = Hydrogel(file, tracking=False, multiprocess=False)
    
    TrackingGui(tracker)
    
    
    #tracker.track()
