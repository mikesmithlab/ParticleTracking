from Generic import images, video
from ParticleTracking.tracking import ParticleTracker
from ParticleTracking import configurations, preprocessing
import numpy as np


class Bacteria(ParticleTracker):
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
        self.parameters = configurations.BACTERIA_PARAMETERS
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
        '''
        thresh = images.adaptive_threshold(self.blurred_img,
                                    self.parameters['adaptive threshold block size'][0],
                                    self.parameters['adaptive threshold C'][0],
                                    self.parameters['adaptive threshold mode'][0])
        '''

        contours = images.find_contours(new_frame)
        #images.display(images.draw_contours(new_frame, contours))



        for index,contour in enumerate(contours):
            info_bacterium = images.rotated_bounding_rectangle(contour)
            area = info_bacterium[3]*info_bacterium[4]

            '''
            Here we attempt to classify whether it is a bit of noise
            single bacterium, dividing or a clump of them
            '''


            if area < self.parameters['min area bacterium'][0]:
                print(0)
                classifier = int(0) # ie mistake it is not a bacterium
            elif (area > self.parameters['max area bacterium'][0]) & (area <= 2.5* self.parameters['max area bacterium'][0]):
                print(1)
                classifier = int(2) # probably a dividing bacterium
            elif (area > 2.5* self.parameters['max area bacterium'][0]):
                classifier = int(3) # probably an aggregate
            else:
                classifier = int(1) # single bacterium

            info_bacterium.append(classifier)
            if index == 0:
                info = [info_bacterium]
            else:
                info.append(info_bacterium)


        ### ONLY EDIT BETWEEN THESE COMMENTS
        if self.tracking:
            info = list(zip(*info))
            info_headings = ['x', 'y', 'theta', 'width', 'length', 'box',
                             'classifier']
            return info, boundary, info_headings
        else:
            
            for bacterium in info:
                print(bacterium)
                if bacterium[6] == 1:
                    annotated_frame = images.draw_contours(cropped_frame, [np.array(bacterium[5])], col=images.BLUE)
                elif bacterium[6] == 2:
                    annotated_frame = images.draw_contours(cropped_frame, [np.array(bacterium[5])], col=images.RED)
                elif bacterium[6] == 3:
                    annotated_frame = images.draw_contours(cropped_frame, [np.array(bacterium[5])], col=images.GREEN)
            return new_frame, annotated_frame

    def extra_steps(self):
        """
        Add extra steps here which can be performed after tracking.

        Accepts no arguments and cannot return.

        This is done once.
        """
        pass


if __name__ == "__main__":

    from ParticleTracking.tracking.tracking_gui import TrackingGui

    file = '/media/ppzmis/data/ActiveMatter/bacteria_plastic/bacteria.avi'
    tracker = Bacteria(file)
    gui = TrackingGui(tracker)



