from Generic import images, video
from ParticleTracking.tracking import ParticleTracker
from ParticleTracking import configurations, preprocessing
import numpy as np


class Bacteria(ParticleTracker):
    """
    Example class for inheriting ParticleTracker

    analyse_frame works by applying an adaptive threshold to the image.
    It then finds the contours of the bacteria. We then fit the minimum
    bounding rectangle to the bacterium before using the area of this
    to define whether it is a single bacterium or multiple. This classification
    is stored in 'classifier' field.
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
        This is done every frame

        Returns
        -------
        info:
            (N, X) array containing X values for N tracked objects
        info_headings:
            List of X strings describing the values for each object

        """
        if self.tracking:
            frame = self.cap.read_next_frame()
        else:
            frame = self.cap.find_frame(self.parameters['frame'][0])
        new_frame, boundary, cropped_frame = self.ip.process(frame)

        contours = images.find_contours(new_frame)


        for index,contour in enumerate(contours):
            '''
            We fit the bounded rectangle to the bacterium contour
            we then overwrite the x and y coords with contour centre of mass.
            '''
            info_bacterium = images.rotated_bounding_rectangle(contour)
            area = info_bacterium[3]*info_bacterium[4]

            if area <= 0.6*self.parameters['area bacterium'][0]:
                #classify small things as noise.
                classifier = int(0) # ie mistake it is not a bacterium
            else:
                #overwrite x and y
                info_bacterium[0], info_bacterium[1] = images.find_contour_centre(contour)
            '''
            Here we attempt to classify whether it is a bit of noise
            single bacterium, dividing or a clump of them
            '''
            if (area > 0.6*self.parameters['area bacterium'][0]) & (area <= 1.8*self.parameters['area bacterium'][0]):
                classifier = int(1)  # single bacterium - Blue
            elif (area > 1.8*self.parameters['area bacterium'][0]) & (area <= 2.9*self.parameters['area bacterium'][0]):
                classifier = int(2) # probably a dividing bacterium - Red
            elif (area > (2.8* self.parameters['area bacterium'][0])):
                classifier = int(3) # probably an aggregate - Green



            info_bacterium.append(classifier)
            if index == 0:
                info = [info_bacterium]
            else:
                info.append(info_bacterium)

        if self.tracking:
            info = list(zip(*info))
            info_headings = ['x', 'y', 'theta', 'width', 'length', 'box',
                             'classifier']
            return info, boundary, info_headings
        else:
            
            for bacterium in info:

                if bacterium[6] == 1:
                    annotated_frame = images.draw_contours(cropped_frame, [np.array(bacterium[5])], col=(0, 0 ,255))
                elif bacterium[6] == 2:
                    annotated_frame = images.draw_contours(cropped_frame, [np.array(bacterium[5])], col=(255, 0, 0))
                elif bacterium[6] == 3:
                    annotated_frame = images.draw_contours(cropped_frame, [np.array(bacterium[5])], col=(0, 255, 0))
            return new_frame, annotated_frame

    def extra_steps(self):
        """
        x and y are moved to x_raw, y_raw and a smoothed version of x and y
        are added to each frame.
        """
        print(self.data_store.head())



if __name__ == "__main__":

    from ParticleTracking.tracking.tracking_gui import TrackingGui

    file = '/media/ppzmis/data/ActiveMatter/bacteria_plastic/bacteria.mp4'
    tracker = Bacteria(file, tracking=True)
    tracker.track()


    #gui = TrackingGui(tracker)



