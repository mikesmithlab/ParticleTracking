from Generic import images, video
from ParticleTracking.tracking import ParticleTracker
from ParticleTracking import configurations, preprocessing, dataframes
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2


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
        self.parameters = configurations.BACTERIA2_PARAMETERS
        self.colors = self.parameters['colors']
        self.ip = preprocessing.Preprocessor(self.parameters)
        self.input_filename = filename
        self.framenum = 0

        if self.tracking:
            ParticleTracker.__init__(self, multiprocess=multiprocess)
            self.old_info = None
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
        info = None

        if self.tracking:
            frame = self.cap.read_next_frame()
        else:
            frame = self.cap.find_frame(self.parameters['frame'][0])
        new_frame, boundary, cropped_frame = self.ip.process(frame)

        contours = images.find_contours(new_frame)
        for index, contour in enumerate(contours):
            info_bacterium = self._classify(contour, cropped_frame, new_frame)
            if info_bacterium is None:
                pass
            else:
                if info is None:
                    sz = np.size(np.shape(info_bacterium))
                    info = [info_bacterium]
                else:
                    sz = np.size(np.shape(info_bacterium))
                    if self.tracking:
                        if info_bacterium[6] >= 2:
                            if sz == 1:
                                info.append(info_bacterium)
                            else:
                                for i in range(sz):
                                    info.append(info_bacterium[i])
                    else:
                        if sz == 1:
                            info.append(info_bacterium)
                        else:
                            for i in range(sz):
                                info.append(info_bacterium[i])

        if self.tracking:
            self.old_info = info.copy()
            #self._show_track(frame, info)

            info = list(zip(*info))
            #print(info[0])

            info_headings = ['x', 'y', 'theta', 'width', 'length', 'box',
                             'classifier']
            self.framenum = self.framenum + 1
            return info, boundary, info_headings
        else:
            for bacterium in info:
                annotated_frame = images.draw_contours(cropped_frame, [
                    np.array(bacterium[5])], col=self.colors[bacterium[6]], thickness=1)
            return new_frame, annotated_frame

    def _classify(self, contour, cropped_frame, new_frame):

        '''
        We fit the bounded rectangle to the bacterium contour
        we then overwrite the x and y coords with contour centre of mass.
        '''
        info_bacterium = images.rotated_bounding_rectangle(contour)
        area = info_bacterium[3]*info_bacterium[4]

        if area <= 0.01*self.parameters['noise cutoff'][0]*self.parameters['area bacterium'][0]:
            classifier = int(1)
            info_bacterium.append(classifier)
            #Too small to be significant
            return info_bacterium
        elif (area > 0.01*self.parameters['noise cutoff'][0]*self.parameters['area bacterium'][0]) & (area <= 0.01*self.parameters['single bacterium cutoff'][0]*self.parameters['area bacterium'][0]):
            '''
            Here we attempt to classify whether it is a 
            single bacterium, dividing or just stuck together.
            Method:
            Single Bacterium has a given area. 
            Dividing bacterium was only a single object in all previous frames
            Stuck bacterium consisted of 2 or more objects in previous frame.
            '''
            aspect = float(info_bacterium[4]) / float(info_bacterium[3])
            if aspect > 0.01*self.parameters['aspect bacterium'][0]:
                classifier = int(2)  # single bacterium - Blue
            else:
                classifier = int(1)
            info_bacterium.append(classifier)
            return info_bacterium
        else:
            #Too big to be single bacterium
            classifier = int(3)
            info_bacterium.append(classifier)
            return info_bacterium

    def _show_track(self,frame,info):
        if self.framenum > 66:
            annotated_frame = self._draw_boxes(frame, info)

            images.display(annotated_frame)

    def _draw_boxes(self, frame, info):
        for bacterium in info:
            annotated_frame = images.draw_contours(frame, [bacterium[5]], col=self.colors[bacterium[6]])

        return annotated_frame


    def extra_steps(self):
        """
        x and y are moved to x_raw, y_raw and a smoothed version of x and y
        are added to each frame.

        df_name = os.path.splitext(self.input_filename)[0] + '.hdf5'
        data_store = dataframes.DataStore(self.data_filename,
                                          load=True)
        df1 = data_store.df

        df1['x raw'] = df1['x']
        df1['y raw'] = df1['y']
1
        bacterium_ids = np.unique(df1['particle'])

        for id in bacterium_ids:
            df1.update(df1.loc[df1['particle'] == id,'x'].rolling(self.parameters['trajectory smoothing']).mean())
            df1.update(df1.loc[df1['particle'] == id,'y'].rolling(self.parameters['trajectory smoothing']).mean())

        print(df1[df1['particle'] == id]['x'])
        print(df1[df1['particle'] == id]['x raw'])

        data_store.df = df1
        data_store.save()
        """







if __name__ == "__main__":

    from ParticleTracking.tracking.tracking_gui import TrackingGui

    file = '/media/ppzmis/data/ActiveMatter/Microscopy/190709MRaggregates/videos/test2.mp4'
    #file = '/media/ppzmis/data/ActiveMatter/Microscopy/videosexample/videoDIC.avi'
    tracker = Bacteria(file, tracking=True)
    tracker.track()

    #gui = TrackingGui(tracker)



