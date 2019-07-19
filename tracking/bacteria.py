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
        self.parameters = configurations.BACTERIA_PARAMETERS
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
                             'classifier','split flag']
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

        if area <= self.parameters['noise cutoff'][0]*self.parameters['area bacterium'][0]:
            if area <= self.parameters['noise floor']:
                return None
            else:
                #classify small things as noise. We do not store
                classifier = int(0)
                info_bacterium.append(classifier)
                info_bacterium.append(False)
            return info_bacterium
        else:
            '''
            Here we attempt to classify whether it is a 
            single bacterium, dividing or just stuck together.
            Method:
            Single Bacterium has a given area. 
            Dividing bacterium was only a single object in all previous frames
            Stuck bacterium consisted of 2 or more objects in previous frame.
            '''
            if (area > self.parameters['noise cutoff'][0]*self.parameters['area bacterium'][0]) & (area <= self.parameters['single bacterium cutoff'][0]*self.parameters['area bacterium'][0]):
                classifier = int(1)  # single bacterium - Blue
                info_bacterium.append(classifier)
                info_bacterium.append(True) #specifies that bacterium has been observed as individual.
                # overwrite box cx and cy with contour cx and cy
                #info_bacterium[0], info_bacterium[1] = images.find_contour_centre(contour)
            else: #area > 1.8*self.parameters['area bacterium'][0]:

                if self.tracking:
                    #This splitting relies on historical info so won't work with gui.

                    info_bacterium = self._split_bacteria(cropped_frame, new_frame, info_bacterium)
                else:
                    #In gui both dividing and stuck are classified as 4 and are not split
                    classifier = 4
                    info_bacterium.append(classifier)
                    info_bacterium.append(False)

            return info_bacterium




    def _split_bacteria(self,frame,bwframe, info_bacterium):
        box = info_bacterium[5]

        if self.old_info is not None:

            #True for all frames except first

            # Check for points from previous frame inside bounding box and add to a list of ids
            pts=[]
            for index,row in enumerate(self.old_info):
                output = cv2.pointPolygonTest(box, (row[0], row[1]), True)
                if output >= self.parameters['outside cutoff']:
                    pts.append(index)

            if pts == []:
                info_bacterium = images.rotated_bounding_rectangle(
                    box)
                classifier = 4
                info_bacterium.append(classifier)
                info_bacterium.append(False)
                return info_bacterium
            elif len(pts) == 1:
                #Either dividing or not classified previously
                if self.old_info[pts[0]][7]:
                    classifier = 2
                    info_bacterium = images.rotated_bounding_rectangle(
                        box)
                    info_bacterium.append(classifier)
                    info_bacterium.append(True)
                else:
                    classifier = 4
                    info_bacterium.append(classifier)
                    info_bacterium.append(False)
                return info_bacterium
            else:
                '''
                If more than 1 then these points were previously separate
                They are therefore currently stuck together and we need to separate them.
                The method is to mask the image with the box of the other bacterium's bounding
                boxes from the previous frame. Then calculate the contour on what remains.
                '''
                #cut out binary parallel bounding box for original rotated box
                bwim,r=images.cut_out_object(bwframe,box, setsurroundblack=True)


                newcontours = []


                for currentpt in pts:
                    #for a point in the last frame
                    bwim2 = bwim.copy()

                    mask = np.zeros(np.shape(bwim), dtype=np.uint8)
                    orig_contour = [np.array(self.old_info[currentpt][5]) - np.array([r[0], r[1]])]

                    for pt in pts:
                        if pt != currentpt:
                            contour = [np.array(self.old_info[pt][5]) - np.array([r[0],r[1]])]
                            mask = images.draw_contours(mask, contour, col=255, thickness=-1)

                            #mask other bacteria in binary image using rotated bounding boxes from previous image

                    mask = images.draw_contours(mask, orig_contour, col=0, thickness=-1)
                    bwim2 = bwim2*mask


                    #find the contours in new masked image. Sort
                    mini_contour = images.sort_contours(images.find_contours(bwim2))


                    #Shift coordinates of contours back to original image
                    #for contour in mini_contour:
                    newcontours.append(mini_contour[-1] + np.array([r[0],r[1]]))

                stuck_bacterium_info = None
                for contour in newcontours:#index in range(np.size(newcontours)):

                    info_bacterium = images.rotated_bounding_rectangle(contour)#newcontours[index])
                    area = info_bacterium[3] * info_bacterium[4]

                    '''
                    Here we attempt to classify whether it is a bit of noise
                    single bacterium, dividing or a clump of them
                    
                    '''
                    if area < self.parameters['noise floor']:
                        return None
                    elif (area <= self.parameters['noise cutoff'][0]*self.parameters['area bacterium'][0]):
                        classifier = int(0)
                        info_bacterium.append(classifier)
                        info_bacterium.append(False)
                    elif (area > self.parameters['noise cutoff'][0] * self.parameters['area bacterium'][0]) & \
                            (area <= self.parameters['single bacterium cutoff'][0]*self.parameters['area bacterium'][0]):
                        classifier = int(1)  # single bacterium - Blue
                        info_bacterium.append(classifier)
                        info_bacterium.append(True)

                        if stuck_bacterium_info is None:
                            stuck_bacterium_info = [info_bacterium]
                        else:
                            stuck_bacterium_info.append(info_bacterium)
                    else:
                        classifier = int(4)  # Problem with classification
                        info_bacterium.append(classifier)
                        info_bacterium.append(False)
                        return info_bacterium

                #If you only have one contour in the above for loop then it is
                #possible to create a list of lists. We therefore correct this here
                if np.shape(stuck_bacterium_info) == (1,8):
                    return stuck_bacterium_info[0]
                else:
                    return stuck_bacterium_info

        else:
            #This code is executed for the first frame only when old_info is None
            classifier = 4
            info_bacterium.append(classifier)
            info_bacterium.append(False)
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

    file = '/media/ppzmis/data/ActiveMatter/Microscopy/videosexample/videoDIC.avi'
    tracker = Bacteria(file, tracking=False)
    #tracker.track()


    gui = TrackingGui(tracker)



