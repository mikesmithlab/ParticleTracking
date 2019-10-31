from Generic.video import ReadVideo
from ParticleTracking import tracking, configurations, preprocessing, postprocessing,annotation, linking
from ParticleTracking import dataframes
import os.path
import numpy as np

PARAMETERS = {
    'crop method': 'no_crop',
    'preprocessor method': ('grayscale','adaptive_threshold'),
    'tracking method':('trackpy',),#track_big_blob',),
    'trackpy:size estimate':[5,1, 101,2],
    'trackpy:invert':[0,0,1,1],
    'postprocessor method': '',
    'annotation method': ('_draw_circles',),
    'circle:radius':10,
    'circle:cmap':(0,0,255),
    'circle:thickness':2,
    'adaptive threshold block size': [81, 3, 101, 2],
    'adaptive threshold C': [12, -30, 30, 1],
    'adaptive threshold mode': [1, 0, 1, 1],
    'max frame displacement': 50,
    'min frame life': 1,
    'memory': 3,
    'trajectory smoothing': 3,
    'noise cutoff': [5, 0, 100, 1],
    'single bacterium cutoff': [237, 100, 300, 1],
    'colors': {0:(0,0,0),
           1:(255,0,0),
           2:(0,255,0),
           3:(0,0,255),
           4:(255,255,0),
           5:(0,255,255),
           6:(255,0,255),
           7:(128,0,0),
           8:(0,128,0),
           9:(0,0,128),
           10:(128,128,0),
           11:(128,0,128),
           12:(0,128,128),
           13:(165,42,42),
           14:(255,69,0),
           15:(0,250,154),
           16:(32,178,170),
           17:(30,144,255),
           18:(139,0,128),
           19:(128,128,128)
           },
    'contour thickness': 2,
    'trajectory thickness': 2,
    'font size': 2,
    'fps': 5,
    'scale':1
    }



class Tracking:
    def __init__(self, video_filename=None):
        #Load video file, load dataframe, load config
        #create video object
        self.video_filename=video_filename
        self.filename=os.path.splitext(self.video_filename)[0]
        self.data_filename=self.filename + '.hdf5'



        ''' These should be overwritten in Daughter class'''
        #self.crop
        self.preprocess_select=False
        self.track_select=False
        self.link_select=False
        self.postprocess_select=False
        self.annotate_select=False

        self.parameters = {}


        #setup a datastore



    def _setup(self):
        '''Create classes that will be used'''
        self.cap = ReadVideo(filename=self.video_filename)
        self.frame=self.cap.read_next_frame()

        #if self.crop_select:
        #
        if self.preprocess_select:
            self.ip = preprocessing.Preprocessor(self.parameters, self.cap)
        else:
            self.ip = None
        if self.track_select:
            self.pt = tracking.ParticleTracker(parameters=self.parameters, preprocessor=self.ip, vidobject=self.cap, data_filename=self.data_filename)
        if self.link_select:
            self.link = linking.LinkTrajectory(data_filename=self.data_filename, parameters=self.parameters)
        if self.postprocess_select:
            self.pp = postprocessing.PostProcessor(data_filename=self.data_filename)
        if self.annotate_select:
            self.an = annotation.TrackingAnnotator(parameters=self.parameters, vidobject=self.cap, data_filename=self.data_filename)

    def preprocess(self):
        frame, self.boundary, cropped_frame=self.ip.process()

    def track(self):
        self.pt.track()

    def link(self):
        pass

    def postprocess(self):
        pass

    def annotate(self):
        pass

    def process(self):
        if self.track_select:
            self.pt.track()
        if self.postprocess_select:
            self.pp
        if self.annotate_select:
            self.an.annotate()


    def process_frame(self, frame_num):
        #For use with the TrackingGui
        frame=self.cap.find_frame(frame_num)
        if self.preprocess_select:
            newframe,_,_=self.ip.process(frame)
        else:
            newframe=frame
        if self.track_select:
            self.pt.track(f_index=frame_num)
        if self.postprocess_select:
            self.pp
        if self.annotate_select:
            annotatedframe=self.an.annotate(f_index=frame_num)
        else:
            annotatedframe=frame
        return newframe, annotatedframe



class Tracking_Daughter(Tracking):
    def __init__(self, video_filename=None):
        #Select operations to be performed

        Tracking.__init__(self, video_filename=video_filename)

        self.preprocess_select = True
        self.track_select = True
        self.postprocess_select = False
        self.annotate_select = True

        self.parameters = PARAMETERS

        self._setup()





if '__main__' == __name__:
    from ParticleTracking.tracking.tracking_gui import TrackingGui

    track = Tracking_Daughter(video_filename='/home/mike/Documents/HydrogelTest.m4v')
    track.process()
    #TrackingGui(track)