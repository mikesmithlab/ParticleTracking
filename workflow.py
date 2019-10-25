from Generic.video import ReadVideo
from ParticleTracking import tracking, configurations, preprocessing, postprocessing,annotation, linking
from ParticleTracking import dataframes
import os.path

PARAMETERS = {
    'crop method': 'no_crop',
    'preprocessor methods': ('grayscale','adaptive_threshold'),
    'adaptive threshold block size': [61, 3, 101, 2],
    'adaptive threshold C': [-17, -30, 30, 1],
    'adaptive threshold mode': [0, 0, 1, 1],
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
        if self.preprocess_select or self.track_select or self.annotate_select:
            self.frame=self.cap.read_next_frame()

        #if self.crop_select:
        #
        if self.preprocess_select:
            self.ip = preprocessing.Preprocessor(self.parameters)
        if self.track_select:
            self.pt = tracking.ParticleTracker()
        if self.link_select:
            self.link = linking.LinkTrajectory(data_filename=self.data_filename, parameters=self.parameters)
        if self.postprocess_select:
            self.pp = postprocessing.PostProcessor()
        if self.annotate_select:
            self.an = annotation.TrackingAnnotator()

    def preprocess(self):
        pass

    def track(self):
        pass

    def link(self):

    def postprocess(self):
        pass

    def annotate(self):
        pass

    def process(self,start=None,stop=None):
        self.start=start
        self.stop=stop

        if self.preprocess_select:
            self.preprocess()
        if self.track_select:
            self.track()
        if self.postprocess_select:
            self.postprocess()
        if self.annotate_select:
            self.annotate()


class Tracking_Daughter(Tracking):
    def __init__(self, video_filename=None):
        #Select operations to be performed

        Tracking.__init__(self, video_filename=video_filename)

        self.preprocess_select = True
        self.track_select = False
        self.postprocess_select = False
        self.annotate_select = False

        self.parameters = PARAMETERS

        self._setup()

    def process_frame(self, frame_num):
        self.process(start=frame_num,stop=frame_num)



if '__main__' == __name__:
    from ParticleTracking.tracking.tracking_gui import TrackingGui

    track = Tracking_Daughter(video_filename='/home/mike/Documents/HydrogelTest.m4v')
    #track.process()
    TrackingGui(track)