from Generic.video import ReadVideo
from ParticleTracking import tracking, preprocessing, postprocessing,annotation, linking
import os.path


class PTWorkflow:
    '''
    PTWorkflow is a parent class that handles the workflow of a project.
    
    '''
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

    def _setup(self):
        '''Create classes that will be used'''
        self.cap = ReadVideo(filename=self.video_filename)
        self.frame=self.cap.read_next_frame()

        #if self.crop_select:
        #
        if self.preprocess_select:
            self.ip = preprocessing.Preprocessor(self.parameters['preprocess'], self.cap)
        else:
            self.ip = None
        if self.track_select:
            self.pt = tracking.ParticleTracker(parameters=self.parameters['track'], preprocessor=self.ip, vidobject=self.cap, data_filename=self.data_filename)
        if self.link_select:
            self.link = linking.LinkTrajectory(data_filename=self.data_filename, parameters=self.parameters['link'])
        if self.postprocess_select:
            self.pp = postprocessing.PostProcessor(data_filename=self.data_filename, parameters=self.parameters['postprocess'])
        if self.annotate_select:
            self.an = annotation.TrackingAnnotator( vidobject=self.cap, data_filename=self.data_filename, parameters=self.parameters['annotate'])

    def process(self):
        if self.track_select:
            self.pt.track()
        if self.postprocess_select:
            self.pp.process()
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

