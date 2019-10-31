from ParticleTracking.project import PTWorkflow

#crop = {
#    'crop method': 'no_crop'
#    }

preprocess = {
    'preprocessor method': ('grayscale','adaptive_threshold'),
    'adaptive threshold block size': [81, 3, 101, 2],
    'adaptive threshold C': [12, -30, 30, 1],
    'adaptive threshold mode': [1, 0, 1, 1],
    }

track = {
    'track method':('trackpy',),
    'trackpy:size estimate':[5,1, 101,2],
    'trackpy:invert':[0,0,1,1]
    }

link = {
    'link method':'',
    'max frame displacement': 50,
    'min frame life': 1,
    'memory': 3,
    'trajectory smoothing': 3,
    }

postprocess = {
    'postprocess method': ''
    }

annotate = {
    'annotate method': ('_draw_circles',),
    'circle:radius':10,
    'circle:cmap':(0,0,255),
    'circle:thickness':2
    }



PARAMETERS = {
    #'crop':crop,
    'preprocess':preprocess,
    'track':track,
    'link':link,
    'postprocess':postprocess,
    'annotate':annotate
    }

class Example(PTWorkflow):
    def __init__(self, video_filename=None):
        #Select operations to be performed

        PTWorkflow.__init__(self, video_filename=video_filename)

        self.preprocess_select = True
        self.track_select = True
        self.postprocess_select = False
        self.annotate_select = True

        self.parameters = PARAMETERS

        self._setup()





if '__main__' == __name__:

    from ParticleTracking.tracking.tracking_gui import TrackingGui

    track = Example(video_filename='/home/mike/Documents/HydrogelTest.m4v')
    #track.process()
    TrackingGui(track)