from ParticleTracking.project import PTWorkflow

preprocess = {
    'crop method':'no_crop',
    'preprocessor method': ('grayscale','adaptive_threshold'),
    'adaptive threshold':{'block size': 81,#[81, 3, 101, 2],
                           'C': [12, -30, 30, 1],
                            'mode': [1, 0, 1, 1]
                          }
    }

track = {
    'track method':'trackpy',
    'trackpy':{'size estimate':[19,1, 101,2],
                'invert':[0,0,1,1]
               }
    }

link = {
    'link method':'',
    'max frame displacement': 50,
    'min frame life': 1,
    'memory': 3,
    'trajectory smoothing': 3,
    }

postprocess = {
    'postprocess method': None
    }

annotate = {
    'annotate method': ('circles',),
    'circles':{'radius':10,
               'cmaptype':'discrete',#'continuous'
               'cmap':{0:(0,0,255),0.5:(0,255,0)},#{0:0}
               'thickness':2
               }
    }

PARAMETERS = {
    #'crop':crop,
    'preprocess':preprocess,
    'track':track,
    'link':link,
    'postprocess':postprocess,
    'annotate':annotate
    }

class PTProject(PTWorkflow):
    '''
    PTProject is a daughter class which is used as the interface to a particle tracking project.

    Setup:
    Select which bits of the process you are interested in by setting the Boolean
    Operators. You attach a dictionary PARAMETERS which controls all of the settings.
    It is up to you to make sure you have the correct parameters.

    Create an "instance" of the class:
    track=PTProject(video_filename="Full Path To File")

    There are 2 modes of operation:
    1) Pass the instance as an argument to TrackGui to optimise/trial things
    track = PTProject(video_filename='/home/mike/Documents/HydrogelTest.m4v')
    TrackingGui(track)
    2) call "instance".process() to process an entire movie.
    track = PTProject(video_filename='/home/mike/Documents/HydrogelTest.m4v')
    track.process()

    You can select the various parts of the operation by setting the flags to self.process_select
    self.preprocess_select = True
    self.track_select = True
    self.postprocess_select = False
    self.annotate_select = True

    What these processes will do is governed by the respective parts of the PARAMETERS dictionary above
    '''

    def __init__(self, video_filename=None):
        #Select operations to be performed

        PTWorkflow.__init__(self, video_filename=video_filename)

        self.preprocess_select = True
        self.track_select = False
        self.postprocess_select = False
        self.annotate_select = False

        self.parameters = PARAMETERS

        self._setup()



if '__main__' == __name__:

    from ParticleTracking.tracking.tracking_gui import TrackingGui

    track = PTProject(video_filename='/home/mike/Documents/HydrogelTest.m4v')
    #track.process()
    TrackingGui(track)