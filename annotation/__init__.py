from Generic import video
from ParticleTracking.annotation import annotation_methods as am
from tqdm import tqdm
from ParticleTracking.general import dataframes


class TrackingAnnotator:#video.Annotator):

    def __init__(self, parameters=None, vidobject=None, data_filename=None, bitrate='HIGH1080', framerate=50):
        self.parameters = parameters
        self.cap = vidobject
        self.data_filename=data_filename
        self.output_filename = self.cap.filename[:-4] + '_annotate.mp4'
        self.out = video.WriteVideoFFMPEG(self.output_filename,
                                          bitrate=bitrate, framerate=framerate)

    def annotate(self, f_index=None):
        with dataframes.DataStore(self.data_filename, load=True) as data:
            if f_index is None:
                start=0
                stop=self.cap.num_frames
            else:
                start=f_index
                stop=f_index+1
            self.cap.set_frame(start)

            for f in tqdm(range(start, stop, 1), 'Annotating'):
                frame = self.cap.read_next_frame()

                for method in self.parameters['annotate method']:
                    # Use function in preprocessing_methods
                    frame = getattr(am, method)(frame, data, f, self.parameters)
                if f_index is None:
                    self.out.add_frame(frame)
            if f_index is None:
                self.out.close()
            else:
                return frame



