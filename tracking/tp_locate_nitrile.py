from Generic import images, video, audio
from ParticleTracking.tracking import ParticleTracker
from ParticleTracking import configurations, preprocessing, dataframes
import numpy as np
from numba import jit
import trackpy as tp
import matplotlib.path as mpath


class TrackpyPT(ParticleTracker):

    def __init__(self, filename, tracking=False, multiprocess=False):
        self.tracking = tracking
        self.parameters = configurations.TRACKPY_NITRILE_PARAMETERS
        self.ip = preprocessing.Preprocessor(self.parameters)
        self.input_filename = filename
        if self.tracking:
            ParticleTracker.__init__(self, multiprocess=multiprocess)
        else:
            self.cap = video.ReadVideo(self.input_filename)
            self.frame = self.cap.read_next_frame()

    def analyse_frame(self):
        if self.tracking:
            frame = self.cap.read_next_frame()
        else:
            frame = self.cap.find_frame(self.parameters['frame'][0])
        new_frame, boundary, cropped_frame = self.ip.process(frame)
        circles = tp.locate(new_frame, 21, characterize=False)
        circles['r'] = 15
        if self.tracking:
            return circles[['x', 'y', 'r']].values, boundary, ['x', 'y', 'r']
        else:
            annotated_frame = images.draw_circles(cropped_frame, circles)
            return new_frame, annotated_frame

    def extra_steps(self):
        duty_cycle = read_audio_file(self.input_filename, self.num_frames)
        duty_cycle = np.uint16(duty_cycle)
        with dataframes.DataStore(self.data_filename) as data:
            data.add_frame_property('Duty', duty_cycle)
            data.save()

    def _link_trajectories(self):
        pass


def read_audio_file(file, frames):
    wav = audio.extract_wav(file)
    wav_l = wav[:, 0]
    # wav = audio.digitise(wav)
    freqs = audio.frame_frequency(wav_l, frames, 48000)
    d = (freqs - 1000) / 15
    return d

if __name__ == '__main__':
    from Generic import filedialogs
    file = "/home/ppxjd3/Videos/short.mp4"

    jpt = TrackpyPT(file, tracking=True, multiprocess=False)
    jpt.track()
