from Generic import images, video, audio
from ParticleTracking.tracking import ParticleTracker
from ParticleTracking import configurations, preprocessing, dataframes
import numpy as np
from numba import jit
import trackpy as tp


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
        with dataframes.DataStore(self.data_filename) as data:
            # Trackpy methods
            data.df = filter_near_edge(data.df, data.metadata['boundary'], 12)
            # data.reset_index()
            # data.df = tp.link(
            #     data.df,
            #     self.parameters['max frame displacement'],
            #     memory=self.parameters['memory'])
            # data.set_frame_index()


def read_audio_file(file, frames):
    wav = audio.extract_wav(file)
    wav_l = wav[:, 0]
    # wav = audio.digitise(wav)
    freqs = audio.frame_frequency(wav_l, frames, 48000)
    d = (freqs - 1000) / 15
    return d


def filter_near_edge(feat, boundary, threshold):
    print('filtering at edge')
    # Create line equations of form ax + by + c = 0 from corners
    line_ends = [[boundary[i-1, :], boundary[i, :]] for i in range(6)]
    a, b, c = zip(
        *[[(yb - ya) / (xa - xb), 1, xa * (ya - yb) / (xa - xb) - ya] for
          (xa, ya), (xb, yb) in line_ends])
    a, b, c = np.array(a, ndmin=2), np.array(b, ndmin=2), np.array(c, ndmin=2)

    # Get points
    points = feat[['x', 'y']].values
    x, y = points[:, 0, np.newaxis], points[:, 1, np.newaxis]

    # Find distance from each point to each line
    d = np.abs(x@a + y@b + c) / (np.sqrt(a**2 + b**2))

    # Find distance to closest line for each point
    d = np.min(d, axis=1)
    return feat[d > threshold]



if __name__ == '__main__':
    file = "/home/ppxjd3/Videos/short.MP4"

    jpt = TrackpyPT(file, tracking=True, multiprocess=False)
    jpt.track()
