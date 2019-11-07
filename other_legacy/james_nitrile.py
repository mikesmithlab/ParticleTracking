from Generic import images, video, audio
from ParticleTracking.tracking import ParticleTracker2 as ParticleTracker
from ParticleTracking import preprocessing
from ParticleTracking.other_legacy import configurations
from ParticleTracking.general import dataframes
import numpy as np
from numba import jit
import matplotlib.path as mpath


class JamesPT(ParticleTracker):

    def __init__(self, filename, tracking=False, multiprocess=False):
        self.tracking = tracking
        self.parameters = configurations.NITRILE_BEADS_PARAMETERS
        self.ip = preprocessing.Preprocessor(self.parameters)
        self.input_filename = filename
        if self.tracking:
            ParticleTracker.__init__(self, multiprocess=multiprocess)
        else:
            self.cap = video.ReadVideo(self.input_filename)
            self.frame = self.cap.read_next_frame()

        self.headings = ('x', 'y', 'r')

    def analyse_frame(self, args):
        frame = args[0]
        f = args[1]
        # print(frame_and_f)
        # frame = frame_and_f[0]
        #
        # f = frame_and_f[1]
        # if self.tracking:
        #     frame = self.cap.read_next_frame()
        # else:
        #     frame = self.cap.find_frame(self.parameters['frame'][0])
        new_frame, boundary, cropped_frame = self.ip.process(frame)
        circles = images.find_circles(
            new_frame,
            self.parameters['min_dist'][0],
            self.parameters['p_1'][0],
            self.parameters['p_2'][0],
            self.parameters['min_rad'][0],
            self.parameters['max_rad'][0])
        circles = get_points_inside_boundary(circles, boundary)
        circles = check_circles_bg_color(circles, new_frame, 150)
        circles = {'x': circles[:, 0], 'y': circles[:, 1], 'r': circles[:, 2], 'frame': f}
        # self.data.add_tracking_data(f, circles, col_names=self.headings)
        return circles # pd.DataFrame(circles)
        # if self.tracking:
        #     return circles, boundary
        # else:
        #     annotated_frame = images.draw_circles(cropped_frame, circles)
        #     return new_frame, annotated_frame


    def extra_steps(self):
        duty_cycle = read_audio_file(self.input_filename, self.num_frames)
        data_store = dataframes.DataStore(self.data_filename,
                                          load=True)
        data_store.add_frame_property('Duty', duty_cycle)
        data_store.save()


@jit
def check_circles_bg_color(circles, image, threshold):
    """
    Checks the color of circles in an image and returns white ones

    Parameters
    ----------
    circles: ndarray
        Shape (N, 3) containing (x, y, r) for each circle
    image: ndarray
        Image with the particles in white

    Returns
    -------
    circles[white_particles, :] : ndarray
        original circles array with dark circles removed
    """
    circles = np.int32(circles)
    (x, y, r) = np.split(circles, 3, axis=1)
    r = int(np.mean(r))
    ymin = np.int32(np.squeeze(y-r/2))
    ymax = np.int32(np.squeeze(y+r/2))
    xmin = np.int32(np.squeeze(x-r/2))
    xmax = np.int32(np.squeeze(x+r/2))
    all_circles = np.zeros((r, r, len(xmin)))
    for i, (x0, x1, y0, y1) in enumerate(zip(xmin, xmax, ymin, ymax)):
        im = image[y0:y1, x0:x1]
        all_circles[0:im.shape[0], :im.shape[1], i] = im
    circle_mean_0 = np.mean(all_circles, axis=(0, 1))
    out = circles[circle_mean_0 > threshold, :]
    return out


def get_points_inside_boundary(points, boundary):
    """
    Returns the points from an array of input points inside boundary

    Parameters
    ----------
    points: ndarray
        Shape (N, 2) containing list of N input points
    boundary: ndarray
        Either shape (P, 2) containing P vertices
        or shape 3, containing cx, cy, r for a circular boundary

    Returns
    -------
    points: ndarray
        Shape (M, 2) containing list of M points inside the boundary
    """
    centers = points[:, :2]
    if len(np.shape(boundary)) == 1:
        vertices_from_centre = centers - boundary[0:2]
        points_inside_index = np.linalg.norm(vertices_from_centre, axis=1) < \
            boundary[2]
    else:
        path = mpath.Path(boundary)
        points_inside_index = path.contains_points(centers)
    points = points[points_inside_index, :]
    return points

def read_audio_file(file, frames):
    wav = audio.extract_wav(file)
    wav_l = wav[:, 0]
    # wav = audio.digitise(wav)
    freqs = audio.frame_frequency(wav_l, frames, 48000)
    d = (freqs - 1000)/2
    return d

if __name__ == '__main__':
    from Generic import filedialogs
    file = filedialogs.load_filename('Load a video')

    jpt = JamesPT(file, tracking=True, multiprocess=True)
    jpt.track()

