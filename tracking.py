import os
import numpy as np
import trackpy as tp
import multiprocessing as mp
from Generic import video, images, audio
from ParticleTracking import preprocessing, dataframes, annotation
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit


class ParticleTracker:
    """
    Class to track the locations of the particles in a video.

    1) Uses preprocessing.Preprocessor to manipulate images.
    2) Uses houghcircles to locate the particles
    3) Confirms that each detected particle is real
    4) Saves particle positions and boundary information in a dataframe
    5) Saves a cropped copy of the video

    Attributes
    ----------
    video_filename : str
        Contains path of the input video
    filename: str
        Contains path of the input video without extension
    data_store_filename: str
        Contains path to save the dataframe to
    parameters: dict
        Contains arguments for preprocessing, detection, tracking
    multiprocess: bool
        If true using multiprocessing to speed up tracking
    save_crop_video: bool
        If true saves the cropped copy of the video
    save_check_video: bool
        If true saves the annoatated copy of the video
    ip: class instance of preprocessing.Preprocessor

    """

    def __init__(self,
                 filename,
                 methods,
                 parameters,
                 multiprocess=False,
                 auto_crop=False,
                 debug=False):

        self.filename = os.path.splitext(filename)[0]
        self.video_filename = self.filename + '.MP4'
        self.data_store_filename = self.filename + '.hdf5'
        self.parameters = parameters
        self.multiprocess = multiprocess
        self.num_processes = mp.cpu_count() // 2 if self.multiprocess else 1
        self.debug = debug
        self.ip = preprocessing.Preprocessor(
            methods, self.parameters, auto_crop)
        self._check_parameters()

    def _check_parameters(self):
        assert 'min_dist' in self.parameters, 'min_dist not in dictionary'
        assert 'p_1' in self.parameters, 'p_1 not in dictionary'
        assert 'p_2' in self.parameters, 'p_2 not in dictionary'
        assert 'min_rad' in self.parameters, 'min_rad not in dictionary'
        assert 'max_rad' in self.parameters, 'max rad not in dictionary'
        assert 'max frame displacement' in self.parameters, \
            'max frame displacement not in dictionary'
        assert 'memory' in self.parameters, 'memory not in dictionary'

    def track(self):
        self._get_video_info()
        if self.multiprocess:
            self._track_multiprocess()
        else:
            self._track_process(0)
        self._link_trajectories()
        crop = self.ip.crop
        np.savetxt(self.filename+'.txt', crop)

    def _track_multiprocess(self):
        """Call this to start tracking"""
        self.extension = "mp4"
        self.fourcc = "mp4v"
        p = mp.Pool(self.num_processes)
        p.map(self._track_process, range(self.num_processes))
        self._cleanup_intermediate_dataframes()

    def _get_video_info(self):
        """From the video reads properties for other methods"""
        cap = video.ReadVideo(self.video_filename)
        self.duty_cycle = read_audio_file(self.video_filename, cap.num_frames)
        self.frame_jump_unit = cap.num_frames // self.num_processes
        self.fps = cap.fps
        frame = cap.read_next_frame()
        new_frame, _ = self.ip.process(frame)
        self.width, self.height = images.get_width_and_height(new_frame)

    def _track_process(self, group_number):
        data_name = str(group_number)+'.hdf5' if self.multiprocess else self.data_store_filename
        data = dataframes.DataStore(data_name)
        cap = video.ReadVideo(self.video_filename)
        frame_no_start = self.frame_jump_unit * group_number
        cap.set_frame(frame_no_start)
        for f in tqdm(range(self.frame_jump_unit)):
            frame = cap.read_next_frame()
            new_frame, boundary = self.ip.process(frame)
            circles = images.find_circles(
                new_frame,
                self.parameters['min_dist'],
                self.parameters['p_1'],
                self.parameters['p_2'],
                self.parameters['min_rad'],
                self.parameters['max_rad'])
            circles = get_points_inside_boundary(circles, boundary)
            circles = check_circles_bg_color(circles, new_frame)
            data.add_tracking_data(frame_no_start + f,
                                   circles,
                                   boundary)
        data.save()
        cap.close()

    def _cleanup_intermediate_dataframes(self):
        """Concatenates and removes intermediate dataframes"""
        dataframe_list = ["{}.hdf5".format(i) for i in
                          range(self.num_processes)]
        dataframes.concatenate_datastore(dataframe_list,
                                         self.data_store_filename)
        for file in dataframe_list:
            os.remove(file)

    def _link_trajectories(self):
        """Implements the trackpy functions link_df and filter_stubs"""
        data_store = dataframes.DataStore(self.data_store_filename,
                                          load=True)
        data_store.add_frame_property('Duty', self.duty_cycle)
        data_store.particle_data = tp.link_df(
                data_store.particle_data,
                self.parameters['max frame displacement'],
                memory=self.parameters['memory'])
        data_store.particle_data = tp.filter_stubs(data_store.particle_data, self.parameters['min frame life'])
        data_store.save()


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


@jit
def check_circles_bg_color(circles, image):
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
    out = circles[circle_mean_0 > 200, :]
    return out




def read_audio_file(file, frames):
    wav = audio.extract_wav(file)
    wav_l = wav[:, 0]
    # wav = audio.digitise(wav)
    freqs = audio.frame_frequency(wav_l, frames, 48000)
    d = (freqs - 1000)/100
    return d

    # def _track_singleprocess(self):
    #     """Call this to start the tracking"""
    #     self.video = video.ReadVideo(self.video_filename)
    #     self.duty_cycle = read_audio_file(self.video_filename, self.video.num_frames)
    #     if os.path.exists(self.data_store_filename):
    #         os.remove(self.data_store_filename)
    #     data = dataframes.DataStore(self.data_store_filename)
    #     for f in tqdm(range(self.video.num_frames), 'Tracking'):
    #         frame = self.video.read_next_frame()
    #         new_frame, boundary = self.ip.process(frame)
    #         circles = images.find_circles(
    #             new_frame,
    #             self.parameters['min_dist'],
    #             self.parameters['p_1'],
    #             self.parameters['p_2'],
    #             self.parameters['min_rad'],
    #             self.parameters['max_rad'])
    #         circles = get_points_inside_boundary(circles, boundary)
    #         circles = check_circles_bg_color(circles, new_frame)
    #         if self.debug:
    #             temp = new_frame.copy()
    #             temp = images.draw_circles(np.dstack((temp, temp, temp)),
    #                                        circles)
    #             images.display(np.hstack((new_frame, temp)))
    #         data.add_tracking_data(f, circles, boundary)
    #     data.save()
    #     self._link_trajectories()

    # def _track_process(self, group_number):
    #     """
    #     The method which is mapped to the Pool implementing the tracking.
    #
    #     Finds the circles in a percentage of the video and saves the cropped
    #     video and dataframe for this part to the current working directory.
    #
    #     Parameters
    #     ----------
    #     group_number: int
    #         Describes which fraction of the video the method should act on
    #     """
    #     data = dataframes.DataStore(str(group_number) + '.hdf5')
    #     cap = video.ReadVideo(self.video_filename)
    #     frame_no_start = self.frame_jump_unit * group_number
    #     cap.set_frame(frame_no_start)
    #
    #     proc_frames = 0
    #     while proc_frames < self.frame_jump_unit:
    #         frame = cap.read_next_frame()
    #         new_frame, boundary = self.ip.process(frame)
    #         circles = images.find_circles(
    #             new_frame,
    #             self.parameters['min_dist'],
    #             self.parameters['p_1'],
    #             self.parameters['p_2'],
    #             self.parameters['min_rad'],
    #             self.parameters['max_rad'])
    #         circles = get_points_inside_boundary(circles, boundary)
    #         circles = check_circles_bg_color(circles, new_frame)
    #         data.add_tracking_data(frame_no_start+proc_frames,
    #                                circles,
    #                                boundary)
    #         proc_frames += 1
    #     data.save()
    #     cap.close()

if __name__ == "__main__":
    pass
    # vid_name = "../ParticleTracking/test_video.mp4"
    # methods = ['flip', 'threshold tozero', 'opening']
    # options = {
    #     'grayscale threshold': None,
    #     'number of tray sides': 6,
    #     'min_dist': 30,
    #     'p_1': 200,
    #     'p_2': 3,
    #     'min_rad': 15,
    #     'max_rad': 19,
    #     'max frame displacement': 25,
    #     'min frame life': 10,
    #     'memory': 8,
    #     'opening kernel': 23,
    #
    # }
    #
    # PT = ParticleTracker(vid_name,
    #                      methods,
    #                      options,
    #                      multiprocess=False,
    #                      save_crop_video=True,
    #                      save_check_video=True)
    # PT.track()
