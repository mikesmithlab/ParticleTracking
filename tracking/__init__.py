import multiprocessing as mp
from multiprocessing.pool import ThreadPool, Pool
import os
import pandas as pd
import numpy as np
import trackpy
from tqdm import tqdm
import time

from Generic import images, video
from ParticleTracking import dataframes


class ParticleTracker:
    """
    Class to track the locations of the particles in a video.

    1) Uses preprocessing.Preprocessor to manipulate images.
    2) Uses method to locate the particles
    3) Confirms that each detected particle is real
    4) Saves particle positions and boundary information in a dataframe
    5) Saves a cropped copy of the video

    Attributes
    ----------
    input_filename : str
        Contains path of the input video
    filename: str
        Contains path of the input video without extension
    data_filename: str
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

    def __init__(self, multiprocess=False):
        """

        Parameters
        ----------
        filename: str
            Filepath of input video/stack

        methods: list of str
            Contains methods for Preprocessor

        parameters: dictionary
            Contains parameters for any functions

        multiprocess: Bool
            If true then splits the processing between pairs of threads

        crop_method: String or None
            Decides cropping method.
            None: no crop
            'auto': Automatically crop around blue hexagon
            'manual': Manually select cropping points
        """
        self.filename = os.path.splitext(self.input_filename)[0]
        self.multiprocess = multiprocess
        self.data_filename = self.filename + '.hdf5'
        cpus = mp.cpu_count()
        self.num_processes = cpus // 2 if self.multiprocess else 1


    def track(self):
        """Call this to start tracking"""
        self._get_video_info()
        if self.multiprocess:
            self._track_multiprocess()
        else:
            self._track_process(0)
        self._link_trajectories()
        self.save_crop()
        self.extra_steps()

    def save_crop(self):
        with dataframes.DataStore(self.data_filename) as data:
            crop = self.ip.crop
            data.add_metadata('crop', crop)

    def extra_steps(self):
        pass

    def _track_multiprocess(self):
        """Splits processing into chunks"""
        p = mp.Pool(self.num_processes)
        p.map(self._track_process, range(self.num_processes))
        p.close()
        p.join()
        self._cleanup_intermediate_dataframes()

    def _get_video_info(self):
        """
        Reads properties from the video for other methods:

        self.frame_jump_unit: int
            Number of frames for each process

        self.fps: int
            frames per second from the video

        self.width, self.height: ints
            width and height of processed frame

        self.duty_cycle: ndarray
            duty cycles for each frame in the video
        """
        cap = video.ReadVideo(self.input_filename)
        self.num_frames = cap.num_frames
        self.frame_div = self.num_frames // self.num_processes
        self.fps = cap.fps
        frame = cap.read_next_frame()
        new_frame, _, _ = self.ip.process(frame)
        self.width, self.height = images.get_width_and_height(new_frame)


    def _track_process(self, group_number):
        """
        Method called by track.

        If not using multiprocess call with group number 0

        Parameters
        ----------
        group_number: int
            Sets the group number for multiprocessing to split the input.
        """
        # Create the DataStore instance
        data_name = (str(group_number) + '.hdf5'
                     if self.multiprocess else self.data_filename)
        with dataframes.DataStore(data_name, load=False) as data:
            data.add_metadata('number_of_frames', self.num_frames)
            data.add_metadata('video_filename', self.input_filename)
            start = self.frame_div * group_number
            self.cap = video.ReadVideo(self.input_filename)
            self.cap.set_frame(start)
            if group_number == 3:
                missing = self.num_frames - 4*(self.num_frames//4)
                frame_div = self.frame_div + missing
            else:
                frame_div = self.frame_div
            # Iterate over frames
            for f in tqdm(range(frame_div), 'Tracking'):
                info, boundary, info_headings = self.analyse_frame()
                data.add_tracking_data(start + f, info, col_names=info_headings)
                if f == 0:
                    data.add_metadata('boundary', boundary)

    def _cleanup_intermediate_dataframes(self):
        """Concatenates and removes intermediate dataframes"""
        dataframe_list = ["{}.hdf5".format(i) for i in
                          range(self.num_processes)]
        dataframes.concatenate_datastore(dataframe_list,
                                         self.data_filename)
        for file in dataframe_list:
            os.remove(file)

    def _link_trajectories(self):
        """Implements the trackpy functions link_df and filter_stubs"""
        # Reload DataStore
        with dataframes.DataStore(self.data_filename) as data:
            # Trackpy methods
            data.reset_index()
            data.df = trackpy.link_df(
                    data.df,
                    self.parameters['max frame displacement'],
                    memory=self.parameters['memory'])

            data.df = trackpy.filter_stubs(
                data.df, self.parameters['min frame life'])
            data.set_frame_index()

    def update_parameters(self, parameters):
        self.parameters = parameters
        self.ip.update_parameters(self.parameters)


class ParticleTracker2:

    def __init__(self, multiprocess=False):
        self.filename = os.path.splitext(self.input_filename)[0]
        self.multiprocess = multiprocess
        self.data_filename = self.filename + '.hdf5'

    # def track(self):
    #     self._get_video_info()
    #     cap = video.ReadVideo(self.input_filename)
    #     self.data = dataframes.DataStore(self.data_filename, load=False)
    #     frames = cap.frames()
    #     if self.multiprocess:
    #         p = ThreadPool(4)
    #         p.map(self.analyse_frame,
    #               tqdm(frames, total=cap.num_frames))
    #         p.close()
    #         p.join()
    #     else:
    #         map(self.analyse_frame, tqdm(frames, total=cap.num_frames))
    #     print(len(np.unique(self.data.particle_data.frame)))
    #     # self._link_trajectories()
    #     # self.extra_steps()
    #     # print(self.data.inspect_dataframes())

    def track(self):
        self._get_video_info()
        cap = video.ReadVideo(self.input_filename)
        # self.data = dataframes.DataStore(self.data_filename, load=False)
        frames = cap.frames()
        self.data = pd.DataFrame()
        if self.multiprocess:
            p = ThreadPool(4)
            res = []
            for frame in tqdm(frames, 'track', total=cap.num_frames):
                r = p.apply_async(self.analyse_frame, (frame, ), callback=self.append_data)
                res.append(r)
                if len(res) > 50:
                    for r in res:
                        r.wait()
                    res = []
            for r in tqdm(res, 'wait'):
                r.wait()
            p.close()
            p.join()
        else:
            map(self.analyse_frame, tqdm(frames, total=cap.num_frames))
        self.data = self.data.set_index('frame')
        self.data = self.data.sort_index()
        print(self.data.head())
        # print(len(np.unique(self.data.particle_data.frame)))
        # self._link_trajectories()
        # self.extra_steps()
        # print(self.data.inspect_dataframes())

    def append_data(self, data):
        self.data = pd.concat([self.data, pd.DataFrame(data)])

    def add_data(self, data):
        print(data[1])
        circles = data[0]
        f = data[1]
        self.data.add_tracking_data(f, circles, self.headings)


    def _link_trajectories(self):
        """Implements the trackpy functions link_df and filter_stubs"""
        # Trackpy methods
        self.data.particle_data = trackpy.link_df(
            self.data.particle_data,
            self.parameters['max frame displacement'],
            memory=self.parameters['memory'])
        self.data.particle_data = trackpy.filter_stubs(
            self.data.particle_data, self.parameters['min frame life'])

        # Save DataStore
        self.data.save()

    def extra_steps(self):
        pass


    def _get_video_info(self):
        """
        Reads properties from the video for other methods:

        self.frame_jump_unit: int
            Number of frames for each process

        self.fps: int
            frames per second from the video

        self.width, self.height: ints
            width and height of processed frame

        self.duty_cycle: ndarray
            duty cycles for each frame in the video
        """
        cap = video.ReadVideo(self.input_filename)
        self.num_frames = cap.num_frames
        # self.frame_div = self.num_frames // self.num_processes
        self.fps = cap.fps
        frame = cap.read_next_frame()
        new_frame, _, _ = self.ip.process(frame)
        self.width, self.height = images.get_width_and_height(new_frame)