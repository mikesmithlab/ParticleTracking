import os
import numpy as np
import trackpy as tp
import multiprocessing as mp
from Generic import video, images, audio
from ParticleTracking import dataframes

from tqdm import tqdm


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
        self.num_processes = mp.cpu_count() // 2 if self.multiprocess else 1

    def track(self):
        """Call this to start tracking"""
        self._get_video_info()
        if self.multiprocess:
            self._track_multiprocess()
        else:
            self._track_process(0)
        self._link_trajectories()
        self.extra_steps()
        crop = self.ip.crop
        np.savetxt(self.filename+'.txt', crop)

    def extra_steps(self):
        pass

    def _track_multiprocess(self):
        """Splits processing into chunks"""
        p = mp.Pool(self.num_processes)
        p.map(self._track_process, range(self.num_processes))
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
        new_frame, _ = self.ip.process(frame)
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
        data_name = (str(group_number)+'.hdf5'
                     if self.multiprocess else self.data_filename)
        data = dataframes.DataStore(data_name, load=False)

        start = self.frame_div * group_number
        self.cap = video.ReadVideo(self.input_filename)
        self.cap.set_frame(start)
        # Iterate over frames
        for f in tqdm(range(self.frame_div)):
            info, boundary, info_headings = self._analyse_frame()
            data.add_tracking_data(start+f, info, col_names=info_headings)
            data.add_boundary_data(start+f, boundary)
        data.save()

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
        data_store = dataframes.DataStore(self.data_filename,
                                          load=True)
        # Trackpy methods
        data_store.particle_data = tp.link_df(
                data_store.particle_data,
                self.parameters['max frame displacement'],
                memory=self.parameters['memory'])
        data_store.particle_data = tp.filter_stubs(
                data_store.particle_data, self.parameters['min frame life'])

        # Save DataStore
        data_store.save()



