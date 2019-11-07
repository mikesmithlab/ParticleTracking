import cv2
import numpy as np
from Generic import images
from ParticleTracking.preprocessing import preprocessing_methods as pm
from ParticleTracking.preprocessing import preprocessing_crops as pc

"""
All processing methods should be added as functions in preprocessing_methods

To call the function its name as a string should be added to the methods
tuple in configurations.

Any parameters needed for a function should be added to the dictionaries in 
configurations, taking care not to duplicate any names.
"""


class Preprocessor:
    """
    Processes images using a given set of instructions.

    Attributes
    ----------
    methods : list of str
        The names of methods implemented

    parameters : dict
        Contains the arguments needed for each method

    crop_method : str
        None = No crop
        'auto' = auto crop
        'manual' = manual crop

    crop : array_like
        ((xmin, ymin), (xmax, ymax)) which bounds the roi

    mask_img : ndarray
        An array containing an image mask

    calls : int
        Records the number of calls to process

    Methods
    -------
    process(frame)
        Processes the supplied frame and returns the new frame
        and the selected boundary

    update_parameters(parameters)
        Updates the instance attributes `parameters`

    """

    def __init__(self, parameters, cap, return_all=True):
        """
        Parameters
        ----------
        methods: list of str, optional
            A list containing string associated with methods in the order they
            will be used.
            If None, process will only perform a grayscale of the image.

        parameters: dictionary, optional
            A dictionary containing all the parameters needed for functions.
            If None, methods will use their default parameters

        crop_method: str or None:
            If None then no crop takes place
            If 'blue hex' then uses auto crop function
            If 'manual' then uses manual crop function
        """
        self.cap = cap

        self.parameters = parameters
        self.crop_method = self.parameters['crop method']
        self.mask_img = np.array([])
        self.crop = []
        self.boundary = None
        self.return_all = return_all

        self.calls = 0

    def update_parameters(self, parameters):
        self.parameters = parameters

    def process(self, frame):
        """
        Manipulates an image using class methods.

        The order of the methods is described by self.methods

        Parameters
        ----------
        frame: numpy array
            uint8 numpy array containing an image

        Returns
        -------
        new_frame: numpy array
            uint8 numpy array containing the new image

        boundary: numpy array
            Contains information about the boundary points
        """

        # Find the crop for the first frame
        #if self.calls == 0:
        self.crop, self.mask_img, self.boundary = \
            getattr(pc, self.parameters['crop method'])(frame)
        self.parameters['crop'] = self.crop
        self.parameters['mask image'] = self.mask_img

        # Perform each method in the method list
        cropped_frame = frame.copy()

        for method in self.parameters['preprocessor method']:
            # Use function in preprocessing_methods
            frame = getattr(pm, method)(frame, self.parameters)
            if method == 'crop_and_mask':
                cropped_frame = frame.copy()
        self.calls += 1
        if self.return_all:
            return frame, self.boundary, cropped_frame
        else:
            return frame
