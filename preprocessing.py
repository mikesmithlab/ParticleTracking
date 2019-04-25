import cv2
import numpy as np
from Generic import images
from ParticleTracking import preprocessing_methods as pm

"""
Additional processing methods should be added under if statements in process.
To call the new method its name should be added to the methods list.
Any parameters for new methods should come from the parameters dictionary.

Add any methods you add to the dictionary METHODS as the key with any
needed parameters as the items of the list.
"""

METHODS = {'opening': ['opening kernel'],
           'flip': None,
           'threshold tozero': ['grayscale threshold'],
           'simple threshold': ['grayscale threshold'],
           'adaptive threshold': ['adaptive threshold block size',
                                  'adaptive threshold C'],
           'gaussian blur': ['blur kernel'],
           'closing': ['closing kernel'],
           'opening': ['opening kernel'],
           'dilation': ['dilate kernel'],
           'erosion': ['erode kernel'],
           'distance': None,
           'resize': ['resize scale']
           }


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

    update(parameters, methods)
        Updates the instance attributes `parameters` and `methods`

    Notes
    -----
    Methods are implemented by strings in `methods`.
    Methods implemented are with the necessary keys from `parameters`:

    'simple threshold' = simple grayscale threshold
        Uses `grayscale threshold` (int 0 - 255)

    'threshold tozero' = grayscale threshold tozero
        Uses `grayscale threshold` key (int 0 - 255)

    'adaptive threshold' = Gaussian adaptive theshold
        Uses `adaptive threshold block size' (odd int)
        Uses 'adaptive threshold C' (int)

    'flip' = reverses the colours

    'opening' = performs a morphological opening
        Uses 'opening kernel' (odd int)

    'closing' = performs a morphological closing
        Uses 'closing kernel' (odd int)

    'dilation' = performs a dilation on the image
        Uses 'dilate kernel' (odd int)

    'erosion' = performs an erosion on the image
        Uses 'erode kernel' (odd int)

    'distance transform' = distance transform

    """

    def __init__(self, parameters):
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
        self.parameters = parameters
        self.crop_method = self.parameters['crop method']
        self.mask_img = np.array([])
        self.crop = []
        self.boundary = None

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
        if self.calls == 0:
            if self.crop_method == 'blue hex':
                self.crop, self.mask_img, self.boundary = \
                    find_auto_crop_and_mask(frame)
            elif self.crop_method == 'manual':
                self.crop, self.mask_img, self.boundary = \
                    find_manual_crop_and_mask(frame)
            elif self.crop_method is None:
                pass
            self.parameters['crop'] = self.crop
            self.parameters['mask image'] = self.mask_img

        # Perform the crop for every frame
        if self.crop_method is not None:
            new_frame = pm.crop_and_mask(~frame, self.parameters)
            cropped_frame = new_frame.copy()
            new_frame = images.bgr_2_grayscale(~new_frame)
        else:
            cropped_frame = frame.copy()
            new_frame = images.bgr_2_grayscale(frame)

        # Perform each method in the method list
        for method in self.parameters['method']:
            new_frame = getattr(pm, method)(new_frame, self.parameters)

        self.calls += 1
        return new_frame, self.boundary, cropped_frame


def find_manual_crop_and_mask(frame):
    """
    Opens a crop shape instance with the input frame and no_of_sides

    Sets the class variables self.mask_img, self.crop and self.boundary
    """
    no_of_sides = int(input('Enter no of sides'))
    crop_inst = images.CropShape(frame, no_of_sides)
    mask_img, crop, boundary, _ = crop_inst.find_crop_and_mask()

    if np.shape(boundary) == (3,):
        # boundary = [xc, yc, r]
        # crop = ([xmin, ymin], [xmax, ymax])
        boundary[0] -= crop[0][0]
        boundary[1] -= crop[0][1]
    else:
        # boundary = ([x1, y1], [x2, y2], ...)
        # crop = ([xmin, ymin], [xmax, ymax])
        boundary[:, 0] -= crop[0][0]
        boundary[:, 1] -= crop[0][1]

    return crop, mask_img, boundary


def find_auto_crop_and_mask(frame):
    blue = images.find_color(frame, 'Blue')
    contours = images.find_contours(blue)
    contours = images.sort_contours(contours)
    # hex_corners = fit_hexagon_to_contour(contours[-2])
    hex_corners = images.fit_hex(np.squeeze(contours[-2]))
    sketch = images.draw_polygon(frame.copy(), hex_corners, thickness=2)
    # images.display(sketch)
    mask_img = np.zeros(np.shape(frame)).astype('uint8')
    cv2.fillPoly(mask_img, pts=np.array([hex_corners], dtype=np.int32),
                 color=(1, 1, 1))
    crop = ([int(min(hex_corners[:, 0])), int(min(hex_corners[:, 1]))],
            [int(max(hex_corners[:, 0])), int(max(hex_corners[:, 1]))])
    boundary = hex_corners
    boundary[:, 0] -= crop[0][0]
    boundary[:, 1] -= crop[0][1]
    return crop, mask_img, boundary
