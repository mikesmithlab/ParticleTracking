import cv2
import numpy as np
from Generic import images

"""
Additional processing methods should be added under if statements in process.
To call the new method its name should be added to the methods list.
Any parameters for new methods should come from the parameters dictionary.
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

        # Perform the crop for every frame
        if self.crop_method is not None:
            cropped_frame = self._crop_and_mask(~frame)
            new_frame = images.bgr_2_grayscale(~cropped_frame)
        else:
            new_frame = images.bgr_2_grayscale(frame)

        # Perform each method in the method list
        for method in self.parameters['method']:

            if method == 'opening':
                kernel = self.parameters['opening kernel'][0]
                new_frame = images.opening(
                    new_frame,
                    kernel=(kernel, kernel))

            elif method == 'flip':
                new_frame = ~new_frame

            elif method == 'threshold tozero':
                threshold = self.parameters['grayscale threshold'][0]
                new_frame = images.threshold(
                    new_frame, threshold, cv2.THRESH_TOZERO)

            elif method == 'simple threshold':
                threshold = self.parameters['grayscale threshold'][0]
                new_frame = images.threshold(
                    new_frame,
                    threshold)

            elif method == 'adaptive threshold':
                block = self.parameters['adaptive threshold block size'][0]
                const = self.parameters['adaptive threshold C'][0]
                new_frame = images.adaptive_threshold(
                    new_frame,
                    block_size=block,
                    constant=const)

            elif method == 'gaussian blur':
                kernel = self.parameters['blur kernel'][0]
                new_frame = images.gaussian_blur(
                    new_frame,
                    kernel=(kernel, kernel))

            elif method == 'distance transform':
                new_frame = images.distance_transform(new_frame)

            elif method == 'closing':
                kernel = self.parameters['closing kernel'][0]
                kernel_arr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (kernel, kernel))
                new_frame = images.closing(
                    new_frame,
                    kernel=kernel_arr)

            elif method == 'opening':
                kernel = self.parameters['opening kernel'][0]
                new_frame = images.opening(
                    new_frame,
                    kernel=(kernel, kernel),
                    kernel_type=cv2.MORPH_ELLIPSE)

            elif method == 'dilation':
                kernel = self.parameters['dilate kernel'][0]
                new_frame = images.dilate(
                    new_frame,
                    kernel=(kernel, kernel))

            elif method == 'erosion':
                kernel = self.parameters['erode kernel'][0]
                new_frame = images.erode(
                    new_frame,
                    kernel=(kernel, kernel))

            elif method == 'distance':
                new_frame = images.distance_transform(new_frame)

            elif method == 'resize':
                new_frame = images.resize(new_frame, 50)

            else:
                print("method is not defined")

        self.calls += 1
        return new_frame, self.boundary

    def _crop_and_mask(self, frame):
        """
        Masks then crops a given frame

        Takes a frame and uses a bitwise_and operation with the input mask_img
        to mask the image around a shape.
        It then crops the image around the mask.

        Parameters
        ----------
        frame: numpy array
            A numpy array of an image of type uint8

        Returns
        -------
        cropped_frame: numpy array
            A numpy array containing an image which has been cropped and masked
        """
        masked_frame = images.mask_img(frame, self.mask_img)
        cropped_frame = images.crop_img(masked_frame, self.crop)
        return cropped_frame


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
