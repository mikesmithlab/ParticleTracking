import cv2
import numpy as np
from Generic import images, video


class Preprocessor:
    """
    Processes images using a given set of instructions.

    Attributes
    ----------
    methods : list of str
        The names of methods implemented

    parameters : dict
        Contains the arguments needed for each method

    crop_points : array_like
        Contains the points which are selected when cropping.

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

    def __init__(self, methods=[], parameters={}, auto_crop=False):
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

        crop_points: array_like, optional
            shape (N, 2) containing the points selected by the crop.
            Give this parameter to skip the manual cropping step
        """

        self.mask_img = np.array([])
        self.crop = []
        self.auto_crop = auto_crop
        self.methods = methods
        self.parameters = parameters
        self.calls = 0

    def process(self, frame):
        """
        Manipulates an image using class methods.

        The order of the methods is described by self.method_order.

        Parameters
        ----------
        frame: numpy array
            uint8 numpy array containing an image

        Returns
        -------
        new_frame: numpy array
            uint8 numpy array containing the new image

        cropped_frame: numpy array
            uint8 numpy array containing the cropped and masked image

        self.boundary: numpy array
            Contains information about the boundary points
        """

        if self.calls == 0:
            if self.auto_crop:
                self.crop, self.mask_img, self.boundary = \
                    find_auto_crop_and_mask(frame)
            else:
                self._find_manual_crop_and_mask(frame)
        cropped_frame = self._crop_and_mask(~frame)
        new_frame = images.bgr_2_grayscale(~cropped_frame)

        for method in self.methods:

            if method == 'opening':
                kernel = self.parameters['opening kernel']
                new_frame = images.opening(
                    new_frame,
                    kernel=(kernel, kernel))

            elif method == 'flip':
                new_frame = ~new_frame

            elif method == 'threshold tozero':
                threshold = self.parameters['grayscale threshold']
                new_frame = images.threshold(
                    new_frame, threshold, cv2.THRESH_TOZERO)

            elif method == 'simple threshold':
                threshold = self.parameters['grayscale threshold']
                new_frame = images.threshold(
                    new_frame,
                    threshold)

            elif method == 'adaptive threshold':
                block = self.parameters['adaptive threshold block size']
                const = self.parameters['adaptive threshold C']
                new_frame = images.adaptive_threshold(
                    new_frame,
                    block_size=block,
                    constant=const)

            elif method == 'gaussian blur':
                kernel = self.parameters['blur kernel']
                new_frame = images.gaussian_blur(
                    new_frame,
                    kernel=(kernel, kernel))

            elif method == 'distance transform':
                new_frame = images.distance_transform(new_frame)

            elif method == 'closing':
                kernel = self.parameters['closing kernel']
                kernel_arr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (kernel, kernel))
                new_frame = images.closing(
                    new_frame,
                    kernel=kernel_arr)

            elif method == 'opening':
                kernel = self.parameters['opening kernel']
                new_frame = images.opening(
                    new_frame,
                    kernel=(kernel, kernel),
                    kernel_type=cv2.MORPH_ELLIPSE)

            elif method == 'dilation':
                kernel = self.parameters['dilate kernel']
                new_frame = images.dilate(
                    new_frame,
                    kernel=(kernel, kernel))

            elif method == 'erosion':
                kernel = self.parameters['erode kernel']
                new_frame = images.erode(
                    new_frame,
                    kernel=(kernel, kernel))

            elif method == 'distance':
                new_frame = images.distance_transform(new_frame)

            else:
                print("method is not defined")

        self.calls += 1
        return new_frame, self.boundary

    def update(self, parameters, methods):
        """Updates class variables"""
        self.parameters = parameters
        self.methods = methods

    def _find_manual_crop_and_mask(self, frame):
        """
        Opens a crop shape instance with the input frame and no_of_sides

        Sets the class variables self.mask_img, self.crop and self.boundary
        """
        no_of_sides = self.parameters['number of tray sides']
        crop_inst = images.CropShape(frame, no_of_sides)

        self.mask_img, self.crop, self.boundary, _ = \
            crop_inst.find_crop_and_mask()

        if np.shape(self.boundary) == (3,):
            # boundary = [xc, yc, r]
            # crop = ([xmin, ymin], [xmax, ymax])
            self.boundary[0] -= self.crop[0][0]
            self.boundary[1] -= self.crop[0][1]
        else:
            # boundary = ([x1, y1], [x2, y2], ...)
            # crop = ([xmin, ymin], [xmax, ymax])
            self.boundary[:, 0] -= self.crop[0][0]
            self.boundary[:, 1] -= self.crop[0][1]

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


if __name__ == "__main__":
    import cv2
    from Generic import images
    test_image = cv2.imread("../ParticleTracking/test_image.png")
    methods = ['flip', 'threshold tozero', 'adaptive threshold', 'opening',
               'closing', 'dilation', 'erosion', 'distance transform']
    parameters = {'grayscale threshold': 200,
                  'adaptive threshold block size': 31,
                  'adaptive threshold C': 0,
                  'opening kernel': 17,
                  'closing kernel': 17,
                  'dilate kernel': 17,
                  'erode kernel': 17,
                  'number of tray sides': 6}
    ImPro = Preprocessor(methods, parameters)
    new_image, boundary = ImPro.process(test_image)
    images.display(new_image)