import cv2
import numpy as np
from Generic import images, video


class ImagePreprocessor:
    """Class to manage the frame by frame processing of videos"""

    def __init__(self, methods=None, parameters=None, crop_points=None):
        """
        Parameters
        ----------
        methods: list
            A list containing string associated with methods in the order they
            will be used.
            If None, process_image will only perform a grayscale of the image.

        parameters: dictionary
            A dictionary containing all the parameters needed for functions.
            If None, methods will use their keyword parameters

        """

        self.mask_img = np.array([])
        self.crop_points = crop_points
        self.crop = []
        self.methods = methods
        self.parameters = parameters
        self.process_calls = 0

    def process_image(self, frame):
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
        if self.methods:
            method_order = self.methods
        else:
            method_order = []

        if self.process_calls == 0:
            self._find_crop_and_mask(frame)
        cropped_frame = self._crop_and_mask_frame(frame)
        new_frame = images.bgr_2_grayscale(cropped_frame)
        for method in method_order:
            if method == 'opening':
                try:
                    kernel = self.parameters['opening kernel']
                except KeyError as error:
                    print(error)
                    print('opening kernel set to 3')
                    kernel = 3
                new_frame = images.opening(
                    new_frame,
                    kernel=(kernel, kernel))

            elif method == 'flip':
                new_frame = ~new_frame

            elif method == 'threshold tozero':
                try:
                    threshold = self.parameters['grayscale threshold']
                except KeyError as error:
                    print(error, 'threshold set to 100')
                    threshold = 100
                new_frame = images.threshold(
                    new_frame, threshold, cv2.THRESH_TOZERO)

            elif method == 'simple threshold':
                try:
                    threshold = self.parameters['grayscale threshold']
                except KeyError as error:
                    print(error, 'threshold set to 100')
                    threshold = 100
                new_frame = images.threshold(
                    new_frame,
                    threshold)

            elif method == 'adaptive threshold':
                try:
                    block = self.parameters['adaptive threshold block size']
                except KeyError as error:
                    print(error, 'adaptive block size set to 31')
                    block = 31
                try:
                    const = self.parameters['adaptive threshold C']
                except KeyError as error:
                    print(error, 'constant set to 0')
                    const = 0
                new_frame = images.adaptive_threshold(
                    new_frame,
                    block_size=block,
                    constant=const)

            elif method == 'gaussian blur':
                try:
                    kernel = self.parameters['blur kernel']
                except KeyError as error:
                    print(error, 'kernel set to 3')
                    kernel = 3
                new_frame = images.gaussian_blur(
                    new_frame,
                    kernel=(kernel, kernel))

            elif method == 'distance transform':
                new_frame = images.distance_transform(new_frame)

            elif method == 'closing':
                try:
                    kernel = self.parameters['closing kernel']
                except KeyError as error:
                    print(error, 'kernel set to 3')
                    kernel = 3
                kernel_arr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (kernel, kernel))
                new_frame = images.closing(
                    new_frame,
                    kernel=kernel_arr)

            elif method == 'opening':
                try:
                    kernel = self.parameters['opening kernel']
                except KeyError as error:
                    print(error, 'kernel set to 3')
                    kernel = 3
                new_frame = images.opening(
                    new_frame,
                    kernel=(kernel, kernel),
                    kernel_type=cv2.MORPH_ELLIPSE)

            elif method == 'dilation':
                try:
                    kernel = self.parameters['dilate kernel']
                except KeyError as error:
                    print(error, 'kernel set to 3')
                    kernel = 3
                new_frame = images.dilate(
                    new_frame,
                    kernel=(kernel, kernel))

            elif method == 'erosion':
                try:
                    kernel = self.parameters['erode kernel']
                except KeyError as error:
                    print(error, 'kernel set to 3')
                    kernel = 3
                new_frame = images.erode(
                    new_frame,
                    kernel=(kernel, kernel))
            elif method == 'distance':
                new_frame = images.distance_transform(new_frame)

        self.process_calls += 1
        return new_frame, self.boundary

    def update_options(self, options, methods):
        """Updates class variables"""
        self.parameters = options
        self.methods = methods

    def _find_crop_and_mask(self, frame, no_of_sides=1):
        """
        Opens a crop shape instance with the input frame and no_of_sides

        Sets the class variables self.mask_img, self.crop and self.boundary
        """
        if self.parameters:
            no_of_sides = self.parameters['number of tray sides']
        crop_inst = images.CropShape(frame, no_of_sides)
        if self.crop_points is None:
            self.mask_img, self.crop, self.boundary, _ = \
                crop_inst.begin_crop()
        else:
            self.mask_img, self.crop, self.boundary, _ = \
                crop_inst.find_crop_and_mask(self.crop_points)
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

    def _crop_and_mask_frame(self, frame):
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


if __name__ == "__main__":
    import Generic.filedialogs as fd
    input_vid_name = fd.load_filename('Choose a video')
    input_vid = video.ReadVideo(input_vid_name)
    methods = []
    options = {}
    IP = ImagePreprocessor(methods, options)
    for f in range(2):
        frame = input_vid.read_next_frame()
        new_frame, boundary = IP.process_image(frame)
        if np.shape(boundary) == (3,):
            new_frame = cv2.circle(new_frame,
                                   (boundary[0], boundary[1]),
                                   boundary[2],
                                   (255, 0, 255),
                                   2)
        else:
            # convert boundary list of points from (n, 2) to (n, 1, 2)
            boundary = boundary.reshape((-1, 1, 2))
            boundary = boundary.astype(np.int32)
            cv2.polylines(new_frame, [boundary], True, (0, 0, 255))
        images.display(new_frame)
