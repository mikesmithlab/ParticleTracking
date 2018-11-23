import Generic.video as vid
import cv2
import numpy as np
import ParticleTracking.configuration as config
import Generic.images as im

class ImagePreprocessor:
    """Class to manage the frame by frame processing of videos"""

    def __init__(self, method_order=None, options=None):
        """
        Parameters
        ----------
        method_order: list
            A list containing string assoicated with methods in the order they
            will be used.
            If None, process_image will only perform a grayscale of the image.

        options: dictionary
            A dictionary containing all the parameters needed for functions.
            If None, methods will use their keyword parameters

        """

        self.mask_img = np.array([])
        self.crop = []
        self.method_order = method_order
        self.options = options
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
        if self.method_order:
            method_order = self.method_order
        else:
            method_order = []

        if self.process_calls == 0:
            self._find_crop_and_mask_for_first_frame(frame)
        cropped_frame = self._crop_and_mask_frame(frame)
        new_frame = cropped_frame.copy()
        new_frame = im.bgr_2_grayscale(new_frame)
        for method in method_order:
            if method == 'opening':
                try:
                    kernel = self.options['opening kernel']
                except KeyError as error:
                    print(error)
                    print('opening kernel set to 3')
                    kernel = 3
                new_frame = im.opening(
                    new_frame,
                    kernel=(kernel, kernel))

            elif method == 'flip':
                new_frame = ~new_frame

            elif method == 'threshold tozero':
                try:
                    threshold = self.options['grayscale threshold']
                except KeyError as error:
                    print(error, 'threshold set to 100')
                    threshold = 100
                new_frame = im.threshold(new_frame, threshold, cv2.THRESH_TOZERO)

            elif method == 'simple threshold':
                try:
                    threshold = self.options['grayscale threshold']
                except KeyError as error:
                    print(error, 'threshold set to 100')
                    threshold = 100
                new_frame = im.threshold(
                    new_frame,
                    threshold)

            elif method == 'adaptive threshold':
                try:
                    block = self.options['adaptive threshold block size']
                except KeyError as error:
                    print(error, 'adaptive block size set to 31')
                    block = 31
                try:
                    const = self.options['adaptive threshold C']
                except KeyError as error:
                    print(error, 'constant set to 0')
                    const = 0
                new_frame = im.adaptive_threshold(
                    new_frame,
                    block_size=block,
                    constant=const)

            elif method == 'gaussian blur':
                try:
                    kernel = self.options['blur kernel']
                except KeyError as error:
                    print(error, 'kernel set to 3')
                new_frame = im.gaussian_blur(
                    new_frame,
                    kernel=(kernel, kernel))

            elif method == 'distance transform':
                new_frame = im.distance_transform(new_frame)

            elif method == 'closing':
                try:
                    kernel = self.options['closing kernel']
                except KeyError as error:
                    print(error, 'kernel set to 3')
                    kernel = 3
                kernel_arr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (kernel, kernel))
                new_frame = im.closing(
                    new_frame,
                    kernel=kernel_arr)

            elif method == 'opening':
                try:
                    kernel = self.options['opening kernel']
                except KeyError as error:
                    print(error, 'kernel set to 3')
                    kernel = 3
                new_frame = im.opening(
                    new_frame,
                    kernel=(kernel, kernel))

            elif method == 'dilation':
                try:
                    kernel = self.options['dilate kernel']
                except KeyError as error:
                    print(error, 'kernel set to 3')
                new_frame = im.dilate(
                    new_frame,
                    kernel=(kernel, kernel))

            elif method == 'erosion':
                try:
                    kernel = self.options['erode kernel']
                except KeyError as error:
                    print(error, 'kernel set to 3')
                new_frame = im.erode(
                    new_frame,
                    kernel=(kernel, kernel))
            elif method == 'distance':
                new_frame = im.distance_transform(new_frame)


        self.process_calls += 1
        return new_frame, cropped_frame, self.boundary

    def update_options(self, options, methods):
        """Updates class variables"""
        self.options = options
        self.method_order = methods

    def _find_crop_and_mask_for_first_frame(self, frame, no_of_sides=1):
        """
        Opens a crop shape instance with the input frame and no_of_sides

        Sets the class variables self.mask_img, self.crop and self.boundary
        """
        if self.options:
            no_of_sides = self.options['number of tray sides']
        crop_inst = im.CropShape(frame, no_of_sides)
        self.mask_img, self.crop, self.boundary = crop_inst.begin_crop()
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

        crop: a int list
            A list in the format [ymin, ymax], [xmin, xmax] where x and y are
            the points selected when you click during manual cropping

        mask_img: numpy array
            A 2D array with the same dimensions as frame which is used to mask
            the image

        Returns
        -------
        cropped_frame: numpy array
            A numpy array containing an image which has been cropped and masked
        """
        masked_frame = im.mask_img(frame, self.mask_img)
        cropped_frame = im.crop_img(masked_frame, self.crop)
        return cropped_frame






if __name__ == "__main__":
    input_vid = vid.ReadVideo(
        "/home/ppxjd3/Videos/test.mp4")
    ml = config.MethodsList('Glass_Bead', load=True)
    process_config = ml.extract_methods()
    config_df = config.ConfigDataframe()
    options = config_df.get_options('Glass_Bead')
    IP = ImagePreprocessor(process_config, options)
    for f in range(2):
        frame = vid.read_next_frame()
        new_frame, crop_frame, boundary = IP.process_image(frame)
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
        im.display(new_frame)
