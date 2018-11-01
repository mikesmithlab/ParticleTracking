import Generic.video as video
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
                new_frame = im.opening(
                    new_frame,
                    kernel=(self.options['opening kernel'],
                            self.options['opening kernel']))

            elif method == 'simple threshold':
                new_frame = im.threshold(
                    new_frame,
                    self.options['grayscale threshold'])

            elif method == 'adaptive threshold':
                new_frame = im.adaptive_threshold(
                    new_frame,
                    block_size=(self.options['adaptive threshold block size']))

            elif method == 'gaussian blur':
                new_frame = im.gaussian_blur(
                    new_frame,
                    kernel=(self.options['blur kernel'],
                            self.options['blur kernel']))

            elif method == 'distance transform':
                new_frame = im.distance_transform(new_frame)

            elif method == 'closing':
                new_frame = im.closing(
                    new_frame,
                    kernel=(self.options['closing kernel'],
                            self.options['closing kernel']))

            elif method == 'opening':
                new_frame = im.opening(
                    new_frame,
                    kernel=(self.options['opening kernel'],
                            self.options['opening kernel']))

            elif method == 'dilation':
                new_frame = im.dilate(
                    new_frame,
                    kernel=(self.options['dilate kernel'],
                            self.options['dilate kernel']))

            elif method == 'erosion':
                new_frame = im.erode(
                    new_frame,
                    kernel=(self.options['erode kernel'],
                            self.options['erode kernel']))


        self.process_calls += 1
        return new_frame, cropped_frame, self.boundary

    def update_options(self, options, methods):
        self.options = options
        self.method_order = methods

    def _find_crop_and_mask_for_first_frame(self, frame, no_of_sides=1):
        if self.options:
            no_of_sides = self.options['number of tray sides']
        crop_inst = CropShape(frame, no_of_sides)
        self.mask_img, self.crop, self.boundary = crop_inst.begin_crop()
        if np.shape(self.boundary) == (3,):
            self.boundary[0] -= self.crop[1][0]
            self.boundary[1] -= self.crop[0][0]
        else:
            self.boundary[:, 0] -= self.crop[1][0]
            self.boundary[:, 1] -= self.crop[0][0]

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



class CropShape:
    """ Take an interactive crop of a circle"""

    def __init__(self, input_image, no_of_sides=1):
        """
        Initialise with input image and the number of sides:

        Parameters
        ----------
        input_image: 2D numpy array
            contains an image
        no_of_sides: int
            The number of sides that the desired shape contains.
                1: Uses a circle
                >2: Uses a polygon

        """

        super().__init__()
        self.cropping = False
        self.refPt = []
        self.image = input_image
        self.no_of_sides = no_of_sides
        self.scale = 1280/im.get_height(self.image)
        self.original_image = self.image.copy()
        self.image = im.resize(self.image, self.scale*100)

    def _click_and_crop(self, event, x, y, flags, param):
        """Internal method to manage the user cropping"""

        if event == cv2.EVENT_LBUTTONDOWN:
            # x is across, y is down
            self.refPt = [(x, y)]
            self.cropping = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.cropping = False
            if self.no_of_sides == 1:
                self.refPt.append((x, y))
                cx = ((self.refPt[1][0] - self.refPt[0][0]) / 2 +
                      self.refPt[0][0])
                cy = ((self.refPt[1][1] - self.refPt[0][1]) / 2 +
                      self.refPt[0][1])
                rad = int((self.refPt[1][0] - self.refPt[0][0]) / 2)
                cv2.circle(self.image, (int(cx), int(cy)), rad, (0, 255, 0), 2)
                cv2.imshow(str(self.no_of_sides), self.image)

    def begin_crop(self):
        """Method to create the mask image and the crop region"""

        clone = self.image.copy()
        points = np.zeros((self.no_of_sides, 2))
        cv2.namedWindow(str(self.no_of_sides))
        cv2.setMouseCallback(str(self.no_of_sides), self._click_and_crop)
        count = 0

        # keep looping until 'q' is pressed
        while True:
            cv2.imshow(str(self.no_of_sides), self.image)
            key = cv2.waitKey(1) & 0xFF

            if self.cropping and self.no_of_sides > 1:
                points[count, 0] = self.refPt[0][0]
                points[count, 1] = self.refPt[0][1]
                self.cropping = False
                count += 1

            if key == ord("r"):
                self.image = clone.copy()
                count = 0
                points = np.zeros((self.no_of_sides, 2))

            elif key == ord("c"):
                break

        cv2.destroyAllWindows()

        if self.no_of_sides == 1:
            cx = (self.refPt[1][0] - self.refPt[0][0]) / 2 + self.refPt[0][0]
            cy = (self.refPt[1][1] - self.refPt[0][1]) / 2 + self.refPt[0][1]
            rad = int((self.refPt[1][0] - self.refPt[0][0]) / 2)
            cx = int(cx/self.scale)
            cy = int(cy/self.scale)
            rad = int(rad/self.scale)
            mask_img = np.zeros((np.shape(self.original_image))).astype(np.uint8)
            cv2.circle(mask_img, (cx, cy), rad, [1, 1, 1], thickness=-1)
            crop = ([int(cy - rad), int(cy + rad)],
                    [int(self.refPt[0][0]/self.scale), int(self.refPt[1][0]/self.scale)])
            boundary = np.array((cx, cy, rad), dtype=np.int32)
            return mask_img[:, :, 0], np.array(crop, dtype=np.int32), boundary

        else:
            points = points/self.scale
            mask_img = np.zeros(np.shape(self.original_image)).astype('uint8')
            cv2.fillPoly(mask_img, pts=np.array([points], dtype=np.int32),
                         color=(250, 250, 250))
            crop = ([min(points[:, 1]), max(points[:, 1])],
                    [min(points[:, 0]), max(points[:, 0])])
            return mask_img[:, :, 0], np.array(crop, dtype=np.int32), points


if __name__ == "__main__":
    vid = video.ReadVideo(
        "/home/ppxjd3/Code/ParticleTracking/test_data/test_video_EDIT.avi")
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
