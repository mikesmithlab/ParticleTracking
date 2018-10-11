import Generic.video as video
import cv2
import numpy as np


class ImagePreprocessor:
    """Class to manage the frame by frame processing of videos"""

    def __init__(self, video_object, no_of_crop_sides=1):
        self.no_of_crop_sides = no_of_crop_sides
        self.video_object = video_object
        self.mask_img = np.array([])
        self.crop = []
        self.blur_kernel = 5

    def process_image(self, input_frame):
        frame = input_frame
        if self.video_object.frame_num == 1:
            crop_inst = CropShape(frame, self.no_of_crop_sides)
            self.mask_img, self.crop = crop_inst.begin_crop()
        new_frame = self._crop_and_mask_frame(frame)
        new_frame = self._grayscale_frame(new_frame)
        new_frame = self._simple_threshold(new_frame, 100)
        new_frame = self._adaptive_threshold(new_frame)
        new_frame = self._gaussian_blur(new_frame)
        return new_frame

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
        masked_frame = cv2.bitwise_and(frame, frame, mask=self.mask_img)
        cropped_frame = masked_frame[self.crop[0][0]:self.crop[0][1],
                                     self.crop[1][0]:self.crop[1][1],
                                     :]
        return cropped_frame

    def _grayscale_frame(self, frame):
        """Make the a frame grayscale"""
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return new_frame

    def _simple_threshold(self, frame, threshold):
        """Perform a binary threshold of a frame"""
        ret, new_frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
        return new_frame

    def _adaptive_threshold(self, frame):
        new_frame = cv2.adaptiveThreshold(frame, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
        return new_frame

    def _gaussian_blur(self, frame):
        new_frame = cv2.GaussianBlur(frame,
                                     (self.blur_kernel, self.blur_kernel),
                                     0)
        return new_frame


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

    def _click_and_crop(self, event, x, y, flags, param):
        """Internal method to manage the user cropping"""
        if event == cv2.EVENT_LBUTTONDOWN:
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
                cv2.imshow("image", self.image)

    def begin_crop(self):
        """Method to create the mask image and the crop region"""
        clone = self.image.copy()
        points = np.zeros((self.no_of_sides, 2))
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self._click_and_crop)
        count = 0

        # keep looping until 'q' is pressed
        while True:
            cv2.imshow("image", self.image)
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
            roi = clone[int(cy - rad):int(cy + rad),
                        self.refPt[0][0]:self.refPt[1][0]]
            mask_img = np.zeros((np.shape(clone))).astype(np.uint8)
            cv2.circle(mask_img, (int(cx), int(cy)), rad, [1, 1, 1],
                       thickness=-1)
            crop = ([int(cy - rad), int(cy + rad)],
                    [self.refPt[0][0], self.refPt[1][0]])

        else:
            roi = clone.copy()
            for n in np.arange(-1, len(points) - 1):
                roi = cv2.line(roi, (int(points[n, 0]), int(points[n, 1])),
                               (int(points[n + 1, 0]), int(points[n + 1, 1])),
                               [0, 255, 0], 2)
            mask_img = np.zeros(np.shape(clone)).astype('uint8')
            cv2.fillPoly(mask_img, pts=np.array([points], dtype=np.int32),
                         color=(250, 250, 250))
            crop = ([min(points[:, 1]), max(points[:, 1])],
                    [min(points[:, 0]), max(points[:, 0])])
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return mask_img[:, :, 0], np.array(crop, dtype=np.int32)


if __name__ == "__main__":
    vid = video.ReadVideo(
        "/home/ppxjd3/Code/ParticleTracking/test_data/test_video_EDIT.avi")
    IP = ImagePreprocessor(vid, no_of_crop_sides=1)
    frame = vid.read_next_frame()
    new_frame = IP.process_image(frame)
    cv2.imshow("new_frame", new_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    frame = vid.read_next_frame()
    new_frame = IP.process_image(frame)
    cv2.imshow("new_frame", new_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()