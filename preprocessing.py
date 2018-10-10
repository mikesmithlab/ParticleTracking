import Generic.video as video
import cv2
import numpy as np


class VideoPreprocessor:
    """Class to manage the preprocessing of videos"""

    def __init__(self, vid):
        self.vid = vid

    def crop_and_mask(self, no_of_sides=1, show=False):
        """ Crop an image and mask it to a certain shape"""
        frame = self.vid.read_next_frame()
        crop_inst = CropShape(frame, no_of_sides)
        mask_img, crop = crop_inst.begin_crop()
        mask_img = mask_img[:, :, 0]
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask_img)
        cropped_frame = masked_frame[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1], :]

        if show:
            cv2.imshow("mask_img", cropped_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


class CropShape:
    """ Take an interactive crop of a circle"""

    def __init__(self, input_image, no_of_sides=1):
        """Initialise with input image and the masking shape"""
        super().__init__()
        self.cropping = False
        self.refPt = []
        self.image = input_image
        self.no_of_sides = no_of_sides

    def click_and_crop(self, event, x, y, flags, param):
        """Method to manage the user cropping"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = [(x, y)]
            self.cropping = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.cropping = False
            if self.no_of_sides == 1:
                self.refPt.append((x, y))
                cx = (self.refPt[1][0] - self.refPt[0][0])/2 + self.refPt[0][0]
                cy = (self.refPt[1][1] - self.refPt[0][1])/2 + self.refPt[0][1]
                rad = int((self.refPt[1][0] - self.refPt[0][0])/2)
                cv2.circle(self.image, (int(cx), int(cy)), rad, (0, 255, 0), 2)
                cv2.imshow("image", self.image)

    def begin_crop(self):
        """Method to create the mask image and the crop region"""
        clone = self.image.copy()
        points = np.zeros((self.no_of_sides, 2))
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click_and_crop)
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
            cx = (self.refPt[1][0] - self.refPt[0][0])/2 + self.refPt[0][0]
            cy = (self.refPt[1][1] - self.refPt[0][1])/2 + self.refPt[0][1]
            rad = int((self.refPt[1][0] - self.refPt[0][0])/2)
            roi = clone[int(cy-rad):int(cy+rad), self.refPt[0][0]:self.refPt[1][0]]
            mask_img = np.zeros((np.shape(clone))).astype(np.uint8)
            cv2.circle(mask_img, (int(cx), int(cy)), rad, [1, 1, 1], thickness=-1)
            crop = [int(cy - rad), int(cy + rad)], [self.refPt[0][0], self.refPt[1][0]]

        else:
            roi = clone.copy()
            for n in np.arange(-1, len(points)-1):
                roi = cv2.line(roi, (int(points[n, 0]), int(points[n, 1])),
                               (int(points[n+1, 0]), int(points[n+1, 1])), [0, 255, 0], 2)
            mask_img = np.zeros(np.shape(clone)).astype('uint8')
            cv2.fillPoly(mask_img, pts=np.array([points], dtype=np.int32), color=(250, 250, 250))
            crop = [min(points[:, 1]), max(points[:, 1])], [min(points[:, 0]), max(points[:, 0])]
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("mask", mask_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return mask_img, np.array(crop, dtype=np.int32)




if __name__ == "__main__":
    vid = video.ReadVideo("/home/ppxjd3/Code/ParticleTracking/test_data/test_video.avi")
    VP = VideoPreprocessor(vid)
    VP.crop_and_mask(no_of_sides=4, show=True)