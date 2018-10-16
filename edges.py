""" Test canny edge detection"""
import cv2
import numpy as np
import Generic.video as video

def cv2im(image, title=''):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class BoundaryDetection:

    def __init__(self):
        pass

    def find_boundary(self, frame):
        minArea = 60000
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_out = frame.copy()
        img = cv2.GaussianBlur(img, (5, 5), 0)
        cv2im(img, 'blur')
        img = cv2.adaptiveThreshold(img,
                                    255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY,
                                    13,
                                    5)
        cv2im(img, 'thresh')
        kernel = np.ones((3, 3), dtype=np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2im(img, 'close')
        img = cv2.erode(img, kernel, iterations=1)
        cv2im(img, 'erode')
        img2, contours, hierarchy = cv2.findContours(img,
                                                     cv2.RETR_LIST,
                                                     cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        for i, contour in enumerate(contours):
            hull = cv2.convexHull(contour)
            if cv2.contourArea(hull) > minArea:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                img_out = cv2.circle(img_out, (int(x), int(y)), int(radius), (255, 255, 0), 2)
        cv2im(img_out, 'out')





if __name__=="__main__":
    vid = video.ReadVideo("/media/ppxjd3/Nathan Backup Data V1/2015/Raw Data/Acrylic 8mm/Varying energy/2.5g/Set2/50_stitched.avi")
    #vid = video.ReadVideo("/media/ppxjd3/Nathan Backup Data V1/2018/Diffusion/6mm-p1200-757mV/15220002.MP4")
    BD = BoundaryDetection()
    BD.find_boundary(vid.read_next_frame())
