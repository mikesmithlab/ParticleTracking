from Generic import video, images
from ParticleTracking import preprocessing
import numpy as np
import cv2
import time

def load_im():
    # file = "/home/ppxjd3/Videos/Test_camera_audio/22030003.MP4"
    file = "/home/ppxjd3/Videos/Testing Auto Crop/22010001.MP4"
    vid = video.ReadVideo(file)
    frame = vid.read_next_frame()
    crop, mask, boundary = preprocessing.find_auto_crop_and_mask(frame)
    masked_frame = images.mask_img(~frame, mask)
    cropped_frame = images.crop_img(masked_frame, crop)
    return cropped_frame

def divide_im(im):
    gray = images.bgr_2_grayscale(im)
    thresh = images.threshold(gray, 195)
    opened = images.opening(thresh, kernel=(15, 15), kernel_type=cv2.MORPH_ELLIPSE)
    sure_bg = images.dilate(opened, iterations=3)
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    sure_fg = images.threshold(dist, 0.7*dist.max())
    sure_fg = np.uint8(sure_fg)
    return sure_fg, sure_bg

def find_markers(fg, bg):
    unknown = cv2.subtract(bg, fg)
    ret, markers = cv2.connectedComponents(fg)
    markers += 1
    markers[unknown == 255] = 0
    return markers

def watershed(im, markers):
    markers = cv2.watershed(im, markers)
    return markers

def find_circles(im):
    t = time.time()
    fg, bg = divide_im(im)
    print(time.time() - t)
    t = time.time()
    markers = find_markers(fg, bg)
    markers = watershed(im, markers) + 1
    print(time.time() - t, 'markers')
    t = time.time()
    new_markers = images.threshold(np.uint8(markers), 2)


    closed = images.opening(new_markers, kernel=(17, 17),
                            kernel_type=cv2.MORPH_ELLIPSE)
    contours = images.find_contours(closed)
    circles = np.array([cv2.minEnclosingCircle(cnt) for cnt in contours])
    circles = [[circles[i][0][0], circles[i][0][1], circles[i][1]]
               for i in range(len(circles))]
    return circles


im = load_im()
t = time.time()
circles = find_circles(im)
print(time.time() - t)
im = images.draw_circles(im, circles)
images.display(im)




