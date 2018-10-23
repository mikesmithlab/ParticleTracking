import matplotlib.pyplot as plt
import matplotlib as mpl
import Generic.video as vid
mpl.rc('figure', figsize=(10, 5))
mpl.rc('image', cmap='gray')

import numpy as np
import pandas as pd
import trackpy as tp
import pims
import cv2
import Generic.simple_funcs as sf
import ParticleTracking.preprocessing as pp

def mpl_figure(im, title=''):
    plt.figure()
    plt.imshow(im)
    plt.title(title)
    plt.show()


options = {'config': 1,
           'title': 'Glass_Bead',
           'grayscale threshold': 100,
           'adaptive threshold block size': 11,
           'adaptive threshold C': 2,
           'blur kernel': 5,
           'min_dist': 20,
           'p_1': 200,
           'p_2': 10,
           'min_rad': 18,
           'max_rad': 20,
           'number of tray sides': 1,
           'max frame displacement': 10,
           'min frame life': 5}

method_order = ['grayscale', 'simple threshold',
                'adaptive threshold', 'gaussian blur']



video = vid.ReadVideo("/home/ppxjd3/Videos/12240002.MP4")
processor = pp.ImagePreprocessor(video, method_order, options)
frame = video.read_next_frame()
crop_inst = pp.CropShape(frame, 6)
frame = crop_inst.begin_crop()
copy = frame.copy()
copy2 = frame.copy()



mpl_figure(frame)

bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#_, bw = cv2.threshold(bw, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
bw = cv2.adaptiveThreshold(
    bw,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    5)
mpl_figure(bw, 'bw')


dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
mpl_figure(dist, 'dist0')


# Threshold to obtain the peaks
# This will be the markers for the foreground objects
_, dist = cv2.threshold(dist, 0.0, 1.0, cv2.THRESH_BINARY)
kernel1 = np.ones((3,3), dtype=np.uint8)
dist = cv2.dilate(dist, kernel1)
mpl_figure(dist, 'dist')

# Create the CV_8U version of the distance image
# It is needed for findContours()
dist_8u = dist.astype('uint8')

# Find total markers
_, contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hull_list = []
for contour in contours:
    hull = cv2.convexHull(contour)
    cv2.drawContours(copy, [hull],  0, (0, 255, 0), 4)

mpl_figure(copy)

for contour in contours:
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(copy2, center, radius, (0, 255, 0), 2)
mpl_figure(copy2, 'minEnclosingCircle')



