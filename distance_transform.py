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

video = vid.ReadVideo("/home/ppxjd3/Videos/12240002.MP4")
prepro = pp.ImagePreprocessor()

frame = video.read_next_frame()
prepro._find_crop_and_mask_for_first_frame(frame, 1)
frame = prepro._crop_and_mask_frame(frame)

copy = frame.copy()

frame = prepro._grayscale_frame(frame)

frame = prepro._adaptive_threshold(frame, block_size=31)
mpl_figure(frame, 'gray')

frame = prepro._distance_transform(frame)
#mpl_figure(frame, 'distance transform')

frame = cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
#mpl_figure(frame, 'distance transform normalized')

frame = prepro._simple_threshold(frame, 50)
#mpl_figure(frame)

frame = frame.astype(np.uint8)

_, contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hull_list = []
for contour in contours:
    hull = cv2.convexHull(contour)
    if cv2.contourArea(hull) <500:
        cv2.drawContours(copy, [hull],  0, (0, 255, 0), 4)
mpl_figure(copy, 'minEnclosingCircle')