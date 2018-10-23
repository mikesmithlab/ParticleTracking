import matplotlib.pyplot as plt
import matplotlib as mpl
import Generic.video as vid
mpl.rc('figure', figsize=(10, 5))
mpl.rc('image', cmap='gray')

import numpy as np
import trackpy as tp
import cv2
import ParticleTracking.preprocessing as pp


def mpl_figure(im, title=''):
    plt.figure()
    plt.imshow(im)
    plt.title(title)
    plt.show()

prepro = pp.ImagePreprocessor()
video = vid.ReadVideo("/home/ppxjd3/Videos/12240002.MP4")
frame = video.read_next_frame()
prepro._find_crop_and_mask_for_first_frame(frame, 1)
frame = prepro._crop_and_mask_frame(frame)

original = frame.copy()

frame = prepro._grayscale_frame(frame)
frame = prepro._adaptive_threshold(frame, block_size=31, constant=0)


mpl_figure(frame)

f = tp.locate(frame, diameter=33, invert=True)



fig, ax = plt.subplots()
ax.hist(f['mass'], bins=20)

# Optionally, label the axes.
ax.set(xlabel='mass', ylabel='count')



plt.figure()
tp.annotate(f, original)

