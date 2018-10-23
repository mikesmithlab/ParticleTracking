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

video = vid.ReadVideo("/home/ppxjd3/Videos/test_video_EDIT.avi")
frame = video.read_next_frame()


original = frame.copy()

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ret, frame = cv2.threshold(frame, 120, 255, cv2.THRESH_BINARY)
frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, np.ones((3,3), dtype=np.uint8))

plt.figure()
plt.imshow(frame)

f = tp.locate(frame, diameter=29, invert=True, minmass=25000, separation=25)



fig, ax = plt.subplots()
ax.hist(f['mass'], bins=20)

# Optionally, label the axes.
ax.set(xlabel='mass', ylabel='count')



plt.figure()
tp.annotate(f, original)

