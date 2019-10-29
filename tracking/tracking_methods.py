import cv2
from Generic import images
from Generic.images.basics import display
import numpy as np
import skimage
import trackpy as tp

def distance_transform(frame, parameters):
    dist = cv2.distanceTransform(frame, cv2.DIST_L2, 5)
    display(dist/np.max(dist))
    return dist

def track_big_blob(frame, parameters):
    contours = images.find_contours(frame)

    for index, contour in enumerate(contours):
        info = classify(contour, cropped_frame, frame)
        info.append(info)

    info = images.rotated_bounding_rectangle(contour)
    info = list(zip(*info))
    info_headings = ['x', 'y', 'r', 'box']
    framenum = framenum + 1
    return info, boundary, info_headings

def trackpy(frame, parameters):
    df = tp.locate(frame, parameters['trackpy:size estimate'][0], invert=parameters['trackpy:invert'][0])
    return df

