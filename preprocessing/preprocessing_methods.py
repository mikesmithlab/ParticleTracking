from Generic import images
import cv2
import cv2
import numpy as np
from ParticleTracking.general.parameters import  get_param_val

'''Should ideally be in tracking_methods'''
#def distance(frame, parameters):
#    dist = cv2.distanceTransform(frame, cv2.DIST_L2, 5)
#    return dist

def grayscale(frame, parameters=None):
    return images.bgr_2_grayscale(frame)


def crop_and_mask(frame, parameters=None):
    """
    Masks then crops a given frame

    Takes a frame and uses a bitwise_and operation with the input mask_img
    to mask the image around a shape.
    It then crops the image around the mask.

    Parameters
    ----------
    frame: numpy array
        A numpy array of an image of type uint8

    Returns
    -------
    cropped_frame: numpy array
        A numpy array containing an image which has been cropped and masked
    """
    mask_im = parameters['mask image']
    crop = parameters['crop']
    masked_frame = images.mask_img(frame, mask_im)
    cropped_frame = images.crop_img(masked_frame, crop)
    return cropped_frame

def subtract_bkg(frame, parameters=None):
    '''
        Send grayscale frame. Finds mean value of background and then returns
        frame which is the absolute difference of each pixel from that value
        normalise=True will set the largest difference to 255

        :param frame:
        :return:
        '''
    norm = parameters['subtract bkg norm']

    if parameters['subtract bkg type'] == 'mean':
        mean_val = int(np.mean(frame))
        subtract_frame = mean_val * np.ones(np.shape(frame), dtype=np.uint8)
    elif parameters['subtract bkg type'] == 'img':
        temp_params = {}
        temp_params['blur kernel'] = parameters['subtract blur kernel'].copy()
        # This option subtracts the previously created image which is added to dictionary.
        subtract_frame = parameters['bkg_img']
        frame = blur(frame, temp_params)
        subtract_frame = blur(subtract_frame, temp_params)

    frame = cv2.subtract(subtract_frame, frame)

    if norm == True:
        frame = cv2.normalize(frame, None, alpha=0, beta=255,
                              norm_type=cv2.NORM_MINMAX)

    return frame


def variance(frame, parameters=None):
    '''
    Send grayscale frame. Finds mean value of background and then returns
    frame which is the absolute difference of each pixel from that value
    normalise=True will set the largest difference to 255

    :param frame:
    :return:
    '''
    norm=parameters['variance bkg norm']

    if parameters['variance type'] == 'mean':
        mean_val = int(np.mean(frame))
        subtract_frame = mean_val*np.ones(np.shape(frame), dtype=np.uint8)
    elif parameters['variance type'] == 'img':
        temp_params = {}
        temp_params['blur kernel'] = get_param_val(parameters['variance blur kernel'].copy())
        #This option subtracts the previously created image which is added to dictionary.
        subtract_frame = parameters['bkg_img']
        frame = blur(frame, temp_params)
        subtract_frame = blur(subtract_frame, temp_params)
    elif parameters['variance type'] == 'zeros':
        subtract_frame = np.zeros(np.shape(frame))
    frame1 = cv2.subtract(subtract_frame, frame)
    frame1 = cv2.normalize(frame1, frame1 ,0,255,cv2.NORM_MINMAX)
    frame2 = cv2.subtract(frame, subtract_frame)
    frame2 = cv2.normalize(frame2, frame2,0,255,cv2.NORM_MINMAX)
    frame = cv2.add(frame1, frame2)
    if norm == True:
        frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return frame

def flip(frame, parameters=None):
    return ~frame


def threshold(frame, parameters=None):
    threshold = get_param_val(parameters['threshold'])
    mode = get_param_val(parameters['threshold mode'])
    return images.threshold(frame, threshold, mode)


def adaptive_threshold(frame, parameters=None):
    block = get_param_val(parameters['adaptive threshold block size'])
    const = get_param_val(parameters['adaptive threshold C'])
    invert = get_param_val(parameters['adaptive threshold mode'])
    if invert == 1:
        return images.adaptive_threshold(frame, block, const, mode=cv2.THRESH_BINARY_INV)
    else:
        return images.adaptive_threshold(frame, block, const)


def blur(frame, parameters=None):
    kernel = get_param_val(parameters['blur kernel'])
    return images.gaussian_blur(frame, (kernel, kernel))

def medianblur(frame, parameters=None):
    kernel = get_param_val(parameters['blur kernel'])
    return images.median_blur(frame, kernel)

def opening(frame, parameters=None):
    kernel = get_param_val(parameters['opening kernel'])
    return images.opening(frame, (kernel, kernel))


def closing(frame, parameters=None):
    kernel = get_param_val(parameters['closing kernel'])
    return images.closing(frame, (kernel, kernel))


def dilate(frame, parameters=None):
    kernel = get_param_val(parameters['dilate kernel'])
    return images.dilate(frame, (kernel, kernel))


def erode(frame, parameters=None):
    kernel = get_param_val(parameters['erode kernel'])
    return images.erode(frame, (kernel, kernel))


def adjust_gamma(image, parameters=None):
    gamma = get_param_val(parameters['gamma'])/100.0
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def resize(frame, parameters=None):
    scale = get_param_val(parameters['resize scale'])
    return images.resize(frame, scale)

if __name__ == "__main__":
    """Run this to output list of functions"""
    from ParticleTracking.preprocessing import preprocessing_methods as pm
    all_dir = dir(pm)
    all_functions = [a for a in all_dir if a[0] != '_']
    print(all_functions)