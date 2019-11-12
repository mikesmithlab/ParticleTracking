from Generic import images
import cv2
import cv2
import numpy as np
from ParticleTracking.general.parameters import  get_param_val


def distance(frame, parameters=None):
    '''
    Performs distance transform:

    distance implements the opencv distance transform. This
    transform operates on a binary image. For each chosen white pixel
    it calculates the distance to the nearest black pixel. This
    distance is the value of the chosen pixel. Thus if operating on
    a white circle the distance transform is a maximum at the middle and
    1 at the perimeter.

    :param frame: binary frame
    :param parameters: parameters dictionary

    :return: The distance transform image.
    '''
    dist = cv2.distanceTransform(frame, cv2.DIST_L2, 5)
    return dist

def grayscale(frame, parameters=None):
    '''
    Convert colour to grayscale:

    This function converts a bgr colour image into a monochrome
    or grayscale image.

    :param frame: A 3 channel colour image
    :param parameters: parameters dictionary

    :return: A grayscale image
    '''
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
    Subtract bkg image

    This function subtracts a background image from a grayscale frame.

    options:
    parameters['subtract bkg type'] == 'mean' : subtracts the mean intensity from image
                                    == 'img' : subtracts a pre-prepared background image.
                                            (See preprocessing > meanbkg_img.py)
    parameters['subtract bkg norm'] == True   : Stretches range of final image intensities 0-255

    :param frame: grayscale image
    :param parameters: parameters dictionary

    :return: image with background image subtracted
    '''


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

    if parameters['subtract bkg norm'] == True:
        frame = cv2.normalize(frame, None, alpha=0, beta=255,
                              norm_type=cv2.NORM_MINMAX)

    return frame


def variance(frame, parameters=None):
    '''
    variance image

    This function finds the mean value of image and then returns
    frame which is the absolute difference of each pixel from that value

    options:
    parameters['variance type'] == 'mean' : Returns the absolute difference from the mean
                                            img value
    parameters['variance type'] == 'img'  : Returns the absolute difference from a supplied
                                            bkg img. Bkg img is stored in parameters['bkg_img'].
    parameters['variance bkg norm'] == True: will stretch the range of the largest difference to 255

    :param frame: grayscale image
    :param parameters: parameters dictionary

    :return: image of absolute difference from mean
    '''

    if parameters['variance type'] == 'mean':
        mean_val = int(np.mean(frame))
        subtract_frame = mean_val*np.ones(np.shape(frame), dtype=np.uint8)
    elif parameters['variance type'] == 'img':
        temp_params = {}
        temp_params['blur kernel'] = get_param_val(parameters['variance blur kernel'].copy())
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

    if parameters['variance bkg norm'] == True:
        frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return frame

def flip(frame, parameters=None):
    '''
    Inverts a binary frame

    :param frame: binary frame
    :param parameters: parameters dictionary

    :return: inverted binary frame
    '''
    return ~frame


def threshold(frame, parameters=None):
    '''
    Apply a global image threshold

    This takes a cutoff threshold value and returns white above and
    black below this value.

    option:
    parameters['threshold'] : sets the value of the cutoff threshold
    parameters['threshold mode] : Can be used to invert the above behaviour

    :param frame: grayscale img
    :param parameters: parameters dictionary

    :return: binary image
    '''
    threshold = get_param_val(parameters['threshold'])
    mode = get_param_val(parameters['threshold mode'])
    return images.threshold(frame, threshold, mode)


def adaptive_threshold(frame, parameters=None):
    '''
    Adaptive threshold

    This applies an adaptive threshold. This differs from global threshold
    in that for each pixel the cutoff threshold is defined based on a block of local
    pixels around it. This enables you to cope with gradual changes in illumination
    across the image etc.

    options:
    parameters['adaptive threshold']['block size'] : Size of local block of pixels to calculate threshold on
    parameters['adaptive threshold']['C'] : The mean-c value see here: http://homepages.inf.ed.ac.uk/rbf/HIPR2/adpthrsh.htm
    parameters['adaptive threshold']['mode'] : inverts behaviour

    :param frame: grayscale image
    :param parameters: parameters dictionary

    :return: binary image
    '''
    params = parameters['adaptive threshold']
    print('pre test')

    block = get_param_val(params['block size'])
    const = get_param_val(params['C'])
    invert = get_param_val(params['mode'])
    if invert == 1:
        return images.adaptive_threshold(frame, block, const, mode=cv2.THRESH_BINARY_INV)
    else:
        return images.adaptive_threshold(frame, block, const)


def blur(frame, parameters=None):
    '''
    Gaussian blur

    Applies a gaussian blur to the image (https://en.wikipedia.org/wiki/Gaussian_blur)
    Usually useful to apply before subtracting 2 images.

    options:
    parameters['blur kernel'] specifies the dimensions of a square kernel

    :param frame: grayscale image
    :param parameters: parameters dictionary

    :return: blurred image
    '''
    kernel = get_param_val(parameters['blur kernel'])
    return images.gaussian_blur(frame, (kernel, kernel))

def medianblur(frame, parameters=None):
    '''
    Median blur

    Applies a median blur to the image (https://en.wikipedia.org/wiki/Median_filter)
    Good for removing speckle noise

    options:
    parameters['blur kernel'] specifies the dimensions of a square kernel

    :param frame: grayscale image
    :param parameters: parameters dictionary

    :return: blurred image
    '''
    kernel = get_param_val(parameters['blur kernel'])
    return images.median_blur(frame, kernel)

def opening(frame, parameters=None):
    '''
    Opening an image

    This is a morphological operation which applies an erosion followed by a dilation
    This tends to remove small isolated objects.
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/open.htm

    options:
    parameters['opening kernel'] specifies the dimensions of a square kernel

    :param frame: binary image
    :param parameters: parameters dictionary

    :return: binary image
    '''
    kernel = get_param_val(parameters['opening kernel'])
    return images.opening(frame, (kernel, kernel))


def closing(frame, parameters=None):
    '''
    Closing an image

    This is a morphological operation which applies a dilation followed
    by an erosion. Tends to fill in holes.
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/close.htm

    options:
    parameters['closing kernel'] specifies the dimensions of a square kernel

    :param frame: binary image
    :param parameters: parameters dictionary

    :return: binary image
    '''
    kernel = get_param_val(parameters['closing kernel'])
    return images.closing(frame, (kernel, kernel))


def dilate(frame, parameters=None):
    '''
    dilation of an image

    This is a morphological operation which applies a kernel to an image to add
    to edge pixels. https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm

    options:
    parameters['dilate kernel'] specifies the dimensions of a square kernel

    :param frame: binary image
    :param parameters: parameters dictionary

    :return: binary image
    '''
    kernel = get_param_val(parameters['dilate kernel'])
    return images.dilate(frame, (kernel, kernel))


def erode(frame, parameters=None):
    '''
    dilation of an image

    This is a morphological operation which applies a kernel to an image to remove
    edge pixels. https://homepages.inf.ed.ac.uk/rbf/HIPR2/erode.htm

    options:
    parameters['erode kernel'] specifies the dimensions of a square kernel

    :param frame: binary image
    :param parameters: parameters dictionary

    :return: binary image
    '''
    kernel = get_param_val(parameters['erode kernel'])
    return images.erode(frame, (kernel, kernel))


def adjust_gamma(image, parameters=None):
    '''
    Gamma correction

    generates a lookup table which maps the values 0-255 to 0-255
    however not in a linear way. The mapping follows a power law
    with exponent gamma.

    :param image: grayscale image
    :param parameters: parameters dictionary

    :return: image
    '''
    gamma = get_param_val(parameters['gamma'])/100.0
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def resize(frame, parameters=None):
    '''
    Resize an image

    resizes an input image by the scale specified

    options:
    parameters['resize scale'] : factor for scale operation

    :param frame: colour, grayscale or binary image
    :param parameters: parameters dictionary

    :return: image
    '''
    scale = get_param_val(parameters['resize scale'])
    return images.resize(frame, scale)

if __name__ == "__main__":
    """Run this to output list of functions"""
    from ParticleTracking.preprocessing import preprocessing_methods as pm
    all_dir = dir(pm)
    all_functions = [a for a in all_dir if a[0] != '_']
    print(all_functions)