from Generic import images
import cv2

def distance(frame, parameters):
    dist = cv2.distanceTransform(frame, cv2.DIST_L2, 5)
    return cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

def grayscale(frame, parameters):
    return images.bgr_2_grayscale(frame)


def crop_and_mask(frame, parameters):
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


def flip(frame, parameters):
    return ~frame


def threshold(frame, parameters):
    threshold = parameters['threshold'][0]
    mode = parameters['threshold mode']
    return images.threshold(frame, threshold, mode)


def adaptive_threshold(frame, parameters):
    block = parameters['adaptive threshold block size'][0]
    const = parameters['adaptive threshold C'][0]
    invert = parameters['adaptive threshold mode'][0]
    if invert == 1:
        return images.adaptive_threshold(frame, block, const, mode=cv2.THRESH_BINARY_INV)
    else:
        return images.adaptive_threshold(frame, block, const)


def blur(frame, parameters):
    kernel = parameters['blur kernel'][0]
    return images.gaussian_blur(frame, (kernel, kernel))


def opening(frame, parameters):
    kernel = parameters['opening kernel'][0]
    return images.opening(frame, (kernel, kernel))


def closing(frame, parameters):
    kernel = parameters['closing kernel'][0]
    return images.closing(frame, (kernel, kernel))


def dilate(frame, parameters):
    kernel = parameters['dilate kernel'][0]
    return images.dilate(frame, (kernel, kernel))


def erode(frame, parameters):
    kernel = parameters['erode kernel'][0]
    return images.erode(frame, (kernel, kernel))


def resize(frame, parameters):
    scale = parameters['resize scale']
    return images.resize(frame, scale)

if __name__ == "__main__":
    """Run this to output list of functions"""
    all_dir = dir(pm)
    all_functions = [a for a in all_dir if a[0] != '_']
    print(all_functions)