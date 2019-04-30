from Generic import images
import numpy as np
import cv2


def find_manual_crop_and_mask(frame):
    """
    Opens a crop shape instance with the input frame and no_of_sides

    Sets the class variables self.mask_img, self.crop and self.boundary
    """
    no_of_sides = int(input('Enter no of sides'))
    crop_inst = images.CropShape(frame, no_of_sides)
    mask_img, crop, boundary, _ = crop_inst.find_crop_and_mask()

    if np.shape(boundary) == (3,):
        # boundary = [xc, yc, r]
        # crop = ([xmin, ymin], [xmax, ymax])
        boundary[0] -= crop[0][0]
        boundary[1] -= crop[0][1]
    else:
        # boundary = ([x1, y1], [x2, y2], ...)
        # crop = ([xmin, ymin], [xmax, ymax])
        boundary[:, 0] -= crop[0][0]
        boundary[:, 1] -= crop[0][1]

    return crop, mask_img, boundary


def find_blue_hex_crop_and_mask(frame):
    blue = images.find_color(frame, 'Blue')
    contours = images.find_contours(blue)
    contours = images.sort_contours(contours)
    # hex_corners = fit_hexagon_to_contour(contours[-2])
    hex_corners = images.fit_hex(np.squeeze(contours[-2]))
    sketch = images.draw_polygon(frame.copy(), hex_corners, thickness=2)
    # images.display(sketch)
    mask_img = np.zeros(np.shape(frame)).astype('uint8')
    cv2.fillPoly(mask_img, pts=np.array([hex_corners], dtype=np.int32),
                 color=(1, 1, 1))
    crop = ([int(min(hex_corners[:, 0])), int(min(hex_corners[:, 1]))],
            [int(max(hex_corners[:, 0])), int(max(hex_corners[:, 1]))])
    boundary = hex_corners
    boundary[:, 0] -= crop[0][0]
    boundary[:, 1] -= crop[0][1]
    return crop, mask_img, boundary

def no_crop(frame):
    shp = np.shape(frame)
    w, h = shp[:2]
    boundary = np.array([[0, 0], [0, w], [w, h], [0, h]])
    return None, None, boundary
