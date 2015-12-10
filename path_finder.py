import cv2
import numpy as np
from settings import settings

def get_roi_corners(shape):
    center = shape[1] / 2
    roi_radius = 20
    return (shape[0]-roi_radius+2, shape[0], center-roi_radius, center+roi_radius)

def get_roi(img):
    s = img.shape
    corners = get_roi_corners(s)
    roi = img[corners[0]:corners[1], corners[2]:corners[3]]
    return roi

def find_path_tone(img):
    roi = get_roi(img)
    mean = cv2.mean(roi)[0]
    print "mean:" + str(mean)
    return int(mean)

def generate_threshhold_mask(image, minval, maxval):
    image = np.uint8(image)
    retval, mask_lw_high = cv2.threshold(image, maxval, 255, cv2.THRESH_BINARY)
    mask_lw_high = 255 - mask_lw_high
    retval, mask_hg_min = cv2.threshold(image, minval, 255, cv2.THRESH_BINARY_INV)
    mask_hg_min = 255 - mask_hg_min
    # erase mask values

    return cv2.bitwise_and(mask_hg_min, mask_lw_high)

def find_path(img):
    path_tone = find_path_tone(img)
    tolerance = settings['tolerance']
    upper_limit = path_tone + tolerance
    lower_limit = path_tone - tolerance
    path_mask = generate_threshhold_mask(img, lower_limit, upper_limit)

    kernel = np.ones(settings['dil_erod_kernel_size'], np.uint8)

    eroded_mask = cv2.erode(path_mask, kernel, iterations=3)
    path_mask = cv2.dilate(eroded_mask, kernel, iterations=3)

    #dilated_shadow_mask = cv2.dilate(path_mask, kernel, iterations=1)
    #path_mask = cv2.erode(dilated_shadow_mask, kernel, iterations=1)
    return path_mask
