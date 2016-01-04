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
    final = cv2.bitwise_and(mask_hg_min, mask_lw_high)
    return final

def biggest_contour(image):
    mask = image.copy()
    image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    biggest = None
    biggest_count = -1
    for i in range(len(contours)):
        blank = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
        region_mask = cv2.drawContours(blank, contours, i, 255, -1)
        s = cv2.sumElems(region_mask/255)[0]
        if s > biggest_count:
            biggest = region_mask
            biggest_count = s
    return biggest

def find_path(img):
    path_tone = find_path_tone(img)
    tolerance = settings['tolerance']
    upper_limit = path_tone + tolerance
    lower_limit = path_tone - tolerance
    path_mask = generate_threshhold_mask(img, lower_limit, upper_limit)

    kernel = np.ones(settings['dil_erod_kernel_size'], np.uint8)

    kernel_1 = np.ones((2,2), np.uint8)
    path_mask = cv2.dilate(path_mask, kernel_1, iterations=3)

    eroded_mask = cv2.erode(path_mask, kernel, iterations=3)
    path_mask = cv2.dilate(eroded_mask, kernel, iterations=3)

    return biggest_contour(path_mask)
