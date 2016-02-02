import cv2
import numpy as np
from boudary_drawer import draw_boundaries, draw_region
from color_segmentator import ColorSegmentator

settings = {
    'min_size_factor': 80,
    'dil_erod_kernel_size_segmentator': [8, 8]
}

img=cv2.imread('img/road6.png')

def callback1(pos):
    cs.settings['min_size_factor'] = pos
    i = apply_segmentation(img, cs.segment_image())
    cv2.imshow('img', i)

def callback2(pos):
    cs.settings['dil_erod_kernel_size_segmentator'][0] = pos
    i = apply_segmentation(img, cs.segment_image())
    cv2.imshow('img', i)

def callback3(pos):
    cs.settings['dil_erod_kernel_size_segmentator'][1] = pos
    i = apply_segmentation(img, cs.segment_image())
    cv2.imshow('img', i)

def apply_segmentation(img, segmentation):
    i = img.copy()
    i2 = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for segment in segmentation:
        i = draw_boundaries(i, segment)
        i2 = draw_region(i2, segment)
    return np.concatenate((i, i2), axis=1)

cs = ColorSegmentator(settings)
im = apply_segmentation(img,cs.segment_image(img))
cv2.namedWindow('img')
cv2.createTrackbar('min_size_factor', 'img', settings['min_size_factor'], 100, callback1)
cv2.createTrackbar('dil_erod_kernel_X', 'img', settings['dil_erod_kernel_size_segmentator'][0], 50, callback2)
cv2.createTrackbar('dil_erod_kernel_Y', 'img', settings['dil_erod_kernel_size_segmentator'][1], 50, callback3)
# Do whatever you want with contours
cv2.imshow('img', im)
cv2.waitKey(0)
cv2.destroyAllWindows()
