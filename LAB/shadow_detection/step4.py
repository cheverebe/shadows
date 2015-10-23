# Dilate/Erode mask
import cv2
import numpy as np


class Step4(object):
    def __init__(self):
        pass

    def run(self, shadow_mask):
        kernel = np.ones((7, 10), np.uint8)
        dilated_shadow_mask = cv2.dilate(shadow_mask, kernel, iterations=1)
        shadow_mask = cv2.erode(dilated_shadow_mask, kernel, iterations=1)
        return shadow_mask