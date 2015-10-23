# Dilate/Erode mask
import cv2


class Step6(object):
    def __init__(self):
        pass

    def run(self, image):
        return image
        maxval = 255
        threashold = 100
        l,a,b = cv2.split(image)
        retval, l_mask = cv2.threshold(l, threashold, maxval, cv2.THRESH_BINARY_INV)
        retval, a_mask = cv2.threshold(a, threashold, maxval, cv2.THRESH_BINARY_INV)
        retval, b_mask = cv2.threshold(b, threashold, maxval, cv2.THRESH_BINARY_INV)
        #killer = 255 - mask
        l_masked = self.apply_mask(l, l_mask)
        a_masked = self.apply_mask(a, a_mask)
        b_masked = self.apply_mask(b, b_mask)
        return cv2.merge([l_masked, a_masked, b_masked]) + (255-cv2.merge([l_mask, a_mask, b_mask]))

    def apply_mask(self, image, mask):
        return cv2.bitwise_and(image, mask)