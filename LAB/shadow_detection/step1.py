# Convert the RGB image to a LAB image.
import cv2


class Step1(object):
    def __init__(self):
        pass

    def run(self, image):
        lab_image = self.convert_to_lab(image)
        return lab_image

    def convert_to_lab(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # (l, a, b) = cv2.split(lab)
        # m = max(a.max(), b.max())
        # a = a*255/m
        # b = b*255/m
        # return cv2.merge((l, a, b))

    def convert_lab_to_bgr(self, image):
        return cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    def convert_to_hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def convert_hsv_to_bgr(self, image):
        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
