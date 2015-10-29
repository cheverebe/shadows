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

    def convert_to_bgr(self, image):
        return cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
