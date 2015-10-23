import cv2


class Step3A(object):
    def __init__(self):
        pass

    def run(self, image):
        l, a, b = cv2.split(image)
        mean, stddev = cv2.meanStdDev(l)
        threashold = mean - stddev / 3
        maxval = 255
        retval, mask = cv2.threshold(l, threashold, maxval, cv2.THRESH_BINARY_INV)
        print("A")
        return mask
