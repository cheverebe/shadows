# classify the pixels with lower values in both L and B planes as shadow
# pixels and others as non-shadow pixels.
import cv2


class Step3B(object):
    def __init__(self):
        pass

    def run(self, image):
        l, a, b = cv2.split(image)

        maxval = 255

        # L channel
        l_mean, l_stddev = cv2.meanStdDev(l)
        l_threashold = l_mean - l_stddev / 3
        l_retval, l_mask = cv2.threshold(l, l_threashold, maxval, cv2.THRESH_BINARY)
        # L channel
        b_mean, b_stddev = cv2.meanStdDev(b)
        b_threashold = b_mean - b_stddev / 3
        b_retval, b_mask = cv2.threshold(b, b_threashold, maxval, cv2.THRESH_BINARY)

        print("B")
        return cv2.bitwise_and(l_mask, l_mask, mask=b_mask)
