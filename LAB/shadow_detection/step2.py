# Compute the mean values of the pixels in L, A and B planes of the image separately.
import cv2


class Step2(object):
    def __init__(self):
        pass

    def run(self, image):
        mean_values = self.compute_mean_values(image)
        return mean_values

    def compute_mean_values(self, image):
        (l_channel, a_channel, b_channel) = self.split_channels(image)
        l_mean = self.channel_mean(l_channel)[0]
        a_mean = self.channel_mean(a_channel)[0]
        b_mean = self.channel_mean(b_channel)[0]
        return l_mean, a_mean, b_mean

    def split_channels(self, image):
        return cv2.split(image)

    def channel_mean(self, channel):
        return cv2.mean(channel)
