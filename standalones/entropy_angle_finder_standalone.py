import random

import cv2
import numpy as np
import math

from Greyscale.distance2 import Region
from standalones.angle_finder_standalone import AngleFinderStandalone
from LAB.shadow_detection.utils import equalize_hist_3d


class EntropyAngleFinderStandalone(AngleFinderStandalone):

    def initialize_processor(self):
        self.estimated_road_mask = self.select_estimated_road_mask()
        return super(AngleFinderStandalone, self).initialize_processor()

    def get_value(self, img):
        # img2 = Region.apply_mask(img, np.float64(self.estimated_road_mask))
        histogram = np.histogram(img, bins=64)[0]
        histogram_length = sum(histogram)
        samples_probability = [float(h) / histogram_length for h in histogram]
        entropy = -sum([p * math.log(p, 2) for p in samples_probability if p != 0])

        return entropy

    def update_img(self):
        self.processed_img = self.processor.project_into_one_d(
            self.pre_processed_img,
            self.angle
        )
        self.processed_img = equalize_hist_3d(self.processed_img)
        # color = [random.randint(0, 255)]
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # entropy = self.get_value(self.processed_img)
        # center = (50, 50)
        # cv2.putText(self.processed_img, str(entropy)[:5],
        #             center, font, 0.5, color)

    def should_process(self):
        return True

    def select_estimated_road_mask(self):
        pass