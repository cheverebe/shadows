import numpy as np
import cv2
from Greyscale.InvariantImageGenerator import InvariantImageGenerator
from LAB.shadow_detection.pipeline import ShadowDetectionPipeline
from standalones.step_standalone import StepStandalone
from Greyscale.distance2 import DistanceFinder


class AngleFinderStandalone(StepStandalone):
    processor_class = InvariantImageGenerator

    def __init__(self):
        self.angle = -1
        self.dist_finder = 0
        super(AngleFinderStandalone, self).__init__()

    def pre_process_image(self):
        img = self.original_img

        iig = self.processor
        log_chrom = iig.log_chrom_image(img)
        two_dim = iig.project_to_2d(log_chrom)

        colorspace = self.get_colorspace()

        pip = ShadowDetectionPipeline(self.settings)
        dilated_shadow_mask, shadow_mask = pip.find_dilated_shadow_mask(img)
        self.dist_finder = DistanceFinder(img,
                                          dilated_shadow_mask,
                                          colorspace,
                                          self.settings)

        # FIND MIN ANGLE
        min_mono = []

        angles = xrange(0, 180)
        min_angle = 0
        min_distance = -1

        for angle in angles:
            mono = iig.project_into_one_d(two_dim, angle)
            distance = self.dist_finder.run(np.float64(mono))

            print str("%d, %s" % (angle, repr(distance)))
            if min_distance == -1 or distance < min_distance:
                min_distance = distance
                min_mono = mono
                min_angle = angle

        self.angle = min_angle
        return two_dim

    def initialize_windows(self):
        cv2.namedWindow(self.window_name)

        cv2.createTrackbar('angle',
                           self.window_name,
                           self.angle,
                           180,
                           self.angle_callback)

    def angle_callback(self, angle):
        self.angle = angle
        self.update_screen()

    def update_img(self):
        self.processed_img = self.processor.project_into_one_d(
            self.pre_processed_img,
            self.angle
        )
        self.processed_img = self.dist_finder.mono_distance_image(self.processed_img)

AngleFinderStandalone().run()
cv2.destroyAllWindows()