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

    def initialize_processor(self):
        pip = ShadowDetectionPipeline()
        dilated_shadow_mask, shadow_mask = pip.find_dilated_shadow_mask(self.original_img)

        colorspace = self.get_colorspace()
        estimated_road_mask = self.select_estimated_road_mask()
        self.dist_finder = DistanceFinder(self.original_img,
                                          dilated_shadow_mask,
                                          colorspace,
                                          estimated_road_mask,
                                          self.settings)
        return super(AngleFinderStandalone, self).initialize_processor()

    def pre_process_image(self):
        img = self.original_img

        inv_img_gen = self.processor
        log_chrom = inv_img_gen.log_chrom_image(img)
        two_dim = inv_img_gen.project_to_2d(log_chrom)
        # FIND MIN ANGLE

        angles = xrange(0, 180)
        min_angle = -1
        min_distance = -1
        if self.dist_finder.has_shadows():
            for angle in angles:
                mono = inv_img_gen.project_into_one_d(two_dim, angle)
                distance = self.dist_finder.run(np.float64(mono))

                print str("%d, %s" % (angle, repr(distance)))
                if min_distance == -1 or distance < min_distance:
                    min_distance = distance
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

# AngleFinderStandalone().run()
# cv2.destroyAllWindows()