import cv2
from Greyscale.distance2 import DistanceFinder
from LAB.shadow_detection.pipeline import ShadowDetectionPipeline
from standalones.step_standalone import StepStandalone


class DistanceFindStandalone(StepStandalone):
    default_settings = {
        'distance_colorspace_name': 'LABColorSpace',
        'region_distance_balance': 0.5,
        'max_color_dist': 0.3
    }
    window_name = 'Invariant image tester'
    processor_class = DistanceFinder

    def initialize_processor(self):
        pip = ShadowDetectionPipeline()
        dilated_shadow_mask, shadow_mask = pip.find_dilated_shadow_mask(self.original_img)

        colorspace_name = self.settings['distance_colorspace_name']
        mod = __import__('Greyscale.colorspaces', fromlist=[colorspace_name])
        colorspace_class = getattr(mod, colorspace_name)
        colorspace = colorspace_class()
        return self.processor_class(self.original_img,
                                    dilated_shadow_mask,
                                    colorspace,
                                    self.settings)

    def update_img(self):
        self.processor.initialize_regions()
        self.processed_img = self.processor.region_matches_image()

    def region_distance_balance_callback(self, value):
        value /= 100.0
        self.processor.settings['region_distance_balance'] = value
        self.settings['region_distance_balance'] = value
        self.update_screen()

    def max_color_dist_callback(self, value):
        value /= 100.0
        self.processor.settings['max_color_dist'] = value
        self.settings['max_color_dist'] = value
        self.update_screen()

    def initialize_windows(self):
        cv2.namedWindow(self.window_name)

        cv2.createTrackbar('space<->color',
                           self.window_name,
                           int(self.settings['region_distance_balance']*100),
                           100,
                           self.region_distance_balance_callback)
        cv2.createTrackbar('max_color_dist',
                           self.window_name,
                           int(self.settings['max_color_dist']*100),
                           100,
                           self.max_color_dist_callback)

DistanceFindStandalone().run()
cv2.destroyAllWindows()