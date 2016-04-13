import cv2
import numpy as np
from Greyscale.distance2 import DistanceFinder
from LAB.shadow_detection.pipeline import ShadowDetectionPipeline
from standalones.step_standalone import StepStandalone


class DistanceFindStandalone(StepStandalone):
    colorspaces_names = ['BGRColorSpace', 'LABColorSpace', 'HSVColorSpace']
    default_settings = {
        'distance_colorspace_name': 'HSVColorSpace',
        'region_distance_balance': 0.5,
        'max_color_dist': 0.3
    }
    window_name = 'Invariant image tester'
    processor_class = DistanceFinder

    def initialize_processor(self):
        pip = ShadowDetectionPipeline()
        dilated_shadow_mask, shadow_mask = pip.find_dilated_shadow_mask(self.original_img)

        colorspace = self.get_colorspace()
        return self.processor_class(self.original_img,
                                    dilated_shadow_mask,
                                    colorspace,
                                    self.settings)

    def update_img(self):
        self.processor.initialize_regions()
        matches = self.processor.region_matches_image()
        regions = self.processor.segmentation_image()
        self.processed_img = np.concatenate((matches,
                                             regions), axis=1)

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

    def colorspace_callback(self, value):
        colorspace_name = self.colorspaces_names[value]
        mod = __import__('Greyscale.colorspaces', fromlist=[colorspace_name])
        colorspace_class = getattr(mod, colorspace_name)
        colorspace = colorspace_class()

        self.processor.colorspace = colorspace
        self.settings['distance_colorspace_name'] = colorspace_name
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
        cv2.createTrackbar('-'.join([n[:3] for n in self.colorspaces_names]),
                           self.window_name,
                           self.colorspaces_names.index(
                               self.settings['distance_colorspace_name']
                           ),
                           len(self.colorspaces_names)-1,
                           self.colorspace_callback)

DistanceFindStandalone().run()
cv2.destroyAllWindows()