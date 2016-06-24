import cv2
import numpy as np
from Greyscale.distance2 import DistanceFinder
from LAB.shadow_detection.pipeline import ShadowDetectionPipeline
from standalones.step_standalone import StepStandalone


class InteractiveDistanceFindStandalone(StepStandalone):
    colorspaces_names = ['BGRColorSpace', 'LABColorSpace', 'HSVColorSpace']
    default_settings = {
        'distance_colorspace_name': 'HSVColorSpace',
        'region_distance_balance': 0.5,
        'max_color_dist': 0.3
    }
    window_name = 'Invariant image tester'
    processor_class = DistanceFinder

    WAITING = 0
    FIRST_CLICK = 1
    SECOND_CLICK = 2

    def __init__(self):
        self.status = self.WAITING
        self.clicks = [[], []]
        self.needs_reinitialize = False
        super(InteractiveDistanceFindStandalone, self).__init__()
        cv2.setMouseCallback(self.window_name, self.handle_click)

    def handle_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.status == self.WAITING or self.status == self.SECOND_CLICK:
                self.status = self.FIRST_CLICK
                self.clicks[0] = [x, y]
            elif self.status == self.FIRST_CLICK:
                self.status = self.SECOND_CLICK
                self.clicks[1] = [x, y]
            print self.status
            self.needs_reinitialize = False
            self.update_screen()

    def initialize_processor(self):
        pip = ShadowDetectionPipeline()
        dilated_shadow_mask, shadow_mask = pip.find_dilated_shadow_mask(self.original_img)

        colorspace = self.get_colorspace()
        return self.processor_class(self.original_img,
                                    dilated_shadow_mask,
                                    colorspace,
                                    self.settings)

    def update_img(self):
        print 'WAIT'
        if self.needs_reinitialize:
            self.processor.initialize_regions()

        region_a = None
        region_b = None
        if self.status != self.WAITING:
            region_a = self.processor.region_for(self.clicks[0])
        if self.status == self.SECOND_CLICK:
            region_b = self.processor.region_for(self.clicks[1])
        image = self.processor.distance_image(region_a, region_b)
        self.processed_img = image
        print 'READY'
        self.needs_reinitialize = True


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

        cv2.createTrackbar('variance<->color',
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

InteractiveDistanceFindStandalone().run()
cv2.destroyAllWindows()