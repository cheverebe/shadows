import cv2
import numpy as np
from LAB.shadow_detection.pipeline import ShadowDetectionPipeline
from standalones.step_standalone import StepStandalone


class ShadowDetectionStandalone(StepStandalone):
    processor_class = ShadowDetectionPipeline
    default_settings = {
        'dilation_kernel_size_shadow_mask': [8, 8]
    }
    window_name = 'Shadow detection tester'

    def update_img(self):
        mask = self.processor.find_dilated_shadow_mask(self.pre_processed_img)[0]
        inverted_mask = 255 - mask
        mask_as_bgr = cv2.cvtColor(inverted_mask, cv2.COLOR_GRAY2BGR)
        self.processed_img = np.concatenate((self.original_img,
                                             mask_as_bgr), axis=0)

    def kernel_size_x_callback(self, value):
        self.processor.settings['dilation_kernel_size_shadow_mask'][0] = value
        self.settings['dilation_kernel_size_shadow_mask'][0] = value
        self.update_screen()

    def kernel_size_y_callback(self, pos):
        self.processor.settings['dilation_kernel_size_shadow_mask'][1] = pos
        self.settings['dilation_kernel_size_shadow_mask'][1] = pos
        self.update_screen()

    def initialize_windows(self):
        cv2.namedWindow(self.window_name)

        cv2.createTrackbar('dilation_kernel_X',
                           self.window_name,
                           self.settings['dilation_kernel_size_shadow_mask'][0],
                           50,
                           self.kernel_size_x_callback)
        cv2.createTrackbar('dilation_kernel_Y',
                           self.window_name,
                           self.settings['dilation_kernel_size_shadow_mask'][1],
                           50,
                           self.kernel_size_y_callback)


ShadowDetectionStandalone().run()
cv2.destroyAllWindows()
