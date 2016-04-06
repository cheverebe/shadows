import numpy as np
import cv2
from LAB.shadow_detection.utils import equalize_hist_3d
from standalones.step_standalone import StepStandalone
from Greyscale.InvariantImageGenerator import InvariantImageGenerator


class InvariantImageStandalone(StepStandalone):
    default_settings = {
        'predefined_angle': 80
    }
    window_name = 'Invariant image tester'
    processor_class = InvariantImageGenerator

    def __init__(self, img_path):
        super(InvariantImageStandalone, self).__init__(img_path)

    def update_img(self):
        mono = self.processor.project_into_predef_angle(self.pre_processed_img)
        mono = np.uint8(equalize_hist_3d(mono))
        mono_as_bgr = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        self.processed_img = np.concatenate((self.original_img,
                                             mono_as_bgr), axis=1)

    def angle_callback(self, pos):
        self.settings['predefined_angle'] = pos
        self.processor.settings['predefined_angle'] = pos
        self.update_screen()

    def initialize_windows(self):
        cv2.namedWindow(self.window_name)

        cv2.createTrackbar('angle',
                           self.window_name,
                           self.settings['predefined_angle'],
                           180,
                           self.angle_callback)

    def pre_process_image(self):
        log_chrom = self.processor.log_chrom_image(self.original_img)
        return self.processor.project_to_2d(log_chrom)

InvariantImageStandalone('img/r1.png').run()
cv2.destroyAllWindows()
