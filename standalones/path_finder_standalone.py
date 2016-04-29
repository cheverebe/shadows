import numpy as np
import cv2
from LAB.shadow_detection.utils import equalize_hist_3d
from boudary_drawer import draw_boundaries
from path_finder import find_path
from standalones.step_standalone import StepStandalone
from Greyscale.InvariantImageGenerator import InvariantImageGenerator


class PathFinderStandalone(StepStandalone):
    default_settings = {
        'predefined_angle': 80,
        'blur_kernel_size': [5, 5],
        'tolerance': 10,
        'dil_erod_kernel_size': (1, 1)
    }
    window_name = 'Path finder'
    processor_class = InvariantImageGenerator

    def __init__(self):
        super(PathFinderStandalone, self).__init__()

    def update_img(self):
        mono = self.processor.project_into_predef_angle(self.pre_processed_img)
        mono = np.uint8(equalize_hist_3d(mono))
        mono_as_bgr = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

        b_eq_inv_mono = cv2.blur(mono, tuple(self.settings['blur_kernel_size']))
        path_mask = find_path(b_eq_inv_mono, self.settings)
        edged = draw_boundaries(self.original_img, path_mask)

        self.processed_img = np.concatenate((edged,
                                             mono_as_bgr), axis=1)

    def angle_callback(self, pos):
        self.settings['predefined_angle'] = pos
        self.processor.settings['predefined_angle'] = pos
        self.update_screen()

    def tolerance_callback(self, pos):
        self.settings['tolerance'] = pos
        self.processor.settings['tolerance'] = pos
        self.update_screen()

    def initialize_windows(self):
        cv2.namedWindow(self.window_name)

        cv2.createTrackbar('angle',
                           self.window_name,
                           self.settings['predefined_angle'],
                           180,
                           self.angle_callback)

        cv2.createTrackbar('tolerance',
                           self.window_name,
                           self.settings['tolerance'],
                           255,
                           self.tolerance_callback)

    def pre_process_image(self):
        log_chrom = self.processor.log_chrom_image(self.original_img)
        return self.processor.project_to_2d(log_chrom)

PathFinderStandalone().run()
cv2.destroyAllWindows()
