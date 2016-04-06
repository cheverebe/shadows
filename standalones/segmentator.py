import cv2
import numpy as np
from boudary_drawer import draw_boundaries, draw_region
from color_segmentator import ColorSegmentator
from standalones.step_standalone import StepStandalone


class SegmentatorStandalone(StepStandalone):
    default_settings = {
        'min_size_factor': 80,
        'dil_erod_kernel_size_segmentator': [8, 8]
    }
    window_name = 'Color segmentator tester'
    processor_class = ColorSegmentator

    def __init__(self, img_path):
        super(SegmentatorStandalone, self).__init__(img_path)

    def apply_segmentation(self, segmentation):
        boundaries = self.pre_processed_img.copy()
        regions = np.zeros((boundaries.shape[0], boundaries.shape[1], 3), np.uint8)
        for segment in segmentation:
            boundaries = draw_boundaries(boundaries, segment)
            regions = draw_region(regions, segment)
        return np.concatenate((boundaries, regions), axis=1)

    def update_img(self):
        segmentation = self.processor.segment_image(self.pre_processed_img)
        self.processed_img = self.apply_segmentation(segmentation)

    def min_size_factor_callback(self, pos):
        self.processor.settings['min_size_factor'] = pos
        self.settings['min_size_factor'] = pos
        self.update_screen()

    def kernel_size_x_callback(self, value):
        self.processor.settings['dil_erod_kernel_size_segmentator'][0] = value
        self.settings['dil_erod_kernel_size_segmentator'][0] = value
        self.update_screen()

    def kernel_size_y_callback(self, pos):
        self.processor.settings['dil_erod_kernel_size_segmentator'][1] = pos
        self.settings['dil_erod_kernel_size_segmentator'][1] = pos
        self.update_screen()

    def initialize_windows(self):
        cv2.namedWindow(self.window_name)

        cv2.createTrackbar('min_size_factor',
                           self.window_name,
                           self.settings['min_size_factor'],
                           100,
                           self.min_size_factor_callback)
        cv2.createTrackbar('dil_erod_kernel_X',
                           self.window_name,
                           self.settings['dil_erod_kernel_size_segmentator'][0],
                           50,
                           self.kernel_size_x_callback)
        cv2.createTrackbar('dil_erod_kernel_Y',
                           self.window_name,
                           self.settings['dil_erod_kernel_size_segmentator'][1],
                           50,
                           self.kernel_size_y_callback)


SegmentatorStandalone('img/r1.png').run()
cv2.destroyAllWindows()
