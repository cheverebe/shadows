import random
import cv2
import numpy as np
from boudary_drawer import draw_boundaries, draw_region
from color_segmentator import ColorSegmentator
from standalones.step_standalone import StepStandalone

from skimage.segmentation import mark_boundaries

class SegmentatorStandalone(StepStandalone):
    default_settings = {
        'dil_erod_kernel_size_segmentator': 7,
        'min_size_factor': 20,
        'segementation_detail': 20,
        'seg_hist_soften': 8
    }
    window_name = 'Color segmentator tester'
    processor_class = ColorSegmentator

    def __init__(self):
        super(SegmentatorStandalone, self).__init__()

    def apply_segmentation(self, segmentation):
        boundaries = self.pre_processed_img.copy()
        regions = np.zeros((boundaries.shape[0], boundaries.shape[1], 3), np.uint8)
        for segment in segmentation:
            boundaries = draw_boundaries(boundaries, segment)
            regions = draw_region(regions, segment)
        for segment in segmentation:
            color = tuple([random.randint(0, 255) for _ in xrange(3)])
            regions = draw_boundaries(regions, segment, color)
        return np.concatenate((boundaries, regions), axis=0)

    def update_img(self):
        segmentation = self.processor.segment_image(self.pre_processed_img)
        self.processed_img = self.apply_segmentation(segmentation)

    def segementation_detail_callback(self, pos):
        self.processor.settings['segementation_detail'] = pos
        self.settings['segementation_detail'] = pos
        self.update_screen()

    def update_screen(self):
        self.update_img()
        self.display_message()
        cv2.imshow(self.window_name, self.processed_img)

    def seg_hist_soften_callback(self, value):
        self.processor.settings['seg_hist_soften'] = value
        self.settings['seg_hist_soften'] = value
        self.update_screen()

    def dil_erod_kernel_size_segmentator_callback_y(self, pos):
        self.processor.settings['dil_erod_kernel_size_segmentator'][0] = pos
        self.settings['dil_erod_kernel_size_segmentator'][0] = pos
        self.update_screen()

    def dil_erod_kernel_size_segmentator_callback_x(self, pos):
        self.processor.settings['dil_erod_kernel_size_segmentator'][1] = pos
        self.settings['dil_erod_kernel_size_segmentator'][1] = pos
        self.update_screen()

    def seg_hist_soften_callback(self, pos):
        self.processor.settings['seg_hist_soften'] = pos
        self.settings['seg_hist_soften'] = pos
        self.update_screen()

    def min_size_factor_callback(self, pos):
        self.processor.settings['min_size_factor'] = pos
        self.settings['min_size_factor'] = pos
        self.update_screen()

    def initialize_windows(self):
        cv2.namedWindow(self.window_name)

        cv2.createTrackbar('segementation_detail',
                           self.window_name,
                           self.settings['segementation_detail'],
                           30,
                           self.segementation_detail_callback)
        cv2.createTrackbar('seg_hist_soften',
                           self.window_name,
                           self.settings['seg_hist_soften'],
                           20,
                           self.seg_hist_soften_callback)
        cv2.createTrackbar('dil_erod_kernel_size_segmentator_x',
                           self.window_name,
                           self.settings['dil_erod_kernel_size_segmentator'][0],
                           20,
                           self.dil_erod_kernel_size_segmentator_callback_x)
        cv2.createTrackbar('dil_erod_kernel_size_segmentator_y',
                           self.window_name,
                           self.settings['dil_erod_kernel_size_segmentator'][1],
                           20,
                           self.dil_erod_kernel_size_segmentator_callback_y)
        cv2.createTrackbar('seg_hist_soften',
                           self.window_name,
                           self.settings['seg_hist_soften'],
                           20,
                           self.seg_hist_soften_callback)
        cv2.createTrackbar('min_size_factor',
                           self.window_name,
                           self.settings['min_size_factor'],
                           50,
                           self.min_size_factor_callback)


SegmentatorStandalone().run()
cv2.destroyAllWindows()
