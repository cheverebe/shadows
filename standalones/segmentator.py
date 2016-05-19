import random
import cv2
import numpy as np
from boudary_drawer import draw_boundaries, draw_region
from color_segmentator import ColorSegmentator
from standalones.step_standalone import StepStandalone

from skimage.segmentation import mark_boundaries

class SegmentatorStandalone(StepStandalone):
    default_settings = {
        'n_segments': 10,
        'compactness': 10,
        'sigma': 3
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
        return np.concatenate((boundaries, regions), axis=1)

    def update_img(self):
        segmentation = self.processor.segment_image(self.pre_processed_img)
        self.processed_img = self.apply_segmentation(segmentation)

    def n_segments_callback(self, pos):
        self.processor.settings['n_segments'] = pos
        self.settings['n_segments'] = pos
        self.update_screen()

    def update_screen(self):
        self.update_img()
        self.display_message()
        cv2.imshow(self.window_name, self.processed_img)

    def compactness_callback(self, value):
        self.processor.settings['compactness'] = value
        self.settings['compactness'] = value
        self.update_screen()

    def sigma_callback(self, pos):
        self.processor.settings['sigma'] = pos
        self.settings['sigma'] = pos
        self.update_screen()

    def initialize_windows(self):
        cv2.namedWindow(self.window_name)

        cv2.createTrackbar('n_segments',
                           self.window_name,
                           self.settings['n_segments'],
                           50,
                           self.n_segments_callback)
        cv2.createTrackbar('compactness',
                           self.window_name,
                           self.settings['compactness'],
                           50,
                           self.compactness_callback)
        cv2.createTrackbar('sigma',
                           self.window_name,
                           self.settings['sigma'],
                           20,
                           self.sigma_callback)


SegmentatorStandalone().run()
cv2.destroyAllWindows()
