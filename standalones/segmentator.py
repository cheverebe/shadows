import cv2
import numpy as np
from boudary_drawer import draw_boundaries, draw_region
from color_segmentator import ColorSegmentator
import json

class SegmentatorStandalone(object):
    default_settings = {
        'min_size_factor': 80,
        'dil_erod_kernel_size_segmentator': [8, 8]
    }
    window_name = 'Color segmentator tester'

    def __init__(self, img_path):
        self.message = None

        self.settings = self.load_settings()
        self.cs = ColorSegmentator(self.settings)

        self.original_img = cv2.imread(img_path)
        self.segmentation = self.cs.segment_image(self.original_img)
        self.segmented_img = self.apply_segmentation()

        self.initialize_windows()

    def run(self):
        # Do whatever you want with contours
        k = None
        while (not k) or (k == ord('s')):
            k = cv2.waitKey(0)
            if k == ord('s'):
                self.save_settings()
                self.message = {'text': 'Saved'}
                self.update_img()

    def apply_segmentation(self):
        boundaries = self.original_img.copy()
        regions = np.zeros((boundaries.shape[0], boundaries.shape[1], 3), np.uint8)
        for segment in self.segmentation:
            boundaries = draw_boundaries(boundaries, segment)
            regions = draw_region(regions, segment)
        return np.concatenate((boundaries, regions), axis=1)

    def display_message(self):
        if self.message:
            x = 5
            y = 28
            cv2.putText(self.segmented_img,
                        self.message['text'],
                        (x,y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255), 3)
            self.message = None

    def update_img(self):
        self.segmentation = self.cs.segment_image(self.original_img)
        self.segmented_img = self.apply_segmentation()
        self.display_message()
        cv2.imshow(self.window_name, self.segmented_img)

    def save_settings(self):
        settings_file = open('settings.txt', 'w')
        settings_file.write(json.dumps(self.cs.settings))
        settings_file.close()

    def min_size_factor_callback(self, pos):
        self.cs.settings['min_size_factor'] = pos
        self.update_img()

    def kernel_size_x_callback(self, value):
        self.cs.settings['dil_erod_kernel_size_segmentator'][0] = value
        self.update_img()

    def kernel_size_y_callback(self, pos):
        self.cs.settings['dil_erod_kernel_size_segmentator'][1] = pos
        self.update_img()

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

        self.update_img()

    def load_settings(self):
        settings_file = open('settings.txt', 'r+')
        settings = settings_file.readline()
        settings_file.close()

        if len(settings) > 0:
            return json.loads(settings)
        else:
            return self.default_settings


SegmentatorStandalone('img/r1.png').run()
cv2.destroyAllWindows()
