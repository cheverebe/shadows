import cv2

import os
from standalones.angle_finder_standalone import AngleFinderStandalone
from Greyscale.InvariantImageGenerator import InvariantImageGenerator


class AngleFinder(AngleFinderStandalone):
    source = 'img/sequences/1/'
    default_settings = {
        'predefined_angle': 80,
        'blur_kernel_size': [5, 5],
        'tolerance': 10,
        'dil_erod_kernel_size': (1, 1)
    }
    window_name = 'Main App'
    processor_class = InvariantImageGenerator

    def __init__(self):
        self.message = None
        self.angle = -1
        self.dist_finder = 0
        self.settings = self.load_settings()

    def initialize_windows(self):
        cv2.namedWindow(self.window_name)

    def get_image(self):
        if self.source == 'camera':
            ret, frame = self.cap.read()
            return frame
        else:
            return self.get_next_sequence_image()

    def get_next_sequence_image(self):
        img_path = self.sequence_files[self.sequence_index]
        self.original_img = cv2.imread(self.source+img_path)
        self.sequence_index += 1 if self.sequence_index < len(self.sequence_files) else 0

    def setup_image_sequence(self):
        if isinstance(self.source, str) and self.source != 'camera':
            self.sequence_files = os.listdir(self.source)
            self.sequence_files.sort()
            self.sequence_index = 0
        else:
            print "Sequence source must be a folder relative path"
            raise Exception

    def run(self):
        k = None
        while k != ord('x'):
            self.get_image()
            self.processor = self.initialize_processor()

            self.pre_processed_img = self.pre_process_image()
            self.update_screen()
            print "angle:%d" % self.angle
            k = cv2.waitKey(20)
            self.wait_for_new_mask()

    def wait_for_new_mask(self):
        pass


ma = AngleFinder()
ma.setup_image_sequence()
ma.run()
ma.cap.release()
ma.socket.close()
cv2.destroyAllWindows()
