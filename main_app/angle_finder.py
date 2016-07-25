import cv2

import os

import time

from standalones.angle_finder_standalone import AngleFinderStandalone
from Greyscale.InvariantImageGenerator import InvariantImageGenerator


class AngleFinder(AngleFinderStandalone):
    source = 'img/angle_finder_input'
    angle_file_path = 'fount_angle.txt'
    default_settings = {
        'predefined_angle': 80,
        'blur_kernel_size': [5, 5],
        'tolerance': 10,
        'dil_erod_kernel_size': (1, 1)
    }
    window_name = 'Main App'
    processor_class = InvariantImageGenerator
    mask_prefix = "mask_"

    def __init__(self):
        self.message = None
        self.angle = -1
        self.dist_finder = 0
        self.settings = self.load_settings()
        self.original_img = None
        self.img_path = None
        self.road_mask_path = None

    def initialize_windows(self):
        cv2.namedWindow(self.window_name)

    def get_image(self):
        img_path = self.get_new_img_path()
        while img_path is None or self.img_path == img_path:
            time.sleep(1)
            img_path = self.get_new_img_path()
        self.img_path = img_path
        self.original_img = cv2.imread(self.source+img_path)
        self.road_mask_path = self.source+self.mask_prefix+img_path

    def get_new_img_path(self):
        image_files = os.listdir(self.source)
        img_paths = [i for i in image_files if not i.startswith(self.mask_prefix)]
        return img_paths[0] if len(img_paths) > 0 else None

    def select_estimated_road_mask(self):
        super(AngleFinder, self).select_estimated_road_mask(self.road_mask_path)

    def run(self):
        k = None
        while k != ord('x'):
            self.get_image()
            self.processor = self.initialize_processor()

            self.pre_processed_img = self.pre_process_image()
            self.export_angle()
            self.update_screen()
            print "angle:%d" % self.angle
            k = cv2.waitKey(20)

    def export_angle(self):
        f = open(self.angle_file_path, "w")
        f.write("%d %d" % self.angle, time.time())
        f.close()


ma = AngleFinder()
ma.run()
ma.cap.release()
ma.socket.close()
cv2.destroyAllWindows()
