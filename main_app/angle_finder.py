import cv2

import os

import time

from LAB.shadow_detection.utils import equalize_hist_3d
from boudary_drawer import draw_boundaries
from path_finder import find_path
from standalones.angle_finder_standalone import AngleFinderStandalone
from Greyscale.InvariantImageGenerator import InvariantImageGenerator

import numpy as np


class AngleFinder(AngleFinderStandalone):
    source = 'img/angle_finder_input/'
    angle_file_path = 'found_angle.txt'
    default_settings = {
        'predefined_angle': 80,
        'blur_kernel_size': [5, 5],
        'tolerance': 10,
        'dil_erod_kernel_size': (1, 1)
    }
    window_name = 'AngleFinder'
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
        self.idx = 0

    def initialize_windows(self):
        cv2.namedWindow(self.window_name)

    def get_image(self):
        img_path, mask_path = self.get_new_img_path()
        while img_path is None or self.img_path == img_path:
            time.sleep(1)
            img_path, mask_path = self.get_new_img_path()
        self.img_path = img_path
        self.original_img = cv2.imread(self.source+img_path)
        while self.original_img is None:
            self.original_img = cv2.imread(self.source + img_path)
        self.road_mask_path = self.source+mask_path if mask_path is not None else None

    def get_new_img_path(self):
        image_files = os.listdir(self.source)
        img_paths = [i for i in image_files if not i.startswith(self.mask_prefix)]
        img_path = img_paths[0] if len(img_paths) > 0 else None
        mask_path = self.mask_prefix + img_path if img_path is not None else None
        mask_path = mask_path if mask_path in image_files else None
        return img_path, mask_path

    def select_estimated_road_mask(self):
        clean_after = True if self.road_mask_path is None else False
        mask = super(AngleFinder, self).select_estimated_road_mask(self.road_mask_path)
        if clean_after:
            cv2.destroyAllWindows()
        return mask

    def run(self):
        k = None
        while k != ord('x'):
            cv2.destroyAllWindows()
            self.get_image()
            try:
                self.processor = self.initialize_processor()
            except:
                pass

            self.pre_processed_img = self.pre_process_image()
            self.export_angle()
            self.update_img()
            self.idx += 1
            print "angle:%d" % self.angle
            k = cv2.waitKey(20)

    def export_angle(self):
        print('Exporting angle...')
        f = open(self.angle_file_path, "w")
        f.write("%d %d" % (self.angle, time.time()))
        f.close()
        print('Exported angle...')