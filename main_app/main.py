import numpy as np
import cv2
import time
import socket

import os
import sys

from LAB.shadow_detection.utils import equalize_hist_3d
from boudary_drawer import draw_boundaries
from main_app.angle_finder import AngleFinder
from path_finder import find_path
from standalones.step_standalone import StepStandalone
from Greyscale.InvariantImageGenerator import InvariantImageGenerator
from collections import deque

import thread


class MainApp(StepStandalone):
    source = 'img/sequences/21/'
    angle_file = AngleFinder.angle_file_path
    default_settings = {
        'predefined_angle': None,
        'blur_kernel_size': [5, 5],
        'tolerance': 10,
        'dil_erod_kernel_size': (1, 1)
    }
    window_name = 'MainApp'
    processor_class = InvariantImageGenerator
    angle_buffer_max_size = 5
    mask_buffer_max_size = 8
    change_threshold = 0.15

    def __init__(self):
        self.socket = None
        self.last_ts = None
        self.started = False
        self.updated_angle = False
        self.message = None
        self.angles = deque()
        self.masks = deque()
        self.mask_sizes = deque()
        self.mono = None

        if self.source == 'camera':
            self.cap = cv2.VideoCapture(0)
            for i in range(5):
                ret, frame = self.cap.read()
                cv2.imshow(self.window_name, frame)
                time.sleep(1)
            self.should_stop = lambda k: k != ord('x')
        else:
            self.setup_image_sequence()
            self.should_stop = lambda k: self.sequence_index == len(self.sequence_files)

        print('Reading from '+self.source)

        self.settings = self.load_settings()
        self.processor = self.initialize_processor()

        # self.initialize_windows()

    def update_img(self):
        self.mono = self.processor.project_into_predef_angle(self.pre_processed_img)
        self.mono = np.uint8(equalize_hist_3d(self.mono))

        b_eq_inv_mono = cv2.blur(self.mono, tuple(self.settings['blur_kernel_size']))

        self.path_mask = self.effective_mask(b_eq_inv_mono)
        edged = draw_boundaries(self.original_img, self.path_mask)

        self.processed_img = edged

    def pre_process_image(self):
        log_chrom = self.processor.log_chrom_image(self.original_img)
        return self.processor.project_to_2d(log_chrom)

    def get_image(self):
        if self.source == 'camera':
            ret, frame = self.cap.read()
            return frame
        else:
            return self.get_next_sequence_image()

    def get_next_sequence_image(self):
        img_path = self.sequence_files[self.sequence_index]
        print "Reading: " + img_path
        img = cv2.imread(self.source+img_path)
        self.sequence_index += 1 if self.sequence_index < len(self.sequence_files) else 0
        return img

    def setup_image_sequence(self):
        if isinstance(self.source, str) and self.source != 'camera':
            self.sequence_files = sorted([f for f in os.listdir(self.source)])
            print self.sequence_files
            self.sequence_index = 0
        else:
            print "Sequence source must be a folder relative path"
            raise Exception

    def run(self):
        try:
            thread.start_new_thread(self.update_angle, ("Thread-1", 2,))
        except:
            print "Error: unable to start thread"

        k = None
        self.cleanup_angle()
        self.original_img = self.get_image()
        self.export_images()
        print('Waiting for angle update...')
        while not self.updated_angle:
            time.sleep(5)
        while not self.should_stop(k):
            cv2.destroyAllWindows()
            self.original_img = self.get_image()
            self.pre_processed_img = self.pre_process_image()
            self.update_img()
            print "angle:%d" % self.settings['predefined_angle']
            self.export_images()
            k = cv2.waitKey(1)  # TODO: set to 20
            # while not self.updated_angle:  # todo: disable later
            #     print('Waiting for angle update...')
            #     time.sleep(5)
        print('Finished')

    def cleanup_angle(self):
        try:
            os.remove(self.angle_file)
            print('angle file removed')
        except:
            print('angle file didn\'t existed')

    def add_angle(self, angle):
        self.angles.append(angle)
        while len(self.angles) > self.angle_buffer_max_size:
            self.angles.popleft()

    def add_mask(self, mask):
        self.masks.append(mask)
        self.mask_sizes.append(cv2.sumElems(mask)[0])
        while len(self.masks) > self.mask_buffer_max_size:
            self.masks.popleft()
            self.mask_sizes.popleft()

    def avg_angle(self):
        return sum(self.angles)/len(self.angles)

    def avg_mask_size(self):
        return sum(self.mask_sizes)/len(self.mask_sizes)

    def avg_mask(self):
        accum = None
        n = len(self.masks)
        for i in range(n):
            m = self.masks[i]
            if i == 0:
                accum = np.uint16(m)
            else:
                accum += m.reshape(accum.shape)
        avg_m = np.uint8(accum / n)
        retval, tmask = cv2.threshold(avg_m, 128, 255, cv2.THRESH_BINARY)
        return tmask

    def effective_mask(self, b_eq_inv_mono):
        mask = find_path(b_eq_inv_mono, self.settings)
        effective_mask = None
        mask_size = cv2.sumElems(mask)[0]

        if len(self.masks) >= self.angle_buffer_max_size:
            # First try to adapt tolerance
            if self.avg_mask_size() * (1+self.change_threshold) < mask_size and self.decrease_tolerance():
                # mask size has increased too much (a lot of false positives)
                print "Decreased tolerance: " + str(self.settings['tolerance'])
                mask = find_path(b_eq_inv_mono, self.settings)
                mask_size = cv2.sumElems(mask)[0]
                self.increase_tolerance()
            elif self.avg_mask_size() * (1-self.change_threshold) > mask_size and self.increase_tolerance():
                # mask size has decreased too much (a lot of false negatives)
                print "Increased tolerance: " + str(self.settings['tolerance'])
                mask = find_path(b_eq_inv_mono, self.settings)
                mask_size = cv2.sumElems(mask)[0]
                self.decrease_tolerance()

            # If adapting tolerance is not enough use mask buffer
            if self.avg_mask_size() * (1+self.change_threshold) < mask_size:
                # mask size has increased too much (a lot of false positives)
                print "mask size has increased too much (a lot of false positives)"
                effective_mask = cv2.bitwise_and(self.avg_mask(), mask)
            elif self.avg_mask_size() * (1-self.change_threshold) > mask_size:
                # mask size has decreased too much (a lot of false negatives)
                print "mask size has decreased too much (a lot of false negatives)"
                effective_mask = cv2.bitwise_or(self.avg_mask(), mask)

        if effective_mask is None:
            effective_mask = mask

        self.add_mask(effective_mask)

        return effective_mask

    def decrease_tolerance(self):
        if self.settings['tolerance'] >=4:
            self.settings['tolerance'] -= 2
            return True
        return False

    def increase_tolerance(self):
        if self.settings['tolerance'] <=4:
            self.settings['tolerance'] += 2
            return True
        return False

    def update_angle(self, threadName, delay):
        v = 1
        while True:
            changed = False
            try:
                settings_file = open(self.angle_file, 'r+')
                file_content = settings_file.readline()
                settings_file.close()
                angle_str, ts = file_content.split(" ")
                if self.last_ts != ts:
                    changed = True
                    self.last_ts = ts
                    self.updated_angle = True
                    self.started = True
                    angle = int(angle_str)
                    print('New angle: '+str(angle))
                    if angle < 0:
                        angle = self.settings['predefined_angle']
                        print('Angle was negative, reusing previous: '+str(angle))
                    else:
                        self.add_angle(angle)
                    print('Avg angle: '+str(self.avg_angle()))
                    angle = self.avg_angle()
            except:
                angle = self.settings['predefined_angle']
            if changed:
                print('Angle buffer: '+str(self.angles))
                self.settings['predefined_angle'] = angle
                self.processor.settings['predefined_angle'] = angle
                print "-------------------"
            time.sleep(5)

    def initialize_windows(self):
        cv2.namedWindow(self.window_name)

    def export_images(self):
        # Export images for angle finder
        print('Exporting images...')
        self.updated_angle = False
        # Read old files in folder
        folder = AngleFinder.source
        folder = folder + "/" if not folder.endswith("/") else folder
        image_files = os.listdir(folder)

        print('Deleting previous images...')
        for file in image_files:
            print(folder+file)
            os.remove(folder+file)

        # Generate unique crescent image name
        img_name = str(self.sequence_index).zfill(7) + ".png"
        img_path = folder+img_name

        print('Exporting '+img_path)
        if self.started:
            mask_name = AngleFinder.mask_prefix + img_name
            mask_img_path = folder+mask_name
            print('Exporting '+mask_img_path)

            angled_img_name = str(self.sequence_index).zfill(7) + "-" + \
                              str(self.settings['predefined_angle']) + ".png"
            angled_img_path = folder+img_name
            cv2.imwrite(folder+mask_name, self.path_mask)
            # cv2.imwrite('img/out/'+mask_name, self.path_mask)

            # Export mono projected image for debuggimg purpouses
            # cv2.imwrite('img/out/mono_' +
            #             angled_img_name, self.mono) # TODO: remove
            cv2.imwrite('img/out/edged_' +
                        angled_img_name, self.processed_img) # TODO: remove

        # Export the original image after the mask because the angle finder
        # looks first for the original image and assumes that if there is a mask for
        # it it's been already generated
        cv2.imwrite(folder+img_name, self.original_img)
        print('Exported images...')