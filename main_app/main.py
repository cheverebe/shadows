import numpy as np
import cv2
import time

import os

from LAB.shadow_detection.utils import equalize_hist_3d
from boudary_drawer import draw_boundaries
from main_app.angle_finder import AngleFinder
from path_finder import find_path, best_contour
from standalones.step_standalone import StepStandalone
from Greyscale.InvariantImageGenerator import InvariantImageGenerator
from collections import deque

import thread


class MainApp(StepStandalone):
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
    change_threshold = 0.30
    MASK_FOLDER_NAME = 'mask/'
    EDGED_FOLDER_NAME = 'edged/'
    INVARIANT_FOLDER_NAME = 'invariant/'
    TIMES_FILE_NAME = 'times.txt'
    ALLOWED_IMAGE_EXTENSION = '.png'
    DEFAULT_MASK_FOLDER = 'default_mask/'

    def __init__(self, source='img/sequences/3_1/', output_folder='img/out/',
                 export_invariant=False, export_edged=False, angle_file=None, angle_finder_folder=AngleFinder.default_source):
        angle_finder_folder = angle_finder_folder + "/" if not angle_finder_folder.endswith("/") else angle_finder_folder
        self.angle_finder_folder = angle_finder_folder
        if angle_file:
            self.angle_file = AngleFinder.default_angle_file_path
        else:
            self.angle_file = angle_file
        self.socket = None
        self.last_ts = None
        self.started = False
        self.updated_angle = False
        self.message = None
        #   Deques
        self.angles = deque()
        self.masks = deque()
        self.mask_sizes = deque()
        self.mono = None
        self.img_name = ''
        #   Flags
        self.export_invariant = export_invariant
        self.export_edged = export_edged
        #   Folders
        self.source_folder = source
        self.output_folder = output_folder
        self.mask_folder = output_folder + self.MASK_FOLDER_NAME
        self.edged_folder = output_folder + self.EDGED_FOLDER_NAME
        self.invariant_folder = output_folder + self.INVARIANT_FOLDER_NAME
        self.init_folders()
        #   Files
        self.times_file = output_folder + self.TIMES_FILE_NAME
        self.times_file_resource = open(self.times_file, 'w')

        if self.source_folder == 'camera':
            self.cap = cv2.VideoCapture(0)
            for i in range(5):
                ret, frame = self.cap.read()
                cv2.imshow(self.window_name, frame)
                time.sleep(1)
            self.should_stop = lambda k: k != ord('x')
        else:
            self.setup_image_sequence()
            self.should_stop = lambda k: self.sequence_index == len(self.sequence_files)

        print('Reading from ' + self.source_folder)

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
        if self.source_folder == 'camera':
            ret, frame = self.cap.read()
            return frame
        else:
            return self.get_next_sequence_image()

    def get_next_sequence_image(self):
        img_path = self.sequence_files[self.sequence_index]
        self.img_name = self.sequence_files[self.sequence_index]
        print "Reading: " + img_path
        img = cv2.imread(self.source_folder + img_path)
        self.sequence_index += 1 if self.sequence_index < len(self.sequence_files) else 0
        return img

    def setup_image_sequence(self):
        if isinstance(self.source_folder, str) and self.source_folder != 'camera':
            self.sequence_files = sorted([f for f in os.listdir(self.source_folder) if f.endswith(self.ALLOWED_IMAGE_EXTENSION)])
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

        times = []

        while not self.should_stop(k):
            cv2.destroyAllWindows()

            start_time = time.time()

            self.original_img = self.get_image()
            self.pre_processed_img = self.pre_process_image()
            self.update_img()

            print "angle:%d" % self.settings['predefined_angle']
            self.export_images()
            times.append(str(time.time() - start_time)+'\n')
            k = cv2.waitKey(20)
        print('Finished')

        self.times_file_resource.writelines(times)
        self.times_file_resource.close()

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

        if False and len(self.masks) >= self.angle_buffer_max_size: # TODO: WATCH
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

        self.add_mask(mask)

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
        image_files = os.listdir(self.angle_finder_folder)

        print('Deleting previous images...')
        for file in image_files:
            print(self.angle_finder_folder+file)
            os.remove(self.angle_finder_folder+file)

        # Generate unique crescent image name
        # img_name = str(self.sequence_index).zfill(7) + ".png"
        img_name = self.img_name
        img_path = self.angle_finder_folder+img_name

        print('Exporting '+img_path)
        if self.started:
            mask_name = AngleFinder.mask_prefix + img_name
            mask_img_path = self.angle_finder_folder+mask_name
            print('Exporting '+mask_img_path)

            cv2.imwrite(self.angle_finder_folder+mask_name, self.path_mask)
            cv2.imwrite(self.mask_folder+mask_name, self.path_mask)

            # Export mono projected image for debuggimg purpouses
            if self.export_invariant:
                cv2.imwrite(self.invariant_folder+mask_name, self.mono)
            if self.export_edged:
                cv2.imwrite(self.edged_folder+mask_name, self.processed_img)
        elif self.DEFAULT_MASK_FOLDER in os.listdir(self.source_folder):
            default_mask_path = self.source_folder + self.DEFAULT_MASK_FOLDER
            default_mask_name = os.listdir(default_mask_path)[0]
            default_mask_path += default_mask_name
            os.system('cp ' + default_mask_path + ' ' + self.angle_finder_folder)

        # Export the original image after the mask because the angle finder
        # looks first for the original image and assumes that if there is a mask for
        # it it's been already generated
        cv2.imwrite(self.angle_finder_folder+img_name, self.original_img)
        print('Exported images...')

    def export_image_name(self):
        if isinstance(self.source_folder, str) and self.source_folder != 'camera':
            return str(self.sequence_index).zfill(7) + ".png"
        else:
            return self.sequence_files[self.sequence_index]

    def init_folders(self):
        try:
            os.system('rm -rf '+self.output_folder)
            os.system('mkdir '+self.output_folder)
        except Exception as e:
            print(e)
        try:
            os.system('mkdir '+self.mask_folder)
        except Exception as e:
            print(e)
        if self.export_edged:
            try:
                os.system('mkdir '+self.edged_folder)
            except Exception as e:
                print(e)
        if self.export_invariant:
            try:
                os.system('mkdir '+self.invariant_folder)
            except Exception as e:
                print(e)
        if self.angle_finder_folder:
            try:
                os.system('mkdir '+self.angle_finder_folder)
            except Exception as e:
                print(e)
