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

import thread


class MainApp(StepStandalone):
    source = 'img/sequences/1/'
    angle_file = AngleFinder.angle_file_path
    default_settings = {
        'predefined_angle': None,
        'blur_kernel_size': [5, 5],
        'tolerance': 10,
        'dil_erod_kernel_size': (1, 1)
    }
    window_name = 'MainApp'
    processor_class = InvariantImageGenerator

    def __init__(self):
        self.socket = None
        self.last_ts = None
        self.started = False
        self.updated_angle = False
        self.message = None

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

        self.initialize_windows()

    def update_img(self):
        mono = self.processor.project_into_predef_angle(self.pre_processed_img)
        mono = np.uint8(equalize_hist_3d(mono))

        b_eq_inv_mono = cv2.blur(mono, tuple(self.settings['blur_kernel_size']))
        self.path_mask = find_path(b_eq_inv_mono, self.settings)
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
        img = cv2.imread(self.source+img_path)
        self.sequence_index += 1 if self.sequence_index < len(self.sequence_files) else 0
        return img

    def setup_image_sequence(self):
        if isinstance(self.source, str) and self.source != 'camera':
            self.sequence_files = [f for f in os.listdir(self.source)]
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
        while not self.updated_angle:
            print('Waiting for angle update...')
            time.sleep(5)
        while not self.should_stop(k):
            self.original_img = self.get_image()
            self.pre_processed_img = self.pre_process_image()
            self.update_img()
            print "angle:%d" % self.settings['predefined_angle']
            self.export_images()
            k = cv2.waitKey(1)  # TODO: set to 20
            while not self.updated_angle:  # todo: disable later
                print('Waiting for angle update...')
                time.sleep(5)
        print('Finished')

    def cleanup_angle(self):
        try:
            os.remove(self.angle_file)
            print('angle file removed')
        except:
            print('angle file didn\'t existed')

    def update_angle(self, threadName, delay):
        v = 1
        while True:
            try:
                settings_file = open(self.angle_file, 'r+')
                file_content = settings_file.readline()
                settings_file.close()
                angle_str, ts = file_content.split(" ")
                if self.last_ts != ts:
                    self.last_ts = ts
                    self.updated_angle = True
                    self.started = True
                    angle = int(angle_str)
                    print('New angle: '+str(angle))
            except:
                angle = self.settings['predefined_angle']
            self.settings['predefined_angle'] = angle
            self.processor.settings['predefined_angle'] = angle
            print "-------------------"
            time.sleep(5)

    # def open_socket(self):
    #     HOST = ''  # Symbolic name, meaning all available interfaces
    #     PORT = 8888  # Arbitrary non-privileged port
    #
    #     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     print 'Socket created'
    #
    #     # Bind socket to local host and port
    #     try:
    #         s.bind((HOST, PORT))
    #     except socket.error as msg:
    #         print 'Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
    #         sys.exit()
    #
    #     print 'Socket bind complete'
    #     self.socket = s
    #     # Start listening on socket
    #     self.socket.listen(10)
    #     print 'Socket now listening'
    #
    #     return s
    #
    # def listen_socket(self):
    #     conn, addr = self.socket.accept()
    #     print 'Connected with ' + addr[0] + ':' + str(addr[1])

    def initialize_windows(self):
        cv2.namedWindow(self.window_name)

    def export_images(self):
        if self.updated_angle or not self.started:
            print('Exporting images...')
            self.updated_angle = False
            folder = AngleFinder.source
            folder = folder + "/" if not folder.endswith("/") else folder
            image_files = os.listdir(folder)

            print('Deleting previous images...')
            for file in image_files:
                print(folder+file)
                os.remove(folder+file)
            ts = time.time()
            img_name = str(ts) + ".png"

            print('Exporting '+folder+img_name)
            cv2.imwrite(folder+img_name, self.original_img)
            if self.started:
                mask_name = AngleFinder.mask_prefix + img_name
                print('Exporting '+folder+mask_name)
                cv2.imwrite(folder+mask_name, self.path_mask)
                print('Exporting '+'out/mask_'+str(self.sequence_index)+'.png')
                cv2.imwrite('out/'+str(self.sequence_index)+'.png', self.processed_img)
            print('Exported images...')