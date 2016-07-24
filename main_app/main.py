import numpy as np
import cv2
import time
import socket

import os
import sys

from LAB.shadow_detection.utils import equalize_hist_3d
from boudary_drawer import draw_boundaries
from path_finder import find_path
from standalones.step_standalone import StepStandalone
from Greyscale.InvariantImageGenerator import InvariantImageGenerator

import thread


class MainApp(StepStandalone):
    source = 'camera'
    default_settings = {
        'predefined_angle': 80,
        'blur_kernel_size': [5, 5],
        'tolerance': 10,
        'dil_erod_kernel_size': (1, 1)
    }
    window_name = 'Main App'
    processor_class = InvariantImageGenerator

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.socket = None
        for i in range(5):
            ret, frame = self.cap.read()
            cv2.imshow(self.window_name, frame)
            time.sleep(1)
        super(MainApp, self).__init__()

    def update_img(self):
        mono = self.processor.project_into_predef_angle(self.pre_processed_img)
        mono = np.uint8(equalize_hist_3d(mono))

        b_eq_inv_mono = cv2.blur(mono, tuple(self.settings['blur_kernel_size']))
        path_mask = find_path(b_eq_inv_mono, self.settings)
        edged = draw_boundaries(self.original_img, path_mask)

        self.processed_img = edged

    def initialize_windows(self):
        cv2.namedWindow(self.window_name)

    def pre_process_image(self):
        self.original_img = self.get_image()
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
        self.original_img = cv2.imread(img_path)
        self.sequence_index += 1 if self.sequence_index < len(self.sequence_files) else 0

    def setup_image_sequence(self, source):
        self.source = source
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
        while k != ord('x'):
            self.pre_processed_img = self.pre_process_image()
            self.update_screen()
            print "angle:%d" % self.settings['predefined_angle']
            k = cv2.waitKey(20)

    def update_angle(self, threadName, delay):
        v = 1
        while True:
            try:
                settings_file = open('angle.txt', 'r+')
                file_content = settings_file.readline()
                settings_file.close()

                angle = int(file_content)
            except:
                angle = self.settings['predefined_angle']
            self.settings['predefined_angle'] = angle
            self.processor.settings['predefined_angle'] = angle
            print "-------------------"
            time.sleep(5)

    def open_socket(self):
        HOST = ''  # Symbolic name, meaning all available interfaces
        PORT = 8888  # Arbitrary non-privileged port

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print 'Socket created'

        # Bind socket to local host and port
        try:
            s.bind((HOST, PORT))
        except socket.error as msg:
            print 'Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
            sys.exit()

        print 'Socket bind complete'
        self.socket = s
        # Start listening on socket
        self.socket.listen(10)
        print 'Socket now listening'

        return s

    def listen_socket(self):
        conn, addr = self.socket.accept()
        print 'Connected with ' + addr[0] + ':' + str(addr[1])

ma = MainApp()
ma.run()
ma.cap.release()
ma.socket.close()
cv2.destroyAllWindows()
