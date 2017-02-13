import os

import cv2


class ImageMatcher(object):
    def __init__(self):
        self.full_seq_path = ''
        self.part_seq_path = ''

        self.full_sequence_files = []
        self.partial_sequence_files = []
        self.full_sequence_index = 0
        self.partial_sequence_index = 0
        self.correct_rates = []
        self.false_positive_rates = []
        self.false_negative_rates = []

    def run(self):
        self.full_seq_path = raw_input("Enter full sequence folder path:")
        if not self.full_seq_path.endswith('/'):
            self.full_seq_path += '/'
        self.part_seq_path = raw_input("Enter partial folder path:")
        if not self.part_seq_path.endswith('/'):
            self.part_seq_path += '/'

        self.setup_image_sequences()
        matches = []

        for self.partial_sequence_index in range(len(self.partial_sequence_files)):
            part_img_path = self.part_seq_path + self.partial_sequence_files[self.partial_sequence_index]
            print "Reading: " + part_img_path
            part_img = cv2.imread(part_img_path, 0)
            while self.full_sequence_index < len(self.partial_sequence_files):
                full_img_path = self.full_seq_path + self.full_sequence_files[self.full_sequence_index]
                print "Reading: " + full_img_path
                full_img = cv2.imread(full_img_path, 0)
                self.full_sequence_index += 1
                if part_img.shape == full_img.shape and cv2.sumElems(part_img-full_img)[0] == 0:
                    matches.append((part_img_path, full_img_path))
                    break
        print matches

    def setup_image_sequences(self):
        self.full_sequence_files = sorted([f for f in os.listdir(self.full_seq_path)])
        self.partial_sequence_files = sorted([f for f in os.listdir(self.part_seq_path)])

ImageMatcher().run()