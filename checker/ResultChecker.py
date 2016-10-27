import os

import cv2


class ResultCHecker(object):
    def __init__(self):
        self.result_path = ''
        self.ground_truth_path = ''

        self.result_files = []
        self.ground_truth_files = []
        self.sequence_index = 0

    def run(self):
        self.result_path = raw_input("Enter result folder path:")
        self.ground_truth_path = raw_input("Enter ground truth folder path:")

        self.setup_image_sequences()

        if len(self.result_files) != len(self.ground_truth_files):
            print 'Sequences doesn\'t have same legth'
            raise Exception

        for i in range(len(self.result_files)):
            result_path = self.result_files[i]
            print "Reading: " + result_path
            result = cv2.imread(self.result_path+result_path)

            GT_path = self.ground_truth_files[i]
            print "Reading: " + GT_path
            ground_truth = cv2.imread(self.ground_truth_path+GT_path)

            correct = cv2.bitwise_and(result, ground_truth)
            false_positive = result - correct
            false_negative = ground_truth - correct

    def setup_image_sequences(self):
        self.result_files = sorted([f for f in os.listdir(self.result_path)])
        self.ground_truth_files = sorted([f for f in os.listdir(self.ground_truth_path)])
