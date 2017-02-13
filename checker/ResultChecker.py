import os

import cv2


class ResultChecker(object):
    def __init__(self):
        self.result_path = ''
        self.ground_truth_path = ''

        self.result_files = []
        self.ground_truth_files = []
        self.sequence_index = 0
        self.correct_rates = []
        self.false_positive_rates = []
        self.false_negative_rates = []

    def run(self):
        self.result_path = raw_input("Enter result folder path:")
        if not self.result_path.endswith('/'):
            self.result_path += '/'
        self.ground_truth_path = raw_input("Enter ground truth folder path:")
        if not self.ground_truth_path.endswith('/'):
            self.ground_truth_path += '/'

        self.setup_image_sequences()

        if len(self.result_files) != len(self.ground_truth_files):
            print 'Sequences doesn\'t have same legth'
            raise Exception

        for i in range(len(self.result_files)):
            result_path = self.result_path + self.result_files[i]
            print "Reading: " + result_path
            result = cv2.imread(result_path, 0)

            GT_path = self.ground_truth_path + self.ground_truth_files[i]
            print "Reading: " + GT_path
            ground_truth = cv2.imread(GT_path, 0)

            correct = cv2.bitwise_and(result, ground_truth)
            false_positive = result - correct
            false_negative = ground_truth - correct

            real_count = cv2.sumElems(ground_truth)[0] / 255
            correct_count = cv2.sumElems(correct)[0] / 255
            false_positive_count = cv2.sumElems(false_positive)[0] / 255
            false_negative_count = cv2.sumElems(false_negative)[0] / 255

            correct_rate = correct_count / real_count
            false_positive_rate = false_positive_count / real_count
            false_negative_rate = false_negative_count / real_count

            self.correct_rates.append(str(correct_rate))
            self.false_positive_rates.append(str(false_positive_rate))
            self.false_negative_rates.append(str(false_negative_rate))

            base_path = 'checker/out/'
            out_img = cv2.merge((correct, false_positive, false_negative))
            cv2.imwrite(base_path+'out_'+str(i).zfill(5)+'.png', out_img)

        # avg_correct = sum(self.correct_rates) / len(self.correct_rates)
        # avg_positive = sum(self.false_positive_rates) / len(self.false_positive_rates)
        # avg_negative = sum(self.false_negative_rates) / len(self.false_negative_rates)

        print('Correct:\n' + '\n'.join(self.correct_rates))
        print('False positive:\n' + '\n'.join(self.false_positive_rates))
        print('False negative:\n' + '\n'.join(self.false_negative_rates))

    def setup_image_sequences(self):
        self.result_files = sorted([f for f in os.listdir(self.result_path)])
        self.ground_truth_files = sorted([f for f in os.listdir(self.ground_truth_path)])

ResultChecker().run()