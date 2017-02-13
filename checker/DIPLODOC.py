import os

import cv2
import numpy as np


class DIPLODOC(object):
    def __init__(self):
        self.result_path = ''
        self.ground_truth_path = ''

        self.source_files = []
        self.sequence_index = 0

    def run(self):
        self.result_path = 'img/sequences/gtseq/'
        if not self.result_path.endswith('/'):
            self.result_path += '/'

        self.setup_image_sequences()

        for i in range(len(self.source_files)):
            source_path = self.result_path + self.source_files[i]
            if source_path.endswith('-L.png'):
                print "Reading: " + source_path
                source = cv2.imread(source_path, 0)
                gt = self.build_GT(source_path[:-3]+'txt', source)
                cv2.imwrite('img/sequences/DIPLODOC/seq/'+self.source_files[i], source)
                cv2.imwrite('img/sequences/DIPLODOC/GT/'+self.source_files[i], gt)
                print "Writing: " + source_path

    def setup_image_sequences(self):
        self.source_files = sorted([f for f in os.listdir(self.result_path)])

    def build_GT(self, file_path, img):
        w = img.shape[0]
        h = img.shape[1]

        with open(file_path) as temp_file:
            lines = [line.rstrip('\n') for line in temp_file]
            l = lines[0]
            points = l.split(' ')
            points = points[2:]
            points = [[int(float(points[2*i])*w),int(float(points[2*i+1])*h)] for i in range(len(points)/2)]

            p = [points]
            a3 = np.array(p, dtype=np.int32)
            im = np.zeros([w, h], dtype=np.uint8)
            return cv2.fillPoly(im, a3, 255)

DIPLODOC().run()