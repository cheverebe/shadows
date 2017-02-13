import os
import cv2
import numpy as np


class MaskGenerator(object):
    def __init__(self):
        self.original_img = None
        self.window_name = 'Generate mask'
        self.origin_path = ''
        self.masks_path = ''

        self.source_files = []
        self.sequence_index = 0

    def run(self):
        self.origin_path = raw_input('Enter path for input sequence')
        if not self.origin_path.endswith('/'):
            self.origin_path += '/'
        self.masks_path = raw_input('Enter path to store masks')
        if not self.masks_path.endswith('/'):
            self.masks_path += '/'

        self.setup_image_sequences()

        for i in range(len(self.source_files)):
            source_path = self.origin_path + self.source_files[i]
            print "Reading: " + source_path
            original_img = cv2.imread(source_path)
            mask_path = self.masks_path+self.source_files[i]
            mask = self.select_mask_manually(original_img)
            cv2.imwrite(mask_path, mask)
            print "Writing: " + mask_path

    def setup_image_sequences(self):
        self.source_files = sorted([f for f in os.listdir(self.origin_path)])

    def select_mask_manually(self, original_img):
        self.points = []
        cv2.imshow(self.window_name, original_img)
        cv2.setMouseCallback(self.window_name, self.click_and_crop)
        k = -1

        print "Please select points to initialize the road mask and press 'x'"

        while k != ord('x'):
            k = cv2.waitKey(0)

        self.destroy_window()

        w = original_img.shape[0]
        h = original_img.shape[1]
        p = [self.points]
        a3 = np.array(p, dtype=np.int32)
        im = np.zeros([w, h], dtype=np.uint8)
        mask = cv2.fillPoly(im, a3, 255)
        return mask

    def click_and_crop(self, event, x, y, flags, param):

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            print "Selected %d points" % len(self.points)

    def destroy_window(self):
        cv2.destroyWindow(self.window_name)
        cv2.waitKey(1)

MaskGenerator().run()