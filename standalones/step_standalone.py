import cv2
import json
import numpy as np
import time


class StepStandalone(object):
    default_settings = {}
    window_name = 'Generic window'
    processor_class = None

    def __init__(self):
        img_path = 'img/sequences/8_1/000070.png'
        self.message = None
        self.original_img = cv2.imread(img_path)

        self.settings = self.load_settings()
        self.processor = self.initialize_processor()

        self.pre_processed_img = self.pre_process_image()
        self.processed_img = None

        self.initialize_windows()

        self.update_screen()

    def run(self):
        # Do whatever you want with contours
        k = None
        while (not k) or (k == ord('s')) or (k == ord('e')):
            k = cv2.waitKey(0)
            if k == ord('s'):
                self.save_settings()
                self.message = {'text': 'Saved'}
                self.update_screen()
            if k == ord('e'):
                self.export_current_image()
                print('exported')

    def display_message(self):
        if self.message:
            x = 5
            y = 28
            cv2.putText(self.processed_img,
                        self.message['text'],
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255), 3)
            self.message = None

    def update_img(self):
        raise NotImplementedError

    def update_screen(self):
        self.update_img()
        self.display_message()
        cv2.imshow(self.window_name, self.processed_img)

    def load_settings(self):
        settings_file = open('settings.txt', 'r+')
        file_content = settings_file.readline()
        settings_file.close()

        settings = self.default_settings
        if len(file_content) > 0:
            settings.update(json.loads(file_content))

        return settings

    def save_settings(self):
        settings_file = open('settings.txt', 'w')
        settings_file.write(json.dumps(self.settings))
        settings_file.close()

    def initialize_windows(self):
        raise NotImplementedError

    def pre_process_image(self):
        return self.original_img

    def initialize_processor(self):
        return self.processor_class(self.settings)

    def get_colorspace(self):
        colorspace_name = self.settings['distance_colorspace_name']
        mod = __import__('Greyscale.colorspaces', fromlist=[colorspace_name])
        colorspace_class = getattr(mod, colorspace_name)
        return colorspace_class()

    def select_estimated_road_mask(self, road_mask_path="road_mask.png"):
        mask = None
        try:
            if road_mask_path is not None:
                mask = cv2.imread(road_mask_path, 0)
        except:
            pass
        if mask is None:
            self.points = []
            cv2.imshow(self.window_name, self.original_img)
            cv2.setMouseCallback(self.window_name, self.click_and_crop)
            k = -1

            print "Please select points to initialize the road mask and press 'x'"

            while k != ord('x'):
                k = cv2.waitKey(0)

            self.destroy_window()

            w = self.original_img.shape[0]
            h = self.original_img.shape[1]
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

    def export_current_image(self):
        name = "img/export/" + str(time.time()) + '.png'
        cv2.imwrite(name, self.processed_img)
