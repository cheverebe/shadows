import cv2
import json

class StepStandalone(object):
    default_settings = {}
    window_name = 'Generic window'
    processor_class = None

    def __init__(self):
        img_path = 'img/kitti/um_000047.png'
        self.message = None
        self.original_img = cv2.imread(img_path)
        # self.original_img = clahe_2(self.original_img)

        self.settings = self.load_settings()
        self.processor = self.initialize_processor()

        self.pre_processed_img = self.pre_process_image()
        self.processed_img = None

        self.initialize_windows()

        self.update_screen()

    def run(self):
        # Do whatever you want with contours
        k = None
        while (not k) or (k == ord('s')):
            k = cv2.waitKey(0)
            if k == ord('s'):
                self.save_settings()
                self.message = {'text': 'Saved'}
                self.update_screen()

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
