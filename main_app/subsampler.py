import os

import cv2


class SubSampler(object):
    ALLOWED_IMAGE_EXTENSION = '.png'
    RESIZE_FACTORS = [0.75, 0.5, 0.25]

    @staticmethod
    def get_folder_path(message="Enter origin folder:"):
        origin_path = raw_input(message)
        if not origin_path.endswith('/'):
            origin_path += '/'
        return origin_path

    @staticmethod
    def resize_image(image, factor):
        return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

    def __init__(self):
        self.origin_path = self.get_folder_path()
        self.output_folder = self.get_folder_path("Enter destination folder:")
        self.image_names = self.image_name_list(self.origin_path)
        print self.image_names
        self.init_folders()

    def run(self):
        for image_name in self.image_names:
            image = cv2.imread(self.origin_path+image_name)
            for factor in self.RESIZE_FACTORS:
                rsz_image = self.resize_image(image, factor)
                rsz_image_name = self.folder_for(factor) + image_name
                print('Exporting: ' + rsz_image_name)
                cv2.imwrite(rsz_image_name, rsz_image)

    def folder_for(self, factor):
        return self.output_folder + str(factor) + '/'

    @classmethod
    def image_name_list(cls, path):
        return sorted([f for f in os.listdir(path) if f.endswith(cls.ALLOWED_IMAGE_EXTENSION)])

    def init_folders(self):
        try:
            os.system('rm -rf '+self.output_folder)
        except Exception as e:
            print(e)
        tokens = self.output_folder.split('/')
        active_tokens = []
        for token in tokens:
            active_tokens.append(token)
            subpath = '/'.join(active_tokens)
            try:
                os.system('mkdir '+subpath)
            except Exception as e:
                print(e)
        for factor in self.RESIZE_FACTORS:
            try:
                os.system('mkdir '+self.folder_for(factor))
            except Exception as e:
                print(e)

SubSampler().run()