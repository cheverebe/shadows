import cv2
from pipeline import ShadowDetectionPipeline
from utils import load_image
from utils import show_and_save


class Runner(object):
    def __init__(self):
        self.image_name = '../../img/madera'
        #self.image_name = '../../img/datasets/data_long_term/2009.09.08/data1/A_001_008.40_surfnav'

        self.pipeline = None
        self.image_ext = 'png'
        self.image = load_image(self.image_name, self.image_ext)
        self.iterations = 1
        self.methods = [[1], [0], [2]]

    def run(self):
        image = self.image.copy()
        for method in self.methods:
            self.pipeline = ShadowDetectionPipeline()
            for i in range(self.iterations):
                image = self.pipeline.run(image, method)
            show_and_save("out_"+str(method), self.image_name, self.image_ext, image)

    def resize(self):
        out = cv2.resize(self.image, (0,0), fx=0.5, fy=0.5)
        show_and_save("", self.image_name, 'png', out)

if __name__ == '__main__':
    r = Runner()
    #r.resize()
    r.run()