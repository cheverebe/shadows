import cv2
from pipeline import ShadowDetectionPipeline
from utils import load_image
from utils import show_and_save


class Runner(object):
    def __init__(self):
        self.pipeline = ShadowDetectionPipeline()
        self.image_name = '../../img/sh1'
        #self.image_name = '../../img/datasets/data_long_term/2009.09.08/data1/A_001_008.40_surfnav'

        self.image_ext = 'png'
        self.image = load_image(self.image_name, self.image_ext)
        self.iterations = 1
        self.use_lab = True

    def run(self):
        image = self.image.copy()
        #show_and_save("", self.image_name, 'png', image)
        for i in range(self.iterations):
            image = self.pipeline.run(image, self.use_lab)
        #image = self.pipeline.run(image, self.use_lab, self.iterations)
        show_and_save("out_"+str(self.use_lab), self.image_name, self.image_ext, image)
        #ch = ''
        #while not ch.lower() == 'x':
        #    ch = raw_input('To exit enter "X"')

    def resize(self):
        out = cv2.resize(self.image, (0,0), fx=0.5, fy=0.5)
        show_and_save("", self.image_name, 'png', out)

if __name__ == '__main__':
    r = Runner()
    #r.resize()
    r.run()