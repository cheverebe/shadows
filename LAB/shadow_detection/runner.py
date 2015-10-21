from pipeline import ShadowDetectionPipeline
from utils import load_image
from utils import show_and_save


class Runner(object):
    def __init__(self):
        self.pipeline = ShadowDetectionPipeline()
        self.image_name = '../../img/pelota'
        self.image_ext = 'png'
        self.image = load_image(self.image_name, self.image_ext)

    def run(self):
        result = self.pipeline.run(self.image)
        show_and_save("thresholded", self.image_name, self.image_ext, result)
        #ch = ''
        #while not ch.lower() == 'x':
        #    ch = raw_input('To exit enter "X"')

if __name__ == '__main__':
    r = Runner()
    r.run()