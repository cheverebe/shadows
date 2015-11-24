from Greyscale.InvariantImageGenerator import InvariantImageGenerator
from LAB.shadow_detection.utils import load_image, show_and_save
import os

class Runner(object):
    def __init__(self):
        self.image_name = 'img/madera'
        print(os.listdir("img")[5])

        self.invarianGenerator = None
        self.image_ext = 'png'
        self.image = load_image(self.image_name, self.image_ext)

    def run(self):
        self.invarianGenerator = InvariantImageGenerator()
        invariant_image = self.invarianGenerator.get_invariant_image(self.image)
        show_and_save("invariant", self.image_name, self.image_ext, invariant_image)


if __name__ == '__main__':
    r = Runner()
    #r.resize()
    r.run()