from step1 import Step1
from step2 import Step2
from step3A import Step3A
from step3B import Step3B
import cv2


class ShadowDetectionPipeline(object):
    def __init__(self):
        self.step1 = Step1()
        self.step2 = Step2()
        self.step3A = Step3A()
        self.step3B = Step3B()

    def run(self, image):
        lab_image = self.step1.run(image)
        mean_values = self.step2.run(lab_image)
        print(mean_values[1] + mean_values[2])
        if mean_values[1] + mean_values[2] <= 256:
            light_mask = self.step3A.run(lab_image)
        else:
            light_mask = self.step3B.run(lab_image)
        shadow_mask = 255 - light_mask
        lights = self.apply_mask(image, light_mask)
        shadows = self.apply_mask(image, shadow_mask)
        #lights = lights * 2
        shadows = shadows * 1.8
        return lights + shadows

    def apply_mask(self, image, mask):
        return cv2.bitwise_and(image, cv2.merge([mask, mask, mask]))
