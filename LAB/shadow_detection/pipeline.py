import numpy as np
from step1 import Step1
from step2 import Step2
from step3A import Step3A
from step3B import Step3B
from step4 import Step4
from step5 import Step5
from step6 import Step6
import cv2


class ShadowDetectionPipeline(object):
    def __init__(self):
        self.step1 = Step1()
        self.step2 = Step2()
        self.step3A = Step3A()
        self.step3B = Step3B()
        self.step4 = Step4()
        self.step5 = Step5()
        self.step6 = Step6()

    def run(self, image):
        lab_image = self.step1.run(image)
        mean_values = self.step2.run(lab_image)
        if mean_values[1] + mean_values[2] <= 256:
            shadow_mask = self.step3A.run(lab_image)
        else:
            shadow_mask = self.step3B.run(lab_image)

        shadow_mask = self.step4.run(shadow_mask)
        result = self.step5.run(image, shadow_mask)
        return self.step6.run(result)