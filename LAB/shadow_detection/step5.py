# Dilate/Erode mask
import cv2
import numpy as np
from LAB.shadow_detection.step1 import Step1
from LAB.shadow_detection.utils import show_and_save
import math


class Step5(object):
    def __init__(self):
        pass

    def run(self, image, dilated_shadow_mask, shadow_mask, use_lab=False):
        #shadows = np.zeros((shadow_mask.shape[0], shadow_mask.shape[1], 3), np.uint8)

        b_region_masks, s_region_masks = self.get_region_masks(dilated_shadow_mask)

        #recompute the shadow mask using all the region masks to avoid troubles with edges
        dilated_shadow_mask = np.zeros((dilated_shadow_mask.shape[0], dilated_shadow_mask.shape[1], 1), np.uint8)

        for region_mask in s_region_masks:
            region = self.apply_mask(image, region_mask)
            #shadows += region
            dilated_shadow_mask += region_mask

        for region_mask in b_region_masks:
            dilated_shadow_mask += region_mask

        light_mask = 255 - dilated_shadow_mask
        lights = self.apply_mask(image, light_mask)

        means = self.light_regions_means(lights, use_lab=use_lab)

        result = image.copy()

        for region_mask in b_region_masks:
            region = self.apply_mask(image, region_mask)
            coef = self.get_best_coeficients(means, region_mask, region, use_lab=use_lab)
            #coef = self.get_coeficients(light_mask, lights, region_mask, region)
            #print coef
            region_mask = self.sanitize_mask(region_mask, shadow_mask)
            region = self.apply_coefficients(coef, region, use_lab=use_lab)
            region = self.apply_mask(region, region_mask)
            #shadows += region
            no_region = 255 - region_mask
            result = self.apply_mask(result, no_region)
            result += region

        return result

    def apply_mask(self, image, mask):
        return cv2.bitwise_and(image, cv2.merge([mask, mask, mask]))

    def sanitize_mask(self, dilated_mask, original_mask):
        return cv2.bitwise_and(dilated_mask, original_mask)

    #def get_coeficient(self, light_mask, lights, shadow_mask, shadows):
    #    l_avg = sum([sm / cv2.sumElems(light_mask/255)[0] for sm in cv2.sumElems(lights)])/3
    #    s_avg = sum([sm / cv2.sumElems(shadow_mask/255)[0] for sm in cv2.sumElems(shadows)])/3
    #    return l_avg / s_avg

    def get_coeficients(self, light_mask, lights, shadow_mask, shadows):
        l_avg = [sm / cv2.sumElems(light_mask/255)[0] for sm in cv2.sumElems(lights)]
        s_avg = [sm / cv2.sumElems(shadow_mask/255)[0] for sm in cv2.sumElems(shadows)]
        return [l_avg[i] / s_avg[i] for i in range(3)]

    #def apply_coefficient(self, coef, image):
    #    return image * coef

    def apply_coefficients(self, coef, image, use_lab=False):
        lab_image = Step1().convert_to_lab(image) if use_lab else image
        l, a, b = cv2.split(lab_image)
        l = np.uint16(l)
        a = np.uint16(a)
        b = np.uint16(b)
        l *= coef[0]
        if not use_lab:
            a *= coef[1]
            b *= coef[2]

        result = self.saturate(cv2.merge([l, a, b]))
        if use_lab:
            result = Step1().convert_to_bgr(result)
        return result

    def get_region_masks(self, shadow_mask):
        mask = shadow_mask.copy()
        image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        big_regions = []
        small_regions = []
        #print len(contours)
        for i in range(len(contours)):
            blank = np.zeros((shadow_mask.shape[0], shadow_mask.shape[1], 1), np.uint8)
            region_mask = cv2.drawContours(blank, contours, i, 255, -1)
            s = cv2.sumElems(region_mask/255)[0]
            if s > 1000:
                big_regions.append(region_mask)
            else:
                small_regions.append(region_mask)
        #print("big:%d  small:%d" % (len(big_regions), len(small_regions)))
        return big_regions, small_regions

    def saturate(self, image):
        return cv2.convertScaleAbs(image)

    def apply_multi_mask(self, image, mask):
        return cv2.bitwise_and(image, mask)

    def light_regions_means(self, lights, use_lab=False):
        msft = cv2.medianBlur(lights, 5)
        gray_msft = cv2.cvtColor(msft, cv2.COLOR_BGR2GRAY)
        big_regions, small_regions = self.get_region_masks(gray_msft)
        means = []
        for region_mask in big_regions:
            lab_lights = Step1().convert_to_lab(lights) if use_lab else lights
            region = self.apply_mask(lab_lights, region_mask)
            means.append([sm / cv2.sumElems(region_mask/255)[0] for sm in cv2.sumElems(region)])
        return means

    def get_best_coeficients(self, means, shadow_mask, shadows, use_lab=False):
        lab_shadows = Step1().convert_to_lab(shadows) if use_lab else shadows
        coefs = [1000, 1000, 1000]
        distance = 100000000
        for mean in means:
            s_avg = [sm / cv2.sumElems(shadow_mask/255)[0] for sm in cv2.sumElems(lab_shadows)]
            if use_lab:
                start_idx = 1
            else:
                start_idx = 0
            diffs = [math.fabs(mean[i] - s_avg[i]) for i in range(start_idx, 3)]
            new_dis = sum(diffs)
            if new_dis < distance:
                distance = new_dis
                coefs = mean
        #print("distance: %d" % distance)
        return [coefs[i] / s_avg[i] for i in range(3)]