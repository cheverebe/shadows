# Dilate/Erode mask
import cv2
import numpy as np
from LAB.shadow_detection.step1 import Step1
import math


class Step5(object):
    def __init__(self):
        self.shadow_regions = []
        self.shadow_region_masks = []
        self.light_regions = []
        self.light_region_masks = []
        self.light_regions_means = []

    def run(self, image, dilated_shadow_mask, shadow_mask, use_lab=False):

        shadow_region_masks, small_shadow_region_masks = self.get_region_masks(dilated_shadow_mask)

        #recompute the shadow mask using all the region masks to avoid troubles with edges
        dilated_shadow_mask = np.zeros((dilated_shadow_mask.shape[0], dilated_shadow_mask.shape[1], 1), np.uint8)

        for region_mask in small_shadow_region_masks:
            dilated_shadow_mask += region_mask

        for region_mask in self.shadow_region_masks:
            region = self.apply_mask(image, region_mask)
            self.shadow_regions.append(region)
            dilated_shadow_mask += region_mask
        #---------------------------------------------------


        light_mask = 255 - dilated_shadow_mask
        lights = self.apply_mask(image, light_mask)

        self.light_regions_means = self.calculate_light_regions_means(lights, use_lab=use_lab)

        result = image.copy()

        for region_mask in self.shadow_region_masks:
            region = self.apply_mask(image, region_mask)
            coef = self.get_best_coeficients(region_mask, region, use_lab=use_lab)
            region_mask = self.sanitize_mask(region_mask, shadow_mask)
            region = self.apply_coefficients(coef, region, use_lab=use_lab)
            region = self.apply_mask(region, region_mask)
            no_region = 255 - region_mask
            result = self.apply_mask(result, no_region)
            result += region

        return result

    def apply_mask(self, image, mask):
        return cv2.bitwise_and(image, cv2.merge([mask, mask, mask]))

    def sanitize_mask(self, dilated_mask, original_mask):
        return cv2.bitwise_and(dilated_mask, original_mask)

    def get_coeficients(self, light_mask, lights, shadow_mask, shadows):
        l_avg = self.get_means(lights, light_mask)
        s_avg = self.get_means(shadows, shadow_mask)
        return [l_avg[i] / s_avg[i] for i in range(3)]

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

        for i in range(len(contours)):
            blank = np.zeros((shadow_mask.shape[0], shadow_mask.shape[1], 1), np.uint8)
            region_mask = cv2.drawContours(blank, contours, i, 255, -1)
            s = cv2.sumElems(region_mask/255)[0]
            if s > 700:
                big_regions.append(region_mask)
            else:
                small_regions.append(region_mask)

        return big_regions, small_regions

    def saturate(self, image):
        return cv2.convertScaleAbs(image)

    def apply_multi_mask(self, image, mask):
        return cv2.bitwise_and(image, mask)

    # DEPRECATE!!!
    def calculate_light_regions_means(self, lights, use_lab=False):
        msft = cv2.medianBlur(lights, 5)
        gray_msft = cv2.cvtColor(msft, cv2.COLOR_BGR2GRAY)
        big_regions, small_regions = self.get_region_masks(gray_msft)
        means = []
        for region_mask in big_regions:
            lab_lights = Step1().convert_to_lab(lights) if use_lab else lights
            region = self.apply_mask(lab_lights, region_mask)
            means.append(self.get_means(region, region_mask))
        return means

    # DEPRECATE!!!
    def get_best_coeficients(self, shadow_mask, shadows, use_lab=False):
        lab_shadows = Step1().convert_to_lab(shadows) if use_lab else shadows
        coefs = [1000, 1000, 1000]
        distance = 100000000
        for mean in self.light_regions_means:
            s_avg = self.get_means(lab_shadows, shadow_mask)
            if use_lab:
                start_idx = 1
            else:
                start_idx = 0
            diffs = [math.fabs(mean[i] - s_avg[i]) for i in range(start_idx, 3)]
            new_dis = sum(diffs)
            if new_dis < distance:
                distance = new_dis
                coefs = mean
        return [coefs[i] / s_avg[i] for i in range(3)]

    def get_coeficients(self, shadow_region_index, light_region_index, method=0):
        #METHODS
        # 0 BGR
        # 1 LAB
        # 2 HSV
        shadows = self.shadow_regions[shadow_region_index]
        shadow_mask = self.shadow_region_masks[shadow_region_index]

        shadows = Step1().convert_to_lab(shadows) if method == 1 else shadows
        shadows = Step1().convert_to_hsv(shadows) if method == 2 else shadows

        lights = self.light_regions[light_region_index]
        light_mask = self.light_region_masks[light_region_index]

        lights = Step1().convert_to_lab(lights) if method == 1 else lights
        lights = Step1().convert_to_hsv(lights) if method == 2 else lights

        light_region_means = self.get_means(lights, light_mask)
        shadows_means = self.get_means(shadows, shadow_mask)
        return [light_region_means[i] / shadows_means[i] for i in range(3)]

    def get_closest_region_index(self, shadow_region_mask, shadow_region, method=0):
        #METHODS
        # 0 BGR
        # 1 LAB
        # 2 HSV
        shadow_region = Step1().convert_to_lab(shadow_region) if method == 1 else shadow_region
        shadow_region = Step1().convert_to_hsv(shadow_region) if method == 2 else shadow_region
        s_avg = self.get_means(shadow_region, shadow_region_mask)
        index = 0
        distance = 100000000
        for i in range(len(self.light_regions)):
            light_region = self.light_regions[i]
            light_region_mask = self.light_regions[i]
            light_region_mean = self.get_means(light_region, light_region_mask)
            if method == 0:
                start_idx = 0
                end_idx = 3
            elif method == 1:
                start_idx = 1
                end_idx = 3
            elif method == 2:
                start_idx = 0
                end_idx = 2
            diffs = [math.fabs(light_region_mean[i] - s_avg[i]) for i in range(start_idx, end_idx)]
            new_dis = sum(diffs)
            if new_dis < distance:
                distance = new_dis
                index = i
        return index

    def get_means(self, region, mask):
        return [sm / cv2.sumElems(mask/255)[0] for sm in cv2.sumElems(region)]