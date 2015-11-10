# Dilate/Erode mask
import cv2
import numpy as np
import time
from LAB.shadow_detection.step1 import Step1
import math
from LAB.shadow_detection.utils import show_and_save


class Step5(object):
    def __init__(self):
        self.shadow_regions = []
        self.shadow_region_masks = []
        self.light_regions = []
        self.light_region_masks = []
        self.light_regions_means = []

    def run(self, image, dilated_shadow_mask, shadow_mask, method=0):

        self.shadow_region_masks, self.small_shadow_region_masks = self.get_region_masks(dilated_shadow_mask)

        #recompute the shadow mask using all the region masks to avoid troubles with edges
        dilated_shadow_mask = np.zeros((dilated_shadow_mask.shape[0], dilated_shadow_mask.shape[1], 1), np.uint8)

        for region_mask in self.small_shadow_region_masks:
            dilated_shadow_mask += region_mask

        for region_mask in self.shadow_region_masks:
            region = self.apply_mask(image, region_mask)
            self.shadow_regions.append(region)
            dilated_shadow_mask += region_mask
        #---------------------------------------------------

        light_mask = 255 - dilated_shadow_mask
        lights = self.apply_mask(image, light_mask)
        self.generate_light_regions(lights)
        self.light_regions_means = self.calculate_light_regions_means(method=method)

        result = image.copy()

        for shadow_region_index in range(len(self.shadow_region_masks)):
            region = self.shadow_regions[shadow_region_index]
            region_mask = self.shadow_region_masks[shadow_region_index]
            light_region_index = self.get_closest_region_index(shadow_region_index, method=method)
            coef = self.get_coeficients(shadow_region_index, light_region_index, method=method)
            region_mask = self.sanitize_mask(region_mask, shadow_mask)

            #show_and_save("", "dbg_img/shadow_mask", 'png', shadow_mask)
            #show_and_save(str(shadow_region_index)+"_mask", "dbg_img/region", 'png', region_mask)
            #show_and_save(str(shadow_region_index)+"_A", "dbg_img/region", 'png', region)

            region = self.apply_coefficients(coef, region, method=method)

            #show_and_save(str(shadow_region_index)+"_B", "dbg_img/region", 'png', region)

            region = self.apply_mask(region, region_mask)
            no_region = 255 - region_mask
            result = self.apply_mask(result, no_region)
            #show_and_save(str(shadow_region_index)+"_C", "dbg_img/region", 'png', region)
            result += region

        return result

    def apply_mask(self, image, mask):
        return cv2.bitwise_and(image, cv2.merge([mask, mask, mask]))

    def sanitize_mask(self, dilated_mask, original_mask):
        return cv2.bitwise_and(dilated_mask, original_mask)

    #def get_coeficients(self, light_mask, lights, shadow_mask, shadows):
    #    l_avg = self.get_means(lights, light_mask)
    #    s_avg = self.get_means(shadows, shadow_mask)
    #    return [l_avg[i] / s_avg[i] for i in range(3)]

    def apply_coefficients(self, coef, image, method=0):
        #METHODS
        # 0 BGR
        # 1 LAB
        # 2 HSV
        image = Step1().convert_to_lab(image) if method == 1 else image
        image = Step1().convert_to_hsv(image) if method == 2 else image
        l, a, b = cv2.split(image)
        l = np.uint16(l)
        a = np.uint16(a)
        b = np.uint16(b)

        if method == 0:
            l *= coef[0]
            a *= coef[1]
            b *= coef[2]
        elif method == 1:
            l *= coef[0]
        elif method == 2:
            l *= coef[0]
            a *= coef[1]
            b *= coef[2]

        result = self.saturate(cv2.merge([l, a, b]))
        if method == 1:
            result = Step1().convert_lab_to_bgr(result)
        if method == 2:
            result = Step1().convert_hsv_to_bgr(result)
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

    def generate_light_regions(self, lights):
        msft = cv2.medianBlur(lights, 5)
        gray_msft = cv2.cvtColor(msft, cv2.COLOR_BGR2GRAY)
        self.light_region_masks, small_regions = self.get_region_masks(gray_msft)
        valid_masks = []
        for i in range(len(self.light_region_masks)):
            mask = self.light_region_masks[i]

            region = self.apply_mask(lights, mask)
            valid_pixels = cv2.sumElems(mask/255)[0]
            if valid_pixels > 1500:
                self.light_regions.append(region)
                valid_masks.append(mask)
                #show_and_save(str(i), "dbg_img/light_region", 'png', region)
        self.light_region_masks = valid_masks
    # DEPRECATE!!!
    def calculate_light_regions_means(self, method=0):
        #METHODS
        # 0 BGR
        # 1 LAB
        # 2 HSV
        means = []
        for i in range(len(self.light_region_masks)):
            region_mask = self.light_region_masks[i]
            region = self.light_regions[i]
            region = Step1().convert_to_lab(region) if method == 1 else region
            region = Step1().convert_to_hsv(region) if method == 2 else region
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

    def get_closest_region_index(self, shadow_index, method=0):
        #METHODS
        # 0 BGR
        # 1 LAB
        # 2 HSV
        ts = time.time()
        shadow_region_mask = self.shadow_region_masks[shadow_index]
        shadow_region = self.shadow_regions[shadow_index]
        shadow_region = Step1().convert_to_lab(shadow_region) if method == 1 else shadow_region
        shadow_region = Step1().convert_to_hsv(shadow_region) if method == 2 else shadow_region
        s_avg = self.get_means(shadow_region, shadow_region_mask)
        index = 0
        distance = 100000000
        for i in range(len(self.light_regions_means)):
            light_region_mean = self.light_regions_means[i]
            if method == 0:
                start_idx = 0
                end_idx = 3
                mn_start = 0
                mn_end = 3
            elif method == 1:
                start_idx = 1
                end_idx = 3
                mn_start = 0
                mn_end = 1
            elif method == 2:
                start_idx = 0
                end_idx = 2
                mn_start = 2
                mn_end = 3
            lrm = [light_region_mean[j] for j in range(mn_start, mn_end)]
            if sum(lrm) > 50:
                diffs = [math.fabs(light_region_mean[j] - s_avg[j]) for j in range(start_idx, end_idx)]
                new_dis = sum(diffs)
                if new_dis < distance:
                    distance = new_dis
                    index = i
        print("%d -> %d" % (shadow_index, index))
        return index

    def get_means(self, region, mask):
        return [sm / cv2.sumElems(mask/255)[0] for sm in cv2.sumElems(region)]