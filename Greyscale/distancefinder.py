# Dilate/Erode mask
import cv2
import numpy as np
import time
from LAB.shadow_detection.step1 import Step1
import math
from LAB.shadow_detection.utils import show_and_save
from color_segmentator import ColorSegmentator
from settings import settings


class DistanceFinder(object):
    def __init__(self, image, dilated_shadow_mask, method=0):
        self.shadow_regions = []
        self.shadow_region_masks = []
        self.shadow_regions_means = []
        self.light_regions = []
        self.light_region_masks = []
        self.light_regions_means = []
        self.dilated_shadows_mask = dilated_shadow_mask

        self.color_region_masks = ColorSegmentator().segment_image(image)

        self.shadow_region_masks, self.small_shadow_region_masks = self.get_region_masks(dilated_shadow_mask)

        #recompute the shadow mask using all the region masks to avoid troubles with edges
        dilated_shadow_mask = np.zeros((dilated_shadow_mask.shape[0], dilated_shadow_mask.shape[1]), np.uint8)

        for region_mask in self.small_shadow_region_masks:
            dilated_shadow_mask += region_mask

        for region_mask in self.shadow_region_masks:
            region = self.apply_mask(image, region_mask)
            self.shadow_regions.append(region)
            dilated_shadow_mask += region_mask
        #---------------------------------------------------

        light_mask = 255 - dilated_shadow_mask
        lights = self.apply_mask(image, light_mask)
        self.light_region_masks, self.light_regions = self.generate_regions(lights)
        self.light_regions_means = self.calculate_light_regions_means(method=method)
        self.shadow_regions_means = self.calculate_shadow_regions_means(method=method)

        #mono image regions initialization

        self.mono_shadow_regions = []
        self.mono_shadow_regions_means = []
        self.mono_light_regions = []
        self.mono_light_regions_means = []

    def run(self, mono_image, method=0):
        self.mono_light_regions = [self.apply_multi_mask(mono_image, np.float64(light_region_mask)) for light_region_mask in self.light_region_masks]
        self.mono_shadow_regions = [self.apply_multi_mask(mono_image, np.float64(shadow_region_mask)) for shadow_region_mask in self.shadow_region_masks]

        self.mono_light_regions_means = self.calculate_regions_means(self.light_region_masks, self.mono_light_regions, method)
        self.mono_shadow_regions_means = self.calculate_regions_means(self.shadow_region_masks, self.mono_shadow_regions, method)

        distance = 0
        for shadow_region_index in range(len(self.shadow_region_masks)):
            light_region_index = self.get_closest_region_index(shadow_region_index, method=method)
            distance += abs(self.mono_light_regions_means[light_region_index][0]-
                            self.mono_shadow_regions_means[shadow_region_index][0])
        return distance

    def apply_mask(self, image, mask):
        return cv2.bitwise_and(image, cv2.merge([mask, mask, mask]))

    def sanitize_mask(self, dilated_mask, original_mask):
        return cv2.bitwise_and(dilated_mask, original_mask)

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

        min_size = shadow_mask.shape[0] * shadow_mask.shape[1] / settings['min_size_factor']

        for i in range(len(contours)):
            blank = np.zeros((shadow_mask.shape[0], shadow_mask.shape[1], 1), np.uint8)
            region_mask = cv2.drawContours(blank, contours, i, 255, -1)
            for color_mask in self.color_region_masks:
                subregion = self.apply_multi_mask(region_mask,color_mask)
                s = cv2.sumElems(subregion/255)[0]
                if s > min_size:
                    big_regions.append(subregion)
                else:
                    small_regions.append(subregion)

        return big_regions, small_regions

    def saturate(self, image):
        return cv2.convertScaleAbs(image)

    def apply_multi_mask(self, image, mask):
        return cv2.bitwise_and(image, mask)

    def generate_regions(self, lights, is_color=True):
        msft = cv2.medianBlur(lights, 5)
        gray_msft = cv2.cvtColor(msft, cv2.COLOR_BGR2GRAY) if is_color else lights
        big_regions_masks, small_regions = self.get_region_masks(gray_msft)
        light_regions = []
        valid_masks = []

        min_size = lights.shape[0] * lights.shape[1] / 20
        for i in range(len(big_regions_masks)):
            mask = big_regions_masks[i]

            region = self.apply_mask(lights, mask) if is_color else self.apply_multi_mask(lights, mask)
            valid_pixels = cv2.sumElems(mask/255)[0]
            if valid_pixels > min_size:
                light_regions.append(region)
                valid_masks.append(mask)
                #show_and_save(str(i), "dbg_img/light_region", 'png', region)
        return valid_masks, light_regions

    def calculate_light_regions_means(self, method=0):
        return self.calculate_regions_means(self.light_region_masks, self.light_regions, method)

    def calculate_shadow_regions_means(self, method=0):
            return self.calculate_regions_means(self.shadow_region_masks, self.shadow_regions, method)

    def calculate_regions_means(self, region_masks, regions, method=0):
        #METHODS
        # 0 BGR
        # 1 LAB
        # 2 HSV
        means = []
        for i in range(len(region_masks)):
            region_mask = region_masks[i]
            region = regions[i]
            region = Step1().convert_to_lab(region) if method == 1 else region
            region = Step1().convert_to_hsv(region) if method == 2 else region
            means.append(self.get_means(region, region_mask))
        return means

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
        return index

    def get_means(self, region, mask):
        return [sm / cv2.sumElems(mask/255)[0] for sm in cv2.sumElems(region)]

    def print_region_matches(self, printer):
        for shadow_index in range(len(self.shadow_regions)):
            light_index = self.get_closest_region_index(shadow_index)
            shadow = self.shadow_regions[shadow_index]
            light = self.light_regions[light_index]
            out = np.concatenate((shadow, light), axis=1)
            printer(shadow_index, out)

    def print_light_regions(self, printer):
        for light_index in range(len(self.light_regions)):
            light = self.light_regions[light_index]
            printer(light_index, light)

    def print_shadow_regions(self, printer):
        for shadow_index in range(len(self.shadow_regions)):
            shadow = self.shadow_regions[shadow_index]
            printer(shadow_index, shadow)