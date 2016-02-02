# Dilate/Erode mask
import cv2
import numpy as np
import time
from LAB.shadow_detection.step1 import Step1
import math
from LAB.shadow_detection.utils import show_and_save
from color_segmentator import ColorSegmentator
from settings import settings
from LAB.shadow_detection.utils import equalize_hist_3d


class DistanceFinder(object):
    def __init__(self, image, dilated_shadow_mask, method=0):
        self.method = method
        self.shadow_regions = []
        self.shadow_centroids = []
        self.shadow_region_masks = []
        self.shadow_regions_means = []
        self.light_regions = []
        self.light_region_masks = []
        self.light_regions_means = []
        self.dilated_shadows_mask = dilated_shadow_mask

        self.color_region_masks = ColorSegmentator().segment_image(image, settings)

        if method == 1:
            LAB_img = Step1().convert_to_lab(image)
            self.AB_max = LAB_img.max()
            self.AB_min = LAB_img.min()

        self.shadow_region_masks, \
        small_shadow_region_masks, \
        self.shadow_centroids, \
        small_centroids= \
            self.get_region_masks(dilated_shadow_mask)

        #recompute the shadow mask using all the region masks to avoid troubles with edges
        dilated_shadow_mask = np.zeros((dilated_shadow_mask.shape[0], dilated_shadow_mask.shape[1]),
                                       np.uint8)

        for region_mask in small_shadow_region_masks:
            dilated_shadow_mask += region_mask

        for region_mask in self.shadow_region_masks:
            region = self.apply_mask(image, region_mask)
            self.shadow_regions.append(region)
            dilated_shadow_mask += region_mask
        #---------------------------------------------------

        light_mask = 255 - dilated_shadow_mask
        lights = self.apply_mask(image, light_mask)
        self.light_region_masks, self.light_regions,\
           self.light_centroids = self.generate_regions(lights)
        self.light_regions_means = self.calculate_light_regions_means(method=self.method)
        self.shadow_regions_means = self.calculate_shadow_regions_means(method=self.method)

        #mono image regions initialization

        self.mono_shadow_regions = []
        self.mono_shadow_regions_means = []
        self.mono_light_regions = []
        self.mono_light_regions_means = []

        # to standarize spatial distance
        self.diagonal = pow(pow(image.shape[0], 2) + pow(image.shape[1], 2), 1/2.0)
        self.matches = [self.get_closest_region_index(shadow_region_index, method=self.method)
                        for shadow_region_index in range(len(self.shadow_regions))]


    def run(self, mono_image):
        #todo: check if necessary
        mono_image = np.float64(equalize_hist_3d(mono_image))
        self.mono_light_regions = [self.apply_multi_mask(mono_image, np.float64(light_region_mask))
                                   for light_region_mask in self.light_region_masks]
        self.mono_shadow_regions = [self.apply_multi_mask(mono_image, np.float64(shadow_region_mask))
                                    for shadow_region_mask in self.shadow_region_masks]

        self.mono_light_regions_means = self.calculate_regions_means(self.light_region_masks,
                                                                     self.mono_light_regions)
        self.mono_shadow_regions_means = self.calculate_regions_means(self.shadow_region_masks,
                                                                      self.mono_shadow_regions)

        distance = 0
        for shadow_region_index in range(len(self.shadow_region_masks)):
            light_region_index = self.matches[shadow_region_index]
            if light_region_index >= 0:
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
        big_centroids = []
        small_centroids = []
        min_size = shadow_mask.shape[0] * shadow_mask.shape[1] / settings['min_size_factor']

        for i in range(len(contours)):
            blank = np.zeros((shadow_mask.shape[0], shadow_mask.shape[1], 1), np.uint8)
            region_mask = cv2.drawContours(blank, contours, i, 255, -1)
            for color_mask in self.color_region_masks:
                subregion = self.apply_multi_mask(region_mask,color_mask)
                s = cv2.sumElems(subregion/255)[0]
                if s > min_size:
                    big_regions.append(subregion)
                    #big_centroids.append(centroids[i])
                else:
                    small_regions.append(subregion)
                    #small_centroids.append(centroids[i])

        big_centroids = self.get_centroids(big_regions)
        return big_regions, small_regions, big_centroids, small_centroids

    def get_centroids(self, contours):
        centroids = []
        for contour in contours:
            M = cv2.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroids.append((cx, cy))
        return centroids

    def saturate(self, image):
        return cv2.convertScaleAbs(image)

    def apply_multi_mask(self, image, mask):
        return cv2.bitwise_and(image, mask)

    def generate_regions(self, lights, is_color=True):
        cv2.imwrite('dbg_img/lights.png', lights)
        blured = cv2.medianBlur(lights, 5)
        lights_2 = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY) if is_color else lights
        big_regions_masks, small_regions\
            ,big_centroids, small_centroids= self.get_region_masks(lights_2)
        light_regions = []
        valid_masks = []
        valid_centroids = []
        min_size = lights.shape[0] * lights.shape[1] / 20
        for i in range(len(big_regions_masks)):
            mask = big_regions_masks[i]

            region = self.apply_mask(lights, mask) if is_color else self.apply_multi_mask(lights, mask)
            valid_pixels = cv2.sumElems(mask/255)[0]
            if valid_pixels > min_size:
                light_regions.append(region)
                valid_masks.append(mask)
                valid_centroids.append(big_centroids[i])
        return valid_masks, light_regions, valid_centroids

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
            #is color image
            if len(region.shape) > 2 and region.shape[2] == 3:
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

        light_regions_means = self.get_means(lights, light_mask)
        shadows_means = self.get_means(shadows, shadow_mask)
        return [light_regions_means[i] / shadows_means[i] for i in range(3)]

    def get_closest_region_index(self, shadow_index, method=0):
        #METHODS
        # 0 BGR
        # 1 LAB
        # 2 HSV
        shadow_region_mask = self.shadow_region_masks[shadow_index]
        shadow_region = self.shadow_regions[shadow_index]
        shadow_region = Step1().convert_to_lab(shadow_region) if method == 1 else shadow_region
        shadow_region = Step1().convert_to_hsv(shadow_region) if method == 2 else shadow_region
        sh_avg = self.get_means(shadow_region, shadow_region_mask)
        index = 0
        distance = 100000000
        for light_index in range(len(self.light_regions_means)):
            light_region_mean = self.light_regions_means[light_index]
            color_distance = self.color_distance(light_region_mean, sh_avg, method)
            if color_distance >= 0:
                spatial_distance = self.spatial_distance(shadow_index, light_index)
                new_dis = settings['region_distance_balance']*color_distance + (1-settings['region_distance_balance']) * spatial_distance
                if new_dis < distance:
                    distance = new_dis
                    index = light_index
        return index if distance < settings['max_color_dist'] else -1

    def spatial_distance(self, shadow_index, light_index):
        shadow_centroid = self.shadow_centroids[shadow_index]
        light_centroid = self.light_centroids[light_index]
        dis = pow(pow(shadow_centroid[0]-light_centroid[0],2)+pow(shadow_centroid[1]-light_centroid[1],2), 0.5)
        return dis / self.diagonal

    def color_distance(self, light_region_mean, sh_avg, method=None):
        if method is None:
            method = self.method
        if method == 0:
            color_indices = [0,1,2]
            light_indices = [0,1,2]
            value_range= 255
        elif method == 1:
            color_indices = [1,2]
            light_indices = [0]
            value_range= self.AB_max - self.AB_min
        elif method == 2:
            color_indices = [0]
            light_indices = [2]
            value_range= 255
        lrm = [light_region_mean[j] for j in light_indices]
        #  todo: parametrize
        if self.is_not_black_region(lrm, method):
            diffs = [math.fabs(light_region_mean[j] - sh_avg[j]) for j in color_indices]
            return sum(diffs) / (value_range * len(diffs))
        else:
            return -1

    @staticmethod
    def is_not_black_region(region_means, method):  #todo: fix for method 1 y 2
        return sum(region_means) > 50 or method > 0

    def get_means(self, region, mask):
        try:
            region = self.apply_mask(region, mask) if len(region.shape) > 2 and \
                                                      region.shape[2] == 3 \
                else self.apply_multi_mask(region.astype(np.uint8()), mask)
        except:
            pass
        return [sm / cv2.sumElems(mask/mask)[0] for sm in cv2.sumElems(region)]

    def print_region_matches(self, printer):
        for shadow_index in range(len(self.shadow_regions)):
            light_index = self.matches[shadow_index]
            if light_index >= 0:
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

    def print_region_distances(self, printer):
        for shadow_index in range(len(self.shadow_regions)):
            shadow = self.shadow_regions[shadow_index]
            shadow_region_mask = self.shadow_region_masks[shadow_index]
            sh_avg = self.get_means(shadow, shadow_region_mask)
            for light_index in range(len(self.light_regions)):
                light = self.light_regions[light_index]
                light_region_mean = self.light_regions_means[light_index]
                color_distance = self.color_distance(light_region_mean, sh_avg)
                spatial_distance = self.spatial_distance(shadow_index, light_index)
                new_dis = settings['region_distance_balance']*color_distance + (1-settings['region_distance_balance']) * spatial_distance

                out = np.concatenate((shadow, light), axis=1)

                shadow_centroid = self.shadow_centroids[shadow_index]
                cv2.circle(out, shadow_centroid, 10, (255,0,255))
                light_centroid = self.light_centroids[light_index]
                cv2.circle(out, light_centroid, 10, (255,0,255))

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(out, "sh-color:"+str(sh_avg), (30,200), font, 1,(255,255,255))
                cv2.putText(out, "light-color:"+str(light_region_mean), (30,250), font, 1,(255,255,255))
                cv2.putText(out, "color-dist:"+repr(color_distance), (30,300), font, 1,(255,255,255))
                cv2.putText(out, "space-dist:"+repr(spatial_distance), (30,350), font, 1,(255,255,255))
                cv2.putText(out, "total-dist:"+repr(new_dis), (30,400), font, 1,(255,255,255))
                printer(shadow_index, light_index, out)