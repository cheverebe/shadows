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


class Region(object):
    def __init__(self, image, mask, colorspace):
        self.image = self.apply_mask(image(), mask)
        self.mask = mask
        self.means = self.calculate_means()
        self.cetroid = self.get_centroid()
        self.colorspace = colorspace

    def calculate_means(self):
        region = self.apply_mask(self.image.astype(np.uint8()), self.mask)  #todo: remove because shpuld be necessary
        return [sm / cv2.sumElems(self.mask/self.mask)[0] for sm in cv2.sumElems(region)]

    def apply_mask(self, image, mask):
        if len(mask.shape) > 2:
            if not len(image.shape) == len(mask.shape):
                print "Mask should be a grayscale or have the same depth of image"
                raise Exception
            if not len(image.shape[:2]) == len(mask.shape[:2]):
                print "Mask should have the same size of the image"
                raise Exception
            return cv2.bitwise_and(image, cv2.merge(mask))
        else:
            return cv2.bitwise_and(image, cv2.merge([mask, mask, mask]))

    def get_centroid(self):
        M = cv2.moments(self.mask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return (cx, cy)

    def spatial_distance(self, other_region):
        dis = pow(pow(self.centroid[0]-other_region.centroid[0],2)+pow(self.centroid[1]-other_region.centroid[1],2), 0.5)
        return dis / self.diagonal

    def color_distance(self, other_region):
        if not self.colorspace == other_region.colorspace:
            raise Exception
        #  todo: parametrize
        if self.colorspace.is_not_black_region(other_region):
            diffs = [math.fabs(self.means[j] - other_region.means[j]) for j in self.colorspace.color_indices]
            return sum(diffs) / (self.colorspace.value_range() * len(diffs))
        else:
            return -1


class ColorSpaceMethod(object):
    def __init__(self):
        pass

    def value_range(self, image):
        raise NotImplementedError

    def pre_process_image(self, image):
        raise NotImplementedError

    def post_process_image(self, image):
        raise NotImplementedError

    def color_indices(self):
        raise NotImplementedError

    def light_indices(self,):
        raise NotImplementedError

    def is_not_black_region(self, region):  # todo: fix for method 1 y 2
        means = region.means
        region_light_means = [means[j] for j in self.light_indices()]
        return sum(region_light_means) > 50


class DistanceFinder(object):
    def __init__(self, image, dilated_shadow_mask, method=0):
        self.image = image
        self.method = method
        self.shadow_regions = []
        self.light_regions = []
        self.dilated_shadows_mask = dilated_shadow_mask

        if method == 1:
            LAB_img = Step1().convert_to_lab(image)
            self.AB_max = LAB_img.max()
            self.AB_min = LAB_img.min()

        self.shadow_regions = self.generate_regions(dilated_shadow_mask)

        #recompute the shadow mask using all the region masks to avoid troubles with edges
        #dilated_shadow_mask = np.zeros((dilated_shadow_mask.shape[0], dilated_shadow_mask.shape[1]),
        #                               np.uint8)

        #for region_mask in small_shadow_region_masks:
        #    dilated_shadow_mask += region_mask

        #for region_mask in self.shadow_region_masks:
        #    region = self.apply_mask(image, region_mask)
        #    self.shadow_regions.append(region)
        #    dilated_shadow_mask += region_mask
        #---------------------------------------------------

        light_mask = 255 - dilated_shadow_mask
        self.light_regions = self.generate_regions(light_mask)
        #mono image regions initialization

        self.mono_shadow_regions = []
        self.mono_light_regions = []

        # to standarize spatial distance
        self.diagonal = pow(pow(image.shape[0], 2) + pow(image.shape[1], 2), 1/2.0)
        self.matches = {}
        for shadow_region in self.shadow_regions:
            self.matches[shadow_region] = self.get_closest_region(shadow_region)

    def run(self, mono_image):
        #todo: check if necessary
        mono_image = np.float64(equalize_hist_3d(mono_image))
        mono_light_regions = {}
        for light_region in self.light_regions:
            mono_light_regions[light_region] = Region(mono_image, np.float64(light_region.mask))

        mono_shadow_regions = {}
        for shadow_region in self.shadow_regions:
            mono_shadow_regions[shadow_region] = Region(mono_image, np.float64(shadow_region.mask))

        distance = 0
        for shadow_region in self.shadow_regions:
            light_region = self.matches[shadow_region]
            if light_region:
                distance += abs(light_region.means[0] - shadow_region.means[0])
        return distance

    def apply_mask(self, image, mask):
        return cv2.bitwise_and(image, cv2.merge([mask, mask, mask]))

    def sanitize_mask(self, dilated_mask, original_mask):
        return cv2.bitwise_and(dilated_mask, original_mask)

    def generate_regions(self, main_mask, colorspace):
        color_region_masks = ColorSegmentator().segment_image(self.image, settings)
        mask = main_mask.copy()
        image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        big_regions = []
        small_regions = []
        min_size = main_mask.shape[0] * main_mask.shape[1] / settings['min_size_factor']

        for i in range(len(contours)):
            blank = np.zeros((main_mask.shape[0], main_mask.shape[1], 1), np.uint8)
            region_mask = cv2.drawContours(blank, contours, i, 255, -1)
            for color_mask in color_region_masks:
                subregion = self.apply_multi_mask(region_mask, color_mask)
                s = cv2.sumElems(subregion/255)[0]
                region = Region(self.image, subregion, colorspace)
                if s > min_size:
                    big_regions.append(region)
                else:
                    small_regions.append(region)

        return big_regions, small_regions

    def apply_multi_mask(self, image, mask):
        return cv2.bitwise_and(image, mask)

    def get_closest_region(self, shadow_index):
        shadow_region = self.shadow_regions[shadow_index]
        closest_region = None
        distance = 100000000
        for light_region in self.light_regions:
            color_distance = shadow_region.color_distance(light_region)
            if color_distance >= 0:
                spatial_distance = shadow_region.spatial_distance(light_region)
                new_dis = settings['region_distance_balance']*color_distance + (1-settings['region_distance_balance']) * spatial_distance
                if new_dis < distance:
                    distance = new_dis
                    closest_region = light_region
        return closest_region if distance < settings['max_color_dist'] else None

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