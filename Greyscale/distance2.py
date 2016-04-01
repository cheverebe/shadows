# Dilate/Erode mask
import cv2
import numpy as np
import math
from color_segmentator import ColorSegmentator
from settings import settings
from LAB.shadow_detection.utils import equalize_hist_3d


class Region(object):
    def __init__(self, image, mask, colorspace=None):
        if colorspace:
            self.image = colorspace.pre_process_image(image)
        else:
            self.image = image.copy()
        self.image = self.apply_mask(self.image, mask)
        self.mask = mask.copy()
        self.means = self.calculate_means()
        self.centroid = self.get_centroid()
        self.colorspace = colorspace
        self.diagonal = pow(pow(image.shape[0], 2) + pow(image.shape[1], 2), 1/2.0)

    def calculate_means(self):
        pixel_count = cv2.sumElems(self.mask/255)[0]
        return [sm / pixel_count if pixel_count > 0 else 0
                for sm in cv2.sumElems(self.image)]

    def apply_mask(self, image, mask):
        if len(mask.shape) > 2 or len(image.shape) == len(mask.shape):
            if not len(image.shape) == len(mask.shape):
                print "Mask should be a grayscale or have the same depth of image"
                raise Exception
            if not len(image.shape[:2]) == len(mask.shape[:2]):
                print "Mask should have the same size of the image"
                raise Exception
            return cv2.bitwise_and(image, mask)
        else:
            return cv2.bitwise_and(image, cv2.merge([mask, mask, mask]))

    def get_centroid(self):
        M = cv2.moments(self.mask)

        if M['m00'] == 0: #if region is empty
            return (self.mask.shape[0] / 2,
                    self.mask.shape[1] / 2)

        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return (cx, cy)

    def spatial_distance(self, other_region):
        dis = pow(pow(self.centroid[0]-other_region.centroid[0],2)+pow(self.centroid[1]-other_region.centroid[1],2), 0.5)
        return dis / self.diagonal

    def color_distance(self, other_region):

        if not self.colorspace == other_region.colorspace:
            raise Exception

        if self.colorspace.is_not_black_region(other_region.image):
            diffs = [math.fabs(self.means[j] - other_region.means[j])
                     for j in self.colorspace.color_indices()]
            return sum(diffs) / (self.colorspace.value_range(self.image) * len(diffs))
        else:
            return -1

    def showable_image(self):
        return self.colorspace.post_process_image(self.image)


class DistanceFinder(object):
    def __init__(self, image, dilated_shadow_mask, colorspace):
        self.image = image
        self.colorspace = colorspace
        self.shadow_regions = []
        self.light_regions = []
        self.dilated_shadows_mask = dilated_shadow_mask

        self.shadow_regions = self.generate_regions(dilated_shadow_mask, colorspace)

        light_mask = 255 - dilated_shadow_mask
        self.light_regions = self.generate_regions(light_mask, colorspace)
        #mono image regions initialization

        self.mono_shadow_regions = []
        self.mono_light_regions = []

        # to standarize spatial distance
        self.diagonal = pow(pow(image.shape[0], 2) + pow(image.shape[1], 2), 1/2.0)
        self.matches = {}
        for shadow_region in self.shadow_regions:
            self.matches[shadow_region] = self.get_closest_region(shadow_region)

    def run(self, mono_image):
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
        color_region_masks = ColorSegmentator(settings).segment_image(self.image)
        mask = main_mask.copy()
        image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        big_regions = []
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

        return big_regions

    def apply_multi_mask(self, image, mask):
        return cv2.bitwise_and(image, mask)

    def get_closest_region(self, shadow_region):
        closest_region = None
        distance = 100000000
        for light_region in self.light_regions:
            color_distance = shadow_region.color_distance(light_region)
            if color_distance >= 0:
                spatial_distance = shadow_region.spatial_distance(light_region)
                new_dis = settings['region_distance_balance']*color_distance + \
                          (1-settings['region_distance_balance']) * spatial_distance
                if new_dis < distance:
                    distance = new_dis
                    closest_region = light_region
        return closest_region if distance < settings['max_color_dist'] else None

    def print_region_matches(self, printer):
        i = 0
        for shadow in self.shadow_regions:
            light = self.matches[shadow]
            if light >= 0:
                out = np.concatenate((shadow.showable_image(),
                                      light.showable_image()), axis=1)
                printer(i, out)
                i += 1

    def print_light_regions(self, printer):
        i = 0
        for light in self.light_regions:
            printer(i, light.showable_image())
            i += 1

    def print_shadow_regions(self, printer):
        i = 0
        for shadow in self.shadow_regions:
            printer(i, shadow.showable_image())
            i += 1

    def print_region_distances(self, printer):
        i = 0
        for shadow in self.shadow_regions:
            sh_avg = shadow.means
            i += 1
            j = 0
            for light in self.light_regions:
                light_region_mean = light.means
                color_distance = light.color_distance(shadow)
                spatial_distance = light.spatial_distance(shadow)
                new_dis = settings['region_distance_balance']*color_distance + \
                          (1-settings['region_distance_balance']) * spatial_distance

                out = np.concatenate((shadow.showable_image(), light.showable_image()), axis=1)

                shadow_centroid = shadow.centroid
                cv2.circle(out, shadow_centroid, 10, (255,0,255))
                light_centroid = (shadow.image.shape[1] + light.centroid[0], light.centroid[1])
                cv2.circle(out, light_centroid, 10, (255,0,255))

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(out, "sh-color:"+str(sh_avg), (30,200), font, 1,(255,255,255))
                cv2.putText(out, "light-color:"+str(light_region_mean), (30,250), font, 1,(255,255,255))
                cv2.putText(out, "color-dist:"+repr(color_distance), (30,300), font, 1,(255,255,255))
                cv2.putText(out, "space-dist:"+repr(spatial_distance), (30,350), font, 1,(255,255,255))
                cv2.putText(out, "total-dist:"+repr(new_dis), (30,400), font, 1,(255,255,255))
                printer(i, j, out)
                j += 1