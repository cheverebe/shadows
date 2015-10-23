# Dilate/Erode mask
import cv2
import numpy as np


class Step5(object):
    def __init__(self):
        pass

    def run(self, image, shadow_mask):
        shadows = np.zeros((shadow_mask.shape[0], shadow_mask.shape[1], 3), np.uint8)

        b_region_masks, s_region_masks = self.get_region_masks(shadow_mask)

        #recompute the shadow mask using all the region masks to avoid troubles with edges
        shadow_mask = np.zeros((shadow_mask.shape[0], shadow_mask.shape[1], 1), np.uint8)

        for region_mask in s_region_masks:
            region = self.apply_mask(image, region_mask)
            shadows += region
            shadow_mask += region_mask

        for region_mask in b_region_masks:
            shadow_mask += region_mask

        light_mask = 255 - shadow_mask

        lights = self.apply_mask(image, light_mask)

        for region_mask in b_region_masks:
            region = self.apply_mask(image, region_mask)
            coef = self.get_coeficient(light_mask, lights, region_mask, region)
            print coef
            region *= coef
            shadows += region

        return shadows + lights

    def apply_mask(self, image, mask):
        return cv2.bitwise_and(image, cv2.merge([mask, mask, mask]))

    def get_coeficient(self, light_mask, lights, shadow_mask, shadows):
        l_avg = sum([sm / cv2.sumElems(light_mask/255)[0] for sm in cv2.sumElems(lights)])/3
        s_avg = sum([sm / cv2.sumElems(shadow_mask/255)[0] for sm in cv2.sumElems(shadows)])/3
        return l_avg / s_avg

    def get_region_masks(self, shadow_mask):
        mask = shadow_mask.copy()
        image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        big_regions = []
        small_regions = []
        print len(contours)
        for i in range(len(contours)):
            blank = np.zeros((shadow_mask.shape[0], shadow_mask.shape[1], 1), np.uint8)
            img = cv2.drawContours(blank, contours, i, 255, -1)
            if len(contours[i]) > 30:
                big_regions.append(img)
            else:
                small_regions.append(img)
        return big_regions, small_regions