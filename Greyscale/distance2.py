# Dilate/Erode mask
import random
import cv2
import numpy as np
import math
from Greyscale.colorspaces import GrayscaleColorSpace
from boudary_drawer import draw_boundaries, draw_region
from color_segmentator import ColorSegmentator
from settings import settings
from LAB.shadow_detection.utils import equalize_hist_3d
from scipy.spatial import distance as dist


class Region(object):
    def __init__(self, image, mask, colorspace=None):
        #
        # Calculate means over rgb image for mahalanobis
        #
        self.image = image
        self.mask = mask.copy()
        # self.values = self.calculate_region_values()
        # self.means_rgb = self.calculate_means()
        # try:  #todo: creo q no hace falta
        #     self.means_rgb_lab = colorspace.pre_process_image(
        #         np.uint8(np.array(self.means_rgb).reshape([1, 1, colorspace.channels_count()])))
        # except:
        #     pass
        # self.means_rgb_lab = self.means_rgb_lab[0][0]
        #--------------------------
        if colorspace:
            self.image = colorspace.pre_process_image(image)
        else:
            self.image = image.copy()

        self.colorspace = colorspace
        self.image = self.apply_mask(self.image, mask)
        self.mask = mask.copy()
        self.values = self.calculate_region_values()
        self.means = self.calculate_means()
        self.variances = self.calculate_standard_deviation()
        self.covariance = self.calculate_covariance_matrix()
        self.centroid = self.get_centroid()
        self.diagonal = pow(pow(image.shape[0], 2) + pow(image.shape[1], 2), 1 / 2.0)
        self.hists = self.calculate_hists()

    def calculate_hists(self):
        hists = []
        for ch in self.values:
            ch = np.uint8(ch)
            total = len(ch)
            try:
                hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).reshape([256])
            except:
                pass
            hist = hist * 100 / total
            hists.append(hist)
        return hists

    def calculate_means(self):
        means = []
        for value in self.values:
            means.append(np.average(value))
        return means

    def calculate_standard_deviation(self):
        variances = []
        for i in range(len(self.values)):
            value = self.values[i]
            mean = np.average(value)
            N = len(value)
            if N <= 1:
                variances.append(0)
            else:
                v2 = (value - mean)
                var = cv2.sumElems(np.array(v2) * np.array(v2))[0] / N
                variances.append(math.sqrt(var))
        return variances

    def calculate_covariance_matrix(self):
        v = np.array([self.values[i] for i in self.colorspace.color_indices()])
        return np.cov(v)

    def set_minimum_cov(self, cov, minimums):
        retval, eigenvalues, eigenvectors = cv2.eigen(cov)
        for i in xrange(3):
            if eigenvalues[i] < (minimums(i) * (360 * 50 if i == 0 else 0.25)):
                eigenvalues[i] = minimums(i) * (360.0 * 50 if i == 0 else 0.25)
        cov = eigenvectors * np.diag(eigenvalues) * eigenvectors.t()
        return cov

    def calculate_region_values(self):
        N = cv2.sumElems(self.mask)[0]/255
        if N > 0:
            ch = cv2.split(self.image)
            values = [[] for i in range(len(ch))]
            region_indices = np.where(self.mask)
            # values2 = [[] for i in range(len(ch))]
            # for i in range(len(region_indices[0])):
            #     for j in range(len(ch)):
            #         values2[j].append(ch[j][region_indices[0][i]][region_indices[1][i]])
            for j in range(len(ch)):
                values[j] = ch[j][region_indices]
            return np.array(values)
        else:
            no_channels = self.image.shape[2] if len(self.image.shape) > 2 else 1
            return [np.array([])]*no_channels

    @staticmethod
    def apply_mask(image, mask):
        if len(mask.shape) > 2 or (len(image.shape) == len(mask.shape)):
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

        if M['m00'] == 0:  # if region is empty
            return (self.mask.shape[0] / 2,
                    self.mask.shape[1] / 2)

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)

    def spatial_distance(self, other_region):
        dis = pow(
            pow(self.centroid[0] - other_region.centroid[0], 2) + pow(self.centroid[1] - other_region.centroid[1], 2),
            0.5)
        return dis / self.diagonal

    def color_distance(self, other_region):

        if not self.colorspace == other_region.colorspace:
            raise Exception

        if self.colorspace.is_not_black_region(other_region.image):
            sq_diff = 0
            for index in self.colorspace.color_indices():
                sq_diff += math.pow(self.means[index] - other_region.means[index], 2)
            dist = math.sqrt(sq_diff)
            #return dist * 10 / self.colorspace.value_range(self.image)
            return dist / 10
        else:
            return -1

    def means_relations(self):
        color_indices = self.colorspace.color_indices()
        relations = []
        for i in range(len(color_indices)-1):
            index_a = color_indices[i]
            for j in range(i+1, len(color_indices)):
                index_b = color_indices[j]
                relations.append(self.means[index_a]-self.means[index_b])
        return relations

    def means_rel_distance(self, other_region):

        if not self.colorspace == other_region.colorspace:
            raise Exception
        mrel_a = self.means_relations()
        mrel_b = other_region.means_relations()
        if self.colorspace.is_not_black_region(other_region.image) and len(mrel_a) == len(mrel_b):
            sq_diff = 0
            for index in range(len(mrel_a)):
                sq_diff += math.pow(mrel_a[index] - mrel_b[index], 2)
            dist = math.sqrt(sq_diff)
            return dist / 10
        else:
            return -1

    def variance_distance(self, other_region):
        if not self.colorspace == other_region.colorspace:
            raise Exception

        if self.colorspace.is_not_black_region(other_region.image):

            idxs = self.colorspace.color_indices()
            vars1 =np.array([self.variances[i] for i in idxs])
            vars2 =np.array([other_region.variances[i] for i in idxs])
            diff2 = vars1 - vars2
            dist = math.sqrt(cv2.sumElems(diff2*diff2)[0])
            return dist / 10
        else:
            return -1

    def bhatttacharyya_distance(self, other_region):
        if not self.colorspace == other_region.colorspace:
            raise Exception

        if self.colorspace.is_not_black_region(other_region.image):
            idxs = self.colorspace.color_indices()
            means1 = np.array([self.means[i] for i in idxs])
            means2 = np.array([other_region.means[i] for i in idxs])
            means_diff = means1 - means2
            cov_mat_sum = (self.covariance + other_region.covariance) / 2
            inv_cov_mat_sum = cv2.invert(cov_mat_sum, cv2.DECOMP_SVD)

            first_term = (np.matrix(means_diff) * inv_cov_mat_sum[1] * np.matrix(means_diff).transpose()) / 8
            d1 = np.linalg.det(self.covariance)
            d1 = d1 if d1 != 0 else 0.0000000000000000000000000000001
            d2 = np.linalg.det(other_region.covariance)
            d2 = d2 if d2 != 0 else 0.0000000000000000000000000000001
            ds = np.linalg.det(cov_mat_sum)
            ds = ds if ds != 0 else 0.0000000000000000000000000000001
            second_term = (math.log(ds/math.sqrt(d1*d2))) / 2
            dist = np.array(first_term)[0][0] + second_term
            return dist
        else:
            return -1

    def hist_dist(self, other_region):
        total = 0
        for j in self.colorspace.color_indices():
            #---------------------------------
            score = 0
            hist1 = np.float32(self.hists[j])
            hist2 = np.float32(other_region.hists[j])
            total += cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
            #total += dist.chebyshev(self.hists[j], other_region.hists[j])
            #---------------------------------
        return total

    def balanced_distance(self, other_region, region_distance_balance):
        if self.colorspace.channels_count() > 1:
            # return self.hist_dist(other_region)  # todo: remove
            return self.bhatttacharyya_distance(other_region)
        else:
            color_distance = self.color_distance(other_region)
            if color_distance >= 0:
                variance_distance = self.variance_distance(other_region)
                means_rel_distance = self.means_rel_distance(other_region)
                return (region_distance_balance * color_distance + \
                       region_distance_balance * means_rel_distance + \
                       (1 - region_distance_balance) * variance_distance)
            else:
                return -1

    def showable_image(self):
        return self.colorspace.post_process_image(self.image)

    def plot_histograms(self, name='hists.png'):
        img = None
        m = max(self.hists[0])
        colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i in range(len(self.hists)):
            hist = self.hists[i]
            color = colors[i]
            img = ColorSegmentator.plot_histogram(hist, m, img, color)
        return img


class DistanceFinder(object):
    def __init__(self, image, dilated_shadow_mask, colorspace, road_estimated_mask, settings=settings):
        self.image = image
        self.colorspace = colorspace
        self.shadow_regions = []
        self.light_regions = []
        self.dilated_shadows_mask = dilated_shadow_mask
        self.settings = settings
        print "Segmenting image..."
        self.color_region_masks = \
            ColorSegmentator(self.settings).segment_image(self.image)

        self.road_estimated_mask = road_estimated_mask

        self.matches = {}
        self.initialize_regions()

        # to standarize spatial distance
        self.diagonal = pow(pow(self.image.shape[0], 2) + pow(self.image.shape[1], 2), 1 / 2.0)

        pass

    def has_shadows(self):
        return len(self.shadow_regions) > 0

    def initialize_regions(self):
        print "Generating shadow regions..."
        self.shadow_regions = self.generate_regions(self.dilated_shadows_mask,
                                                    self.colorspace)

        light_mask = 255 - self.dilated_shadows_mask
        print "Generating light regions..."
        self.light_regions = self.generate_regions(light_mask, self.colorspace)
        # mono image regions initialization

        self.mono_shadow_regions = []
        self.mono_light_regions = []
        print "Finding matches..."
        self.find_matches()

    def find_matches(self):
        min_dist = -1

        best_match = (None, None)
        for shadow_region in self.shadow_regions:
            matching_region, dist = self.get_closest_region(shadow_region)
            if dist < self.settings['max_color_dist']:
                self.matches[shadow_region] = matching_region
            else:
                self.matches[shadow_region] = None
            if min_dist == -1 or dist < min_dist:
                min_dist = dist
                best_match = (shadow_region, matching_region)
        if min_dist >= self.settings['max_color_dist']:
            self.matches[best_match[0]] = best_match[1]

    def run(self, mono_image):
        if self.matches == {}:
            self.find_matches()
        mono_image = np.float64(equalize_hist_3d(mono_image))
        mono_light_regions = {}
        colorspace = GrayscaleColorSpace()
        for light_region in self.light_regions:
            mono_light_regions[light_region] = Region(mono_image,
                                                      np.float64(light_region.mask),
                                                      colorspace)

        mono_shadow_regions = {}
        for shadow_region in self.shadow_regions:
            mono_shadow_regions[shadow_region] = Region(mono_image,
                                                        np.float64(shadow_region.mask),
                                                        colorspace)

        distance = -1
        any_light_region = False
        for shadow_region in self.shadow_regions:
            mono_shadow = mono_shadow_regions[shadow_region]

            light_region = self.matches[shadow_region]

            if light_region:
                any_light_region = True
                mono_light = mono_light_regions[light_region]
                d = mono_shadow.color_distance(mono_light)
                if distance == -1 or (d >= 0 and d < distance):
                    distance = d
        return distance if any_light_region else -1

    def apply_mask(self, image, mask):
        return cv2.bitwise_and(image, cv2.merge([mask, mask, mask]))

    def sanitize_mask(self, dilated_mask, original_mask):
        return cv2.bitwise_and(dilated_mask, original_mask)

    def generate_regions(self, main_mask, colorspace):
        mask = main_mask.copy()
        image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        big_regions = []
        min_size = main_mask.shape[0] * main_mask.shape[1] / self.settings['min_size_factor']

        for i in range(len(contours)):
            blank = np.zeros((main_mask.shape[0], main_mask.shape[1], 1), np.uint8)
            region_mask = cv2.drawContours(blank, contours, i, 255, -1)
            #clean the mask because detects light and shadow contours
            region_mask = self.apply_multi_mask(region_mask, main_mask)
            #avoid unnecesary processing
            s = cv2.sumElems(region_mask / 255)[0]
            if s > min_size:
                for color_mask in self.color_region_masks:
                    subregions = self.apply_multi_mask(region_mask, color_mask)
                    #avoid unnecesary processing
                    s = cv2.sumElems(subregions / 255)[0]
                    if s > min_size:
                        subimage, subcontours, subhierarchy = cv2.findContours(subregions, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        for j in range(len(subcontours)):
                            blank = np.zeros((main_mask.shape[0], main_mask.shape[1]), np.uint8)
                            subregion_mask = cv2.drawContours(blank, subcontours, j, 255, -1)
                            #avoid unnecesary processing
                            s = cv2.sumElems(subregion_mask / 255)[0]
                            subregion_in_path_mask = self.apply_multi_mask(subregion_mask, self.road_estimated_mask)
                            sp = cv2.sumElems(subregion_in_path_mask / 255)[0]
                            if s > min_size and sp > (0.6 * s):
                                region = Region(self.image, subregion_mask, colorspace)
                                big_regions.append(region)

        return big_regions

    def apply_multi_mask(self, image, mask):
        return cv2.bitwise_and(image, mask)

    def get_closest_region(self, shadow_region):
        closest_region = None
        distance = 99999999999999999999999999999999999999999999999999999999999999999999999999
        for light_region in self.light_regions:
            new_distance = \
                shadow_region.balanced_distance(light_region,
                                                self.settings['region_distance_balance'])
            if 0 <= new_distance < distance:
                distance = new_distance
                closest_region = light_region
        return closest_region, distance

    def print_region_matches(self, printer):
        i = 0
        for shadow in self.shadow_regions:
            light = self.matches[shadow]
            if light >= 0:
                out = np.concatenate((shadow.showable_image(),
                                      light.showable_image()), axis=1)
                printer(i, out)
                i += 1

    def region_matches_image(self):
        out = self.image.copy()

        for shadow in self.shadow_regions:
            light = self.matches[shadow]
            color = [random.randint(0, 255) for _ in xrange(3)]
            out = draw_boundaries(out, shadow.mask, color)
            if light is not None:
                radius = 4
                thickness = 2
                cv2.circle(out, shadow.get_centroid(),
                           radius, color, thickness)
                out = draw_boundaries(out, light.mask)
                cv2.circle(out, light.get_centroid(),
                           radius, color, thickness)
                cv2.line(out, shadow.get_centroid(),
                         light.get_centroid(), color, thickness)
                font = cv2.FONT_HERSHEY_SIMPLEX
                distance = shadow.balanced_distance(light,
                                                    self.settings['region_distance_balance'])
                displaced_centroid = (shadow.get_centroid()[0] + 15,
                                      shadow.get_centroid()[1])
                cv2.putText(out, str(distance)[:5],
                            displaced_centroid, font, 0.5, color)

        return out

    def distance_image(self, region_a=None, region_b=None):
        out = self.image.copy()
        white = [0, 0, 0]
        for shadow in self.shadow_regions:
            out = draw_boundaries(out, shadow.mask, white)
        for light in self.light_regions:
            out = draw_boundaries(out, light.mask, white)
        if region_a is not None:
            color = [random.randint(0, 255) for _ in xrange(3)]
            out = draw_boundaries(out, region_a.mask, color)
            if region_b is not None:
                radius = 4
                thickness = 2
                cv2.circle(out, region_a.get_centroid(),
                           radius, color, thickness)
                out = draw_boundaries(out, region_b.mask)
                cv2.circle(out, region_b.get_centroid(),
                           radius, color, thickness)
                cv2.line(out, region_a.get_centroid(),
                         region_b.get_centroid(), color, thickness)
                font = cv2.FONT_HERSHEY_SIMPLEX
                distance = region_a.balanced_distance(region_b,
                                                      self.settings['region_distance_balance'])
                displaced_centroid = (region_a.get_centroid()[0] + 15,
                                      region_a.get_centroid()[1])
                cv2.putText(out, str(distance)[:5],
                            displaced_centroid, font, 0.5, color)
                print distance

        return out

    def region_for(self, point):
        for shadow in self.shadow_regions:
            if shadow.mask[point[1]][point[0]] > 0:
                return shadow
        for light in self.light_regions:
            if light.mask[point[1]][point[0]] > 0:
                return light

    def mono_distance_image(self, image):
        out = equalize_hist_3d(image)
        # out = np.float64(equalize_hist_3d(image))

        # ------------

        mono_light_regions = {}
        colorspace = GrayscaleColorSpace()
        for light_region in self.light_regions:
            mono_light_regions[light_region] = Region(out,
                                                      light_region.mask,
                                                      colorspace)

        mono_shadow_regions = {}
        for shadow_region in self.shadow_regions:
            mono_shadow_regions[shadow_region] = Region(out,
                                                        shadow_region.mask,
                                                        colorspace)
        for shadow_region in self.shadow_regions:
            mono_shadow = mono_shadow_regions[shadow_region]
            color = [random.randint(0, 255)]
            out = draw_boundaries(out, shadow_region.mask, color)

            light_region = self.matches[shadow_region]

            if light_region is not None:
                mono_light = mono_light_regions[light_region]
                radius = 4
                thickness = 2
                cv2.circle(out, shadow_region.get_centroid(),
                           radius, color, thickness)
                out = draw_boundaries(out, light_region.mask, color)
                cv2.circle(out, light_region.get_centroid(),
                           radius, color, thickness)
                cv2.line(out, shadow_region.get_centroid(),
                         light_region.get_centroid(), color, thickness)
                font = cv2.FONT_HERSHEY_SIMPLEX
                distance = mono_shadow.color_distance(mono_light)
                displaced_centroid = (shadow_region.get_centroid()[0] + 15,
                                      shadow_region.get_centroid()[1])
                cv2.putText(out, str(distance)[:5],
                            displaced_centroid, font, 0.5, color)
        return out

    def segmentation_image(self):

        out = np.zeros((self.image.shape[0],
                        self.image.shape[1], 3), np.uint8)

        for shadow in self.shadow_regions:
            #color = [random.randint(0, 255) for _ in xrange(3)]
            out = draw_region(out, shadow.mask)

        for light in self.light_regions:
            out = draw_region(out, light.mask)

        return out

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
                new_dis = \
                    shadow.balanced_distance(light,
                                             self.settings['region_distance_balance'])

                out = np.concatenate((shadow.showable_image(), light.showable_image()), axis=1)

                shadow_centroid = shadow.centroid
                cv2.circle(out, shadow_centroid, 10, (255, 0, 255))
                light_centroid = (shadow.image.shape[1] + light.centroid[0], light.centroid[1])
                cv2.circle(out, light_centroid, 10, (255, 0, 255))

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(out, "sh-color:" + str(sh_avg), (30, 200), font, 1, (255, 255, 255))
                cv2.putText(out, "light-color:" + str(light_region_mean), (30, 250), font, 1, (255, 255, 255))
                cv2.putText(out, "color-dist:" + repr(color_distance), (30, 300), font, 1, (255, 255, 255))
                cv2.putText(out, "space-dist:" + repr(spatial_distance), (30, 350), font, 1, (255, 255, 255))
                cv2.putText(out, "total-dist:" + repr(new_dis), (30, 400), font, 1, (255, 255, 255))
                printer(i, j, out)
                j += 1
