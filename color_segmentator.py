import cv2
import numpy as np
# import matplotlib.pyplot as plt

from skimage.segmentation import felzenszwalb, slic, quickshift, mark_boundaries
from LAB.shadow_detection.utils import equalize_hist_3d


class ColorSegmentator(object):
    def __init__(self, settings):
        self.settings = settings

    @staticmethod
    def smooth(x,window_len=11,window='hanning'):
        radius=int(window_len/2)
        res = list(x[:radius-1])

        r2 = [int(sum(x[i-radius:i+radius])/(2*radius+1)) for i in range(radius, len(x-radius-1))]
        res += r2
        res += list(x[len(x)-radius+1:])
        return np.array(res)

    def peaks(self, a):
        min_diff = cv2.sumElems(a)[0] / (100 * self.settings['segementation_detail'])

        bools = np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True]
        pks = [i for i in range(len(bools)) if bools[i]]
        if not pks[0] == 0:
            pks.insert(0,0)
        if not pks[-1] == 255:
            pks.append(255)

        out = [0]
        for i in range(1,len(pks)-1):
            lw = pks[i-1]
            ac = pks[i]
            hr = pks[i+1]
            mh = (ac + hr) / 2
            ml = (lw + ac) / 2
            if a[ac] - a[mh] > min_diff and a[ac] - a[ml] > min_diff:
                out.append(ac)
        out.append(255)
        return out

    @staticmethod
    def plot_histogram(hist, max=None, img=None, color=(0,0,0)):
        margin = 20
        if img is None:
            img = np.ones((256*3, 600+2*margin, 3)) * 255
        pts = [int(val*600/max) for val in hist]

        pts = [[3*i, 600 + margin - pts[i]] for i in range(len(pts))]
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        img = cv2.polylines(img, [pts], False, color)
        return img

    @staticmethod
    def plot_peak(peaks, img):
        for peak in peaks:
            img = cv2.line(img, (peak*3, 0), (peak*3, 600),(0,255,0))
        return img

    @staticmethod
    def apply_mask(image, mask):
        return cv2.bitwise_and(image, cv2.merge([mask, mask, mask]))

    @staticmethod
    def generate_new_threshhold_mask(image, val):
        image = np.uint8(image)
        retval, mask_le_high = cv2.threshold(image, val, 255, cv2.THRESH_BINARY_INV)
        retval, mask_gt_min = cv2.threshold(image, val-1, 255, cv2.THRESH_BINARY)
        # erase mask values

        return cv2.bitwise_and(mask_gt_min, mask_le_high)

    @staticmethod
    def generate_threshhold_mask(image, minval, maxval):
        image = np.uint8(image)
        retval, mask_ge_min = cv2.threshold(image, minval-1, 255, cv2.THRESH_BINARY)
        retval, mask_lw_max = cv2.threshold(image, maxval-1, 255, cv2.THRESH_BINARY_INV)
        # erase mask values

        return cv2.bitwise_and(mask_lw_max, mask_ge_min)

    def get_region_masks(self, shadow_mask):
        mask = shadow_mask.copy()
        image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        big_regions = []

        min_size = shadow_mask.shape[0] * shadow_mask.shape[1] / self.settings['min_size_factor']

        for i in range(len(contours)):
            blank = np.zeros((shadow_mask.shape[0], shadow_mask.shape[1], 1), np.uint8)

            region_mask = cv2.drawContours(blank, contours, i, 255, -1)

            s = cv2.sumElems(region_mask/255)[0]
            if s > min_size:
                big_regions.append(region_mask)

        return big_regions

    def segment_image(self, img=cv2.imread('img/road6.png'), show=False):
        return self.segment_image_new(img, show)

    def segment_image_new(self, img=cv2.imread('img/road6.png'), show=False):
        #min_size = img.shape[0] * img.shape[1] / self.settings['min_size_factor']

        n_segments = self.settings['n_segments']
        compactness = self.settings['compactness']
        sigma = self.settings['sigma']

        #segments = felzenszwalb(img, scale=5000, sigma=0.5, min_size=300)
        # segments = slic(img,
        #                 n_segments=n_segments,
        #                 compactness=compactness,
        #                 sigma=sigma,
        #                 convert2lab=True)

        segments = slic(img,
                        n_segments=20,
                        compactness=0.3,
                        sigma=sigma,
                        convert2lab=True,
                        min_size_factor=0.05,
                        max_size_factor=0.8)

        #segments = quickshift(img, ratio=0.8, max_dist=80)

        contours = []
        for i in range(segments.min(), segments.max()+1):
            mask = self.generate_new_threshhold_mask(segments, i)
            contours.append(mask)

        #m = mark_boundaries(black, segments)

        return contours

    def segment_image_old(self, img=cv2.imread('img/road6.png'), show=False):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_img], [0], None, [256], [0,256]).reshape([256])

        m = max(hist)
        hist_img = self.plot_histogram(hist, m)
        smooth_msft_hist = self.smooth(hist, self.settings['seg_hist_soften'])
        hist_img = self.plot_histogram(smooth_msft_hist, m, hist_img, (255,0,0))

        hist_peaks = self.peaks(smooth_msft_hist)
        hist_img = self.plot_peak(hist_peaks, hist_img)
        cv2.imwrite('hist.png', hist_img)
        if show:
            print(hist_peaks)
            cv2.imshow('img', img)
        h = cv2.split(hsv_img)[0]

        kernel_1 = np.ones((2,2), np.uint8)
        kernel_2 = np.ones(self.settings['dil_erod_kernel_size_segmentator'], np.uint8)

        result = []
        accum = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        for i in range(len(hist_peaks)-1):
            mask = self.generate_threshhold_mask(h, hist_peaks[i], hist_peaks[i+1])
            mask = cv2.dilate(mask, kernel_1, iterations=2)
            mask = cv2.erode(mask, kernel_1, iterations=2)

            if self.settings['dil_erod_kernel_size_segmentator'] > 0:
                mask = cv2.erode(mask, kernel_2, iterations=1)
                mask = cv2.dilate(mask, kernel_2, iterations=1)

            mask = cv2.bitwise_and(mask, 255 - accum)

            #cv2.imwrite('dbg_img/color_F_mask_'+str((hist_peaks[i], hist_peaks[i+1]))+'.png', mask)

            shape = mask.shape
            shape = [shape[0], shape[1]]
            submasks = self.get_region_masks(mask)

            for submask in submasks:
                submask = submask.reshape(shape)
                accum += submask
                result.append(submask)

            if show:
                j = 0
                for submask in submasks:
                    cv2.imshow('mask_'+str(i)+"_"+str(j), self.apply_mask(img,submask))
                    j += 1

        # plt.show()

        return result