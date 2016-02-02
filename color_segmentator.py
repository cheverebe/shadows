import cv2
import numpy as np
# import matplotlib.pyplot as plt


class ColorSegmentator(object):
    def __init__(self, settings):
        self.settings = settings

    @staticmethod
    def smooth(x,window_len=11,window='hanning'):
        if x.ndim != 1:
            raise ValueError, "smooth only accepts 1 dimension arrays."

        if x.size < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."

        if window_len<3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

        s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        return y

    @staticmethod
    def peaks(a):
        min_diff = 500

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
    def apply_mask(image, mask):
        return cv2.bitwise_and(image, cv2.merge([mask, mask, mask]))

    @staticmethod
    def generate_threshhold_mask(image, minval, maxval):
        image = np.uint8(image)
        retval, mask_lw_high = cv2.threshold(image, maxval+1, 255, cv2.THRESH_BINARY)
        mask_lw_high = 255 - mask_lw_high
        retval, mask_hg_min = cv2.threshold(image, minval-1, 255, cv2.THRESH_BINARY_INV)
        mask_hg_min = 255 - mask_hg_min
        # erase mask values

        return cv2.bitwise_and(mask_hg_min, mask_lw_high)

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
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_img], [0], None, [256], [0,256])

        # plt.plot(hist, color='k')
        # plt.xlim([0,256])

        smooth_msft_hist = self.smooth(hist.reshape([256]), 20)
        # plt.plot(smooth_msft_hist, color='g')

        hist_peaks = self.peaks(smooth_msft_hist)
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
            accum += mask
            mask = cv2.dilate(mask, kernel_1, iterations=3)
            mask = cv2.erode(mask, kernel_2, iterations=3)
            mask = cv2.dilate(mask, kernel_2, iterations=3)

            cv2.imwrite('dbg_img/color_mask_'+str((hist_peaks[i], hist_peaks[i+1]))+'.png', mask)

            shape = mask.shape
            shape = [shape[0], shape[1]]
            submasks = self.get_region_masks(mask)
            result += [submask.reshape(shape) for submask in submasks]

            if show:
                j = 0
                for submask in submasks:
                    cv2.imshow('mask_'+str(i)+"_"+str(j), self.apply_mask(img,submask))
                    j += 1
        cv2.imwrite('dbg_img/accum.png', accum)

        # plt.show()

        return result