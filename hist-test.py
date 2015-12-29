import cv2
import numpy as np
#import matplotlib.pyplot as plt
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

def generate_threshhold_mask(image, minval, maxval):
    image = np.uint8(image)
    retval, mask_lw_high = cv2.threshold(image, maxval, 255, cv2.THRESH_BINARY)
    mask_lw_high = 255 - mask_lw_high
    retval, mask_hg_min = cv2.threshold(image, minval, 255, cv2.THRESH_BINARY_INV)
    mask_hg_min = 255 - mask_hg_min
    # erase mask values

    return cv2.bitwise_and(mask_hg_min, mask_lw_high)

img = cv2.imread('img/ale1.png')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv_img], [0], None, [256], [0,256])

#plt.plot(hist, color='k')
#plt.xlim([0,256])

spatial_radius = 50
color_radius = 50

smooth_msft_hist = smooth(hist.reshape([256]), 20)
#plt.plot(smooth_msft_hist, color='g')

hist_peaks = peaks(smooth_msft_hist)
print(hist_peaks)

cv2.imshow('img', img)
h = cv2.split(hsv_img)[0]
for i in range(len(hist_peaks)-1):
    mask = generate_threshhold_mask(h, hist_peaks[i], hist_peaks[i+1])
    cv2.imshow('mask_'+str(i), mask)

# plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()