import cv2
import numpy as np
import math


def equalize_hist_3d(img):
    max_val = img.max()
    min_val = img.min()
    factor = 255.0 / (max_val - min_val)
    eq_img = (img - min_val) * factor
    return np.uint8(eq_img)


def load_image(name, ext):
    img = cv2.imread('img/' + name + "." + ext)
    show_image('Original image', img)
    return img


def load_L1CI_image(name, ext, color_mode=cv2.IMREAD_UNCHANGED):
    img = cv2.imread('img/out/' + name + '/out_L1_Chromacity_invariant.' + ext, color_mode)
    return img


def show_image(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)


def show_and_save(param, name, ext, img, fact=1):
    cv2.namedWindow(param, cv2.WINDOW_NORMAL)
    cv2.imshow(param, img)
    cv2.imwrite('img/out/' + name + '/out_' + param + '.' + ext, img * fact)


def mono_image(width, height):
    return [[np.int64(0) for j in xrange(width)] for j in xrange(height)]


def adapt_to_range(min_mono):
    # search for max min values
    mono_img = np.array(min_mono)
    min_val = mono_img.min()
    max_val = mono_img.max()

    print "\n>>>Invariant image<<<"
    print "Min val: " + str(min_val)
    print "Max val: " + str(max_val)

    rango = max_val - min_val
    coef = 1 / rango

    print "Range: " + str(rango)
    print "Coefficient: " + str(coef)

    # ADAPT RANGE TO 0..1----------------
    min_mono2 = np.array([[coef * (elem - min_val) for elem in row] for row in min_mono])
    return min_mono2


def entropy(matrix):
    mono_img = np.array(matrix)
    min_val = mono_img.min()
    max_val = mono_img.max()
    rango = max_val - min_val
    coef = 1 / rango

    bins = 64
    sums = [0] * bins

    for row in matrix:
        for val in row:
            normal_val = coef * (val - min_val)
            bin = int(normal_val * bins)
            if bin == 64:
                bin = 63
            sums[bin] += 1
    ent = 0
    for bin in xrange(bins):
        val = sums[bin]
        try:
            ent += val * math.log(val)
        except ValueError:
            ent += val * math.log(0.0001)
    return -ent


def get_extralight(A, I, e):
    a_a = np.reshape(A, -1)
    one_percent = len(a_a) / 100
    brightest_indexes = a_a.argsort()[-one_percent:][::-1]
    brightest = [a_a[index] for index in brightest_indexes]
    A_median = np.median(brightest)
    i_i = np.reshape(I, -1)
    originals = [i_i[index] for index in brightest_indexes]
    I_median = np.median(originals)
    return np.matrix((I_median - A_median) * e).transpose()
