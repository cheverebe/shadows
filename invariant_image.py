import numpy as np
import math
import cv2
from Greyscale.InvariantImageGenerator import InvariantImageGenerator
from Greyscale.distancefinder import DistanceFinder
from Greyscale.colorspaces import LABColorSpace
from Greyscale.distance2 import DistanceFinder as DistanceFinder2
from LAB.shadow_detection.pipeline import ShadowDetectionPipeline
from LAB.shadow_detection.utils import show_and_save
from settings import settings
from utils import entropy, get_extralight, equalize_hist_3d


def get_invariant_l1_chrom_image(two_dim, min_angle):
    U = get_U()
    rad = math.radians(min_angle)
    e_orth = np.array([math.cos(rad), math.sin(rad)])
    e_orth_m = np.matrix(e_orth)
    e_orth_norm = cv2.norm(e_orth)
    P_e_orth = np.matrix(e_orth_m.transpose() * e_orth_m / e_orth_norm)
    td = np.array(two_dim)
    X_tita = np.array([[P_e_orth * np.matrix(elem) for elem in row] for row in two_dim])
    #X_tita = np.array(P_e_orth) * td
    extralight = get_extralight(X_tita, np.array(two_dim), e_orth)
    X_tita = np.array([[np.matrix(elem) + extralight for elem in row] for row in X_tita])
    p_monio = np.array([[(U.transpose() * np.matrix(elem)) for elem in row] for row in X_tita])
    c_monio = np.array([[np.array([math.exp(val) for val in elem]) for elem in row] for row in p_monio])
    (log_chrom, L1chrom) = log_chromaticity_image(c_monio*255)
    L1chrom = np.array(L1chrom)
    return L1chrom

def get_U():
    return np.matrix([[1.0/math.sqrt(2), -1.0/math.sqrt(2), 0], [1.0/math.sqrt(6), 1.0/math.sqrt(6), -2.0/math.sqrt(6)]])

def project_to_2d(log_chrom):
    iig = InvariantImageGenerator()
    return iig.project_to_2d(log_chrom)

def log_chromaticity_image(img):
    iig = InvariantImageGenerator()
    return iig.log_chrom_image(img)

def plot_entropies(angles, ent_list):
    # plt.rcParams['axes.unicode_minus'] = False
    # fig, ax = plt.subplots()
    # ax.plot(angles, ent_list, '.')
    # ax.set_title('Entropy')
    # plt.show()
    #
    # plt.savefig('img/out/plot.png')
    print angles
    print ent_list

def find_invariant_image(original, two_dim, name):
    pip = ShadowDetectionPipeline()
    dilated_shadow_mask, shadow_mask = pip.find_dilated_shadow_mask(original)
    dist_finder = DistanceFinder2(original, dilated_shadow_mask, LABColorSpace())

    printer = lambda index, image: show_and_save('match('+str(index)+')', 'out/'+name, 'png', image)
    dist_finder.print_region_matches(printer)
    printer = lambda index, image: show_and_save('shadow('+str(index)+')', 'out/'+name, 'png', image)
    dist_finder.print_shadow_regions(printer)
    printer = lambda index, image: show_and_save('light('+str(index)+')', 'out/'+name, 'png', image)
    dist_finder.print_light_regions(printer)
    printer = lambda index_s, index_l, image: show_and_save('distance'+str((index_s, index_l)), 'dbg_img/'+name, 'png', image)
    dist_finder.print_region_distances(printer)

    cv2.namedWindow("shadow_mask", cv2.WINDOW_NORMAL)
    cv2.imshow('shadow_mask', shadow_mask)
    iig = InvariantImageGenerator()
    # FIND MIN ANGLE
    min_entropy = 0
    min_mono = []
    entropy_array = []

    #----AUX DATA
    light_mask = 255 - shadow_mask
    shadow_pixels_count = cv2.sumElems(shadow_mask/255)[0]
    light_pixels_count = cv2.sumElems(light_mask/255)[0]

    angles = xrange(0, 180)
    min_angle = 0   #hack horriible
    min_distance = -1


    ent_list = []
    for angle in angles:
        mono = iig.project_into_one_d(two_dim, angle)

        #light_pixels = cv2.bitwise_and(np.float64(mono), np.float64(light_mask))
        #light_mean = cv2.sumElems(light_pixels)[0]/light_pixels_count

        #shadow_pixels = cv2.bitwise_and(np.float64(mono), np.float64(shadow_mask))
        #shadow_mean = cv2.sumElems(shadow_pixels)[0]/shadow_pixels_count

        #distance = abs(light_mean - shadow_mean)

        distance = dist_finder.run(np.float64(mono))

        print str("%d, %s" % (angle, repr(distance)))
        if min_distance == -1 or distance < min_distance:
            min_distance = distance
            min_mono = mono
            min_angle = angle

    return min_mono, min_angle


def minimize_entropy(two_dim, angle=None, name=None):
    iig = InvariantImageGenerator()
    # FIND MIN ANGLE
    min_entropy = 0
    min_mono = []
    entropy_array = []

    if angle is None:
        angles = xrange(1, 180)
        min_angle = 0   #hack horriible
    else:
        angles = [angle]
        min_angle = angle

    ent_list = []
    for angle in angles:
        print str(angle)
        rad = math.radians(angle)
        mono = iig.project_into_one_d(two_dim, angle)
        ent = entropy(mono)
        ent_list.append(ent)
        entropy_array.append(ent)
        if min_angle == 0 or ent < min_entropy:
            min_entropy = ent
            min_mono = mono
            min_angle = angle

    #plot_entropies(angles, ent_list)

    return min_mono, min_angle

def project_into_1d(img, angle):
    return InvariantImageGenerator.project_into_one_d(img, angle)