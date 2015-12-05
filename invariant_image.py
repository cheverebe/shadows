import numpy as np
import math
import cv2
from Greyscale.InvariantImageGenerator import InvariantImageGenerator
from LAB.shadow_detection.utils import show_and_save
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
    height = img.shape[0]
    width = img.shape[1]
    depth = img.shape[2]

    img = iig.limit_image(img)

    log_chrom = [[[0, 0, 0] for j in range(width)] for j in range(height)]

    #GENERATE LOG CHROMATICITY IMAGE
    if depth == 3:
        for j in xrange(height):
            for i in xrange(width):
                b = img[j][i][0]
                g = img[j][i][1]
                r = img[j][i][2]

                prod = float(r)*g*b
                geometric_mean = math.pow(prod, 1.0/3)
                c_1 = float(b) / geometric_mean
                c_2 = float(g) / geometric_mean
                c_3 = float(r) / geometric_mean
                log_chrom[j][i][0] = math.log(c_1)
                log_chrom[j][i][1] = math.log(c_2)
                log_chrom[j][i][2] = math.log(c_3)

    log_chrom = np.array(log_chrom)
    return log_chrom

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