import numpy as np
import math
import cv2
from utils import entropy


def get_invariant_l1_chrom_image(two_dim, min_angle):
    U = get_U()
    rad = math.radians(min_angle)
    e_orth = np.array([math.cos(rad), math.sin(rad)])
    e_orth_m = np.matrix(e_orth)
    e_orth_norm = cv2.norm(e_orth)
    P_e_orth = np.matrix(e_orth_m.transpose() * e_orth_m / e_orth_norm)
    extralight = np.matrix([1, 1]).transpose() * 5
    X_tita = [[P_e_orth * np.matrix(elem) + extralight for elem in row] for row in two_dim]
    #print("pmonio ->"+str(X_tita[0][0]))
    #print("eorth"+str(e_orth))
    p_monio = [[(U.transpose() * np.matrix(elem)) for elem in row] for row in X_tita]
    c_monio = np.array([[np.array([math.exp(val) for val in elem]) for elem in row] for row in p_monio])
    (log_chrom, L1chrom) = log_chromaticity_image(c_monio*255)
    L1chrom = np.array(L1chrom)
    return L1chrom

def get_U():
    return np.matrix([[1.0/math.sqrt(2), -1.0/math.sqrt(2), 0], [1.0/math.sqrt(6), 1.0/math.sqrt(6), -2.0/math.sqrt(6)]])

def project_to_2d(log_chrom):
    U = get_U()
    return [[np.array((U * np.matrix(elem).transpose())) for elem in row] for row in log_chrom]


def log_chromaticity_image(img):

    height = img.shape[0]
    width = img.shape[1]
    depth = img.shape[2]

    log_chrom = [[[0, 0, 0] for j in range(width)] for j in range(height)]
    L1chrom = [[[0, 0, 0] for j in range(width)] for j in range(height)]

    #GENERATE LOG CHROMATICITY IMAGE
    if depth == 3:
        for j in xrange(height):
            for i in xrange(width):
                b = img[j][i][0]
                if b < 10: b= 10
                if b > 245: b= 245

                g = img[j][i][1]
                if g < 10: g= 10
                if g > 245: g= 245

                r = img[j][i][2]
                if r < 10: r= 10
                if r > 245: r= 245

                prod = float(r)*g*b
                geometric_mean = math.pow(prod, 1/3)
                c_1 = float(b) / geometric_mean
                c_2 = float(g) / geometric_mean
                c_3 = float(r) / geometric_mean
                log_chrom[j][i][0] = math.log(c_1)
                log_chrom[j][i][1] = math.log(c_2)
                log_chrom[j][i][2] = math.log(c_3)
                L1chrom[j][i][0] = c_1 / (c_1 + c_2 + c_3)
                L1chrom[j][i][1] = c_2 / (c_1 + c_2 + c_3)
                L1chrom[j][i][2] = c_3 / (c_1 + c_2 + c_3)

    log_chrom = np.array(log_chrom)
    L1chrom = np.array(L1chrom)
    return log_chrom, L1chrom

def minimize_entropy(two_dim, angle=None):
    # FIND MIN ANGLE
    min_entropy = 0
    min_mono = []
    entropy_array = []

    if angle is None:
        angles = xrange(1, 180, 3)
        min_angle = 0   #hack horriible
    else:
        angles = [angle]
        min_angle = angle

    for angle in angles:
        print str(angle)
        rad = math.radians(angle)
        mono = [[elem[0]*math.cos(rad)+elem[1]*math.sin(rad) for elem in row] for row in two_dim]
        ent = entropy(mono)
        entropy_array.append(ent)
        if min_angle == 0 or ent < min_entropy:
            min_entropy = ent
            min_mono = mono
            min_angle = angle
    return min_mono, min_angle