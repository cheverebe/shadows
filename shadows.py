import numpy as np
import cv2
import math
import matplotlib
import matplotlib.pyplot as plt

def get_U():
    return np.matrix([[1.0/math.sqrt(2), -1.0/math.sqrt(2), 0], [1.0/math.sqrt(6), 1.0/math.sqrt(6), -2.0/math.sqrt(6)]])

def project_to_2d(log_chrom):
    U = get_U()
    return [[np.array((U * np.matrix(elem).transpose())) for elem in row] for row in log_chrom]


def minimize_entropy(two_dim):
    # FIND MIN ANGLE
    min_entropy = 0
    min_mono = []
    entropy_array = []

    angles = [158]#xrange(1, 180, 3)
    min_angle = 0   #hack horriible

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
    return (min_mono, min_angle)


def show_image(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)

def show_and_save(param, name, ext, img, fact=1):
    cv2.namedWindow(param, cv2.WINDOW_NORMAL)
    cv2.imshow(param, img)
    cv2.imwrite('img/out/'+name+'/out_'+param+'.'+ext, img*fact)


def adapt_to_range(min_mono):

    #search for max min values
    mono_img = np.array(min_mono)
    min_val = mono_img.min()
    max_val = mono_img.max()

    print "\n>>>Invariant image<<<"
    print "Min val: "+str(min_val)
    print "Max val: "+str(max_val)

    rango = max_val-min_val
    coef = 1 / rango

    print "Range: "+str(rango)
    print "Coefficient: "+str(coef)

    #ADAPT RANGE TO 0..1----------------
    min_mono2 = np.array([[coef*(elem-min_val) for elem in row] for row in min_mono])
    return min_mono2


def get_invariant_l1_chrom_image(two_dim, min_angle):
    U = get_U()
    rad = math.radians(min_angle)
    e_orth = np.array([math.cos(rad), math.sin(rad)])
    e_orth_m = np.matrix(e_orth)
    e_orth_norm = cv2.norm(e_orth)
    P_e_orth = np.matrix(e_orth_m.transpose() * e_orth_m / e_orth_norm)
    X_tita = [[P_e_orth * np.matrix(elem) + e_orth_m.transpose() * 0.7 for elem in row] for row in two_dim]
    print("pmonio ->"+str(X_tita[0][0]))
    print("eorth"+str(e_orth))
    p_monio = [[(U.transpose() * np.matrix(elem)) for elem in row] for row in X_tita]
    c_monio = np.array([[np.array([math.exp(val) for val in elem]) for elem in row] for row in p_monio])
    (log_chrom, L1chrom) = log_chromaticity_image(c_monio*255)
    return L1chrom

def listener():
    # Load an color image in grayscale

    name = 'balcon'
    ext = 'png'
    img = cv2.imread('img/'+name+"."+ext)
    show_image('Original image', img)

    (log_chrom, L1chrom) = log_chromaticity_image(img)

    lc = np.array(log_chrom)
    l1cp = np.array(L1chrom)

    print("max---"+str(l1cp.max()))
    print("min---"+str(l1cp.min()))
    show_and_save('L1_Chromacity', name, ext, l1cp, 255)

    two_dim = project_to_2d(log_chrom)

    (min_mono, min_angle) = minimize_entropy(two_dim)

    print 'min angle: '+str(min_angle)

    log_chrom_inv = get_invariant_l1_chrom_image(two_dim,min_angle)
    lci = np.array(log_chrom_inv)
    show_and_save('L1_Chromacity_invariant', name, ext, lci, 255)
    r = np.uint8(lci*255)
    #matplotlib.rcParams['axes.unicode_minus'] = False
    #fig, ax = plt.subplots()
    #ax.plot(angles, entropy_array, '.')
    #ax.set_title('Entropy')
    #plt.show()
    #plt.savefig('img/out/'+name+'/plot.png')

    #print & show image 1
    #show_and_save('mono', name, ext, mono_img)
    # min_mono2 = adapt_to_range(min_mono)
    #
    # print "\n>>>Invariant image(in range)<<<"
    # min_val = min_mono2.min()
    # max_val = min_mono2.max()
    #
    # print "Min val: "+str(min_val)
    # print "Max val: "+str(max_val)
    #
    # #PRINT IMAGE 2
    # show_and_save('mono2', name, ext, min_mono2, 255)
    #
    #-------------------------------------------
    #EDGES CANNY------------------
    # sp = 15
    # sr = 30
    m = cv2.merge((r,r,r))
    # mshft=cv2.pyrMeanShiftFiltering(m,sp,sr)
    # (r,g,b) = cv2.split(mshft)
    #
    # cv2.namedWindow('Mean Shifted', cv2.WINDOW_NORMAL)
    # cv2.imshow('Mean Shifted', r)
    # cv2.imwrite('img/out/'+name+'/mshft_'+str(min_angle)+'.'+ext, mshft)
    #
    # mshft = r
    tmin = 20
    tmax = 55
    edges1 = cv2.Canny(r,tmin,tmax)
    #
    #PRINT edges
    show_and_save('Edges_L1', name, ext, edges1)
    #-------------------------------------------------
    #
    #MEAN SHIFTED---------------------
    sp = 25
    sr = 50
    mshft2=cv2.pyrMeanShiftFiltering(img,sp,sr)

    show_and_save('Mshft2', name, ext, mshft2)
    #
    tmin2 = 150
    tmax2 = 250
    edges2 = cv2.Canny(mshft2,tmin2,tmax2 )
    #
    #PRINT edges 2
    show_and_save('Edges2', name, ext, edges2)
    # #-------------------------------------------
    # #DIFERENCE---------------------
    diff = edges2-edges1
    show_and_save('Diff', name, ext, diff)

    #-------------------------------------------
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def entropy(matrix):
    mono_img = np.array(matrix)
    min_val = mono_img.min()
    max_val = mono_img.max()
    rango = max_val-min_val
    coef = 1 / rango

    bins = 64
    sums = [0] * bins

    for row in matrix:
        for val in row:
            normal_val = coef*(val-min_val)
            bin = int(normal_val*bins)
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


def mono_image(width, height):
    return [[np.int64(0) for j in xrange(width)] for j in xrange(height)]

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

    return (log_chrom, L1chrom)

if __name__ == '__main__':
    listener()
