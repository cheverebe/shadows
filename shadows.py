import numpy as np
import cv2
import math
import matplotlib
import matplotlib.pyplot as plt


def listener():
    # Load an color image in grayscale

    # # TODO: que se haga una sola vez
    # u = np.array([1/math.sqrt(3)] * 3)
    # ut = np.array([1/math.sqrt(3)] * 3)
    # ut.transpose()
    # I = np.identity(3)
    # P = I - u * ut
    U = np.array([[1.0/math.sqrt(2), -1.0/math.sqrt(2), 0], [1.0/math.sqrt(6), 1.0/math.sqrt(6), -2.0/math.sqrt(6)]])

    name = 'balcon'
    ext = 'png'
    img = cv2.imread('img/'+name+"."+ext)
    cv2.namedWindow('Original image', cv2.WINDOW_NORMAL)
    cv2.imshow('Original image', img)

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

    l1cp = np.array(L1chrom)
    print("max---"+str(l1cp.max()))
    print("min---"+str(l1cp.min()))
    cv2.namedWindow('L1 Chromacity', cv2.WINDOW_NORMAL)
    cv2.imshow('L1 Chromacity', l1cp)
    cv2.imwrite('img/out/'+name+'/out_l1_chromacity.'+ext, l1cp*255)

    #show log chromacity image
    lc = np.array(log_chrom)
    min_val = lc.min()
    max_val = lc.max()

    two_dim = [[np.array((U * np.matrix(elem).transpose())) for elem in row] for row in log_chrom]

    # FIND MIN ANGLE
    min_entropy = 0
    min_mono = []
    entropy_array = []

    angles = [151]#xrange(1, 180, 3)
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

    print 'min angle: '+str(min_angle)

    #matplotlib.rcParams['axes.unicode_minus'] = False
    #fig, ax = plt.subplots()
    #ax.plot(angles, entropy_array, '.')
    #ax.set_title('Entropy')
    #plt.show()
    #plt.savefig('img/out/'+name+'/plot.png')

    #search for max min values
    mono_img = np.array(min_mono)
    min_val = mono_img.min()
    max_val = mono_img.max()

    print "\n>>>Invariant image<<<"
    print "Min val: "+str(min_val)
    print "Max val: "+str(max_val)

    #print & show image 1
    #cv2.namedWindow('Mono', cv2.WINDOW_NORMAL)
    #cv2.imshow('Mono', mono_img)
    #cv2.imwrite('img/out/'+name+'/out_'+str(min_angle)+'.'+ext, mono_img)

    rango = max_val-min_val
    coef = 1 / rango

    print "Range: "+str(rango)
    print "Coefficient: "+str(coef)

    #ADAPT RANGE TO 0..1
    min_mono2 = np.array([[coef*(elem-min_val) for elem in row] for row in min_mono])

    print "\n>>>Invariant image(in range)<<<"
    min_val = min_mono2.min()
    max_val = min_mono2.max()

    print "Min val: "+str(min_val)
    print "Max val: "+str(max_val)

    #PRINT IMAGE 2
    cv2.namedWindow('Mono2', cv2.WINDOW_NORMAL)
    cv2.imshow('Mono2', min_mono2)
    cv2.imwrite('img/out/'+name+'/out2_'+str(min_angle)+'.'+ext, min_mono2*255)

    edges1 = cv2.Canny(np.uint8(min_mono2*255),150,250)

    #PRINT edges
    cv2.namedWindow('Edges1', cv2.WINDOW_NORMAL)
    cv2.imshow('Edges1', edges1)
    cv2.imwrite('img/out/'+name+'/edges_'+str(min_angle)+'.'+ext, edges1)

    #mshft=cv2.pyrMeanShiftFiltering()
    edges2 = cv2.Canny(img,150,250)

    #PRINT edges
    cv2.namedWindow('Edges2', cv2.WINDOW_NORMAL)
    cv2.imshow('Edges2', edges2)
    cv2.imwrite('img/out/'+name+'/edges2_'+str(min_angle)+'.'+ext, edges2)


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

if __name__ == '__main__':
    listener()
