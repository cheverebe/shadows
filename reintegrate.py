import cv2
import numpy as np
from settings import settings
from utils import show_and_save, load_L1CI_image, load_image, equalize_hist_3d


def reintegrate(lci, name, ext, img):

    #  lci_eq = equalize_hist_3d(lci)
    #  show_and_save('LCI_EQ', name, ext, lci_eq)

    # L1 EQ edges
    tmin2 = 30
    tmax2 = 50
    edges_L1 = cv2.Canny(lci, tmin2, tmax2)
    #
    #PRINT Edges_L1_EQ_mshft edges
    show_and_save('Edges_L1', name, ext, edges_L1)

    #+++++++++++++++++++++++++++++

    kernel = np.ones((8,8),np.uint8)
    dilated_edges = cv2.dilate(edges_L1, kernel, iterations=1)
    show_and_save('Dilated_L1', name, ext, dilated_edges)

    #-------------------------------------------------
    #
    #MEAN SHIFTED ORIGINAL---------------------
    sp = 25
    sr = 50
    msft_original=cv2.pyrMeanShiftFiltering(img, sp, sr)

    show_and_save('msft_original', name, ext, msft_original)
    #
    tmin2 = 150
    tmax2 = 250
    original_msft_edges = cv2.Canny(msft_original, tmin2, tmax2)
    #
    #PRINT edges 2
    show_and_save('Edges_msft_original', name, ext, original_msft_edges)
    #-------------------------------------------
    #DIFERENCE---------------------
    diff = np.uint8(original_msft_edges-dilated_edges)
    show_and_save('Diff', name, ext, diff)
    #SUM---------------------
    diff = np.uint8(original_msft_edges+dilated_edges)
    show_and_save('Sum', name, ext, diff)


if __name__ == '__main__':
    name = settings['name']
    ext = settings['extension']
    img = load_image(name,ext)
    lci = load_L1CI_image(name, ext)
    reintegrate(lci, name, ext, img)


    #matplotlib.rcParams['axes.unicode_minus'] = False
    #fig, ax = plt.subplots()
    #ax.plot(angles, entropy_array, '.')
    #ax.set_title('Entropy')
    #plt.show()
    #plt.savefig('img/out/'+name+'/plot.png')