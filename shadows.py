import numpy as np
import cv2
from invariant_image import get_invariant_l1_chrom_image, minimize_entropy, project_to_2d, log_chromaticity_image
from settings import settings
from utils import show_and_save, load_image, load_L1CI_image, equalize_hist_3d


def listener(angle=None):
    name = settings['name']
    ext = settings['extension']
    img = load_image(name, ext)

    (log_chrom, L1chrom) = log_chromaticity_image(img)

    L1chrom = equalize_hist_3d(L1chrom)
    show_and_save('L1_Chromacity', name, ext, L1chrom, 255)

    two_dim = project_to_2d(log_chrom)

    if angle:
        min_angle = angle
    else:
        (min_mono, min_angle) = minimize_entropy(two_dim, angle)

    print 'min angle: '+str(min_angle)

    log_chrom_inv = get_invariant_l1_chrom_image(two_dim,min_angle)
    lci = np.array(log_chrom_inv)
    show_and_save('L1_Chromacity_invariant', name, ext, lci, 255)

    #-------------------------------------------
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    listener(settings['predefined_angle'])


#r = load_L1CI_image(name, ext)
#eq_lci = cv2.equalizeHist(r)
#show_and_save('EQ_L1_Chromacity_invariant', name, ext, eq_lci, 255)