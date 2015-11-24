import numpy as np
import cv2
from Greyscale.InvariantImageGenerator import InvariantImageGenerator
from invariant_image import get_invariant_l1_chrom_image, minimize_entropy, project_to_2d, log_chromaticity_image
from settings import settings
from utils import load_image
from LAB.shadow_detection.utils import entropy, show_and_save, equalize_hist_3d, show_image


def listener(angle=None):
    name = settings['name']
    ext = settings['extension']
    img = load_image(name, ext)

    (log_chrom, L1chrom) = log_chromaticity_image(img)
    #log_chrom = iig.log_chrom_image()
    #show_and_save("chrom_image--old", 'log', 'png', equalize_hist_3d(np.array(log_chrom)))

    two_dim = project_to_2d(log_chrom)

    print(str(two_dim.shape))
    (out_1, out_2) = cv2.split(two_dim)
    #show_and_save("chrom_image--old", '2d_log_1', 'png', equalize_hist_3d(np.array(out_1)))
    #show_and_save("chrom_image--old", '2d_log_2', 'png', equalize_hist_3d(np.array(out_2)))

    (min_mono, angle) = minimize_entropy(two_dim, angle)

    print 'min angle: '+str(angle)
    show_and_save('invariant('+str(angle)+')', 'out/'+name, ext, equalize_hist_3d(np.array(min_mono)), 255)
    #log_chrom_inv = get_invariant_l1_chrom_image(two_dim,min_angle)
    #lci = np.array(log_chrom_inv)
    #show_and_save('L1_Chromacity_invariant', name, ext, lci, 255)

    #-------------------------------------------
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    listener(settings['predefined_angle'])


#r = load_L1CI_image(name, ext)
#eq_lci = cv2.equalizeHist(r)
#show_and_save('EQ_L1_Chromacity_invariant', name, ext, eq_lci, 255)