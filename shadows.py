import numpy as np
import cv2
from invariant_image import minimize_entropy, project_to_2d, log_chromaticity_image, project_into_1d
from settings import settings
from utils import load_image
from LAB.shadow_detection.utils import show_and_save, equalize_hist_3d


def listener(angle=None):
    name = settings['name']
    ext = settings['extension']
    img = load_image(name, ext)

    log_chrom = log_chromaticity_image(img)

    two_dim = project_to_2d(log_chrom)

    if not angle:
        (min_mono, angle) = minimize_entropy(two_dim, None, name)
    else:
        min_mono = project_into_1d(two_dim, angle)

    print 'min angle: '+str(angle)
    show_and_save('invariant('+str(angle)+')', 'out/'+name, ext, equalize_hist_3d(np.array(min_mono)))

    #-------------------------------------------
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    listener(settings['predefined_angle'])


#r = load_L1CI_image(name, ext)
#eq_lci = cv2.equalizeHist(r)
#show_and_save('EQ_L1_Chromacity_invariant', name, ext, eq_lci, 255)