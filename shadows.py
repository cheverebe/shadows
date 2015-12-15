import numpy as np
import cv2
from boudary_drawer import draw_boundaries
from invariant_image import minimize_entropy, project_to_2d, log_chromaticity_image, project_into_1d, \
    find_invariant_image
from path_finder import find_path, get_roi_corners
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
        (inv_mono, angle) = find_invariant_image(img, two_dim)
    else:
        inv_mono = project_into_1d(two_dim, angle)

    print 'min angle: '+str(angle)
    eq_inv_mono = equalize_hist_3d(inv_mono)
    show_and_save('invariant('+str(angle)+')', 'out/'+name, ext, eq_inv_mono)

    b_eq_inv_mono = cv2.blur(eq_inv_mono, settings['blur_kernel_size'])
    show_and_save('invariant_blur('+str(angle)+')', 'out/'+name, ext, b_eq_inv_mono)

    path_mask = find_path(b_eq_inv_mono)
    show_and_save('path('+str(angle)+')', 'out/'+name, ext, path_mask)

    edged = draw_boundaries(img, path_mask)
    roi_corners = get_roi_corners(edged.shape)
    print((roi_corners[0],roi_corners[2]), (roi_corners[1],roi_corners[3]))
    edged_w_roi = cv2.rectangle(edged, (roi_corners[0],roi_corners[2]), (roi_corners[1],roi_corners[3]), (0,0,255))
    show_and_save('edges('+str(angle)+')', 'out/'+name, ext, edged_w_roi)

#-------------------------------------------
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    listener(settings['predefined_angle'])


#r = load_L1CI_image(name, ext)
#eq_lci = cv2.equalizeHist(r)
#show_and_save('EQ_L1_Chromacity_invariant', name, ext, eq_lci, 255)