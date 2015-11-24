import cv2
import numpy as np
from LAB.shadow_detection.utils import load_image, show_and_save

image_name = "LAB/shadow_detection/dbg_img/lr3"
image_ext = "png"
image = load_image(image_name, image_ext)

msft = cv2.medianBlur(image, 5)

image2, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

big_regions = []

for i in range(len(contours)):
    blank = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
    region_mask = cv2.drawContours(blank, contours, i, 255, -1)
    s = cv2.sumElems(region_mask/255)[0]
    if s > 700:
        show_and_save(str(i), "region_mask", "png", region_mask)