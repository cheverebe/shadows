import cv2


def draw_boundaries(image, mask):
    edges = cv2.Canny(mask, 120, 150)
    return cv2.bitwise_or(image, cv2.merge([edges, edges, edges]))