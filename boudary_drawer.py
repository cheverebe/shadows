import cv2
import numpy as np


def draw_boundaries(image, mask):
    edges = cv2.Canny(mask, 120, 150)
    killer = 255 - edges
    black = np.zeros((edges.shape[0], edges.shape[1], 1), np.uint8)
    color = [black, black, black + 255]
    color_edges = cv2.bitwise_and(cv2.merge(color), cv2.merge([edges, edges, edges]))
    black_edges = cv2.bitwise_and(image, cv2.merge([killer, killer, killer]))
    res = cv2.bitwise_or(black_edges, color_edges)
    return res