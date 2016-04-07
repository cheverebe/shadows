import random
import cv2
import numpy as np


def draw_boundaries(image, mask, color=(0, 0, 255)):
    edges = cv2.Canny(mask, 120, 150)
    killer = 255 - edges
    black = np.zeros((edges.shape[0], edges.shape[1], 1), np.uint8)
    color = [black + color[i] for i in xrange(3)]
    color_edges = cv2.bitwise_and(cv2.merge(color), cv2.merge([edges, edges, edges]))
    black_edges = cv2.bitwise_and(image, cv2.merge([killer, killer, killer]))
    res = cv2.bitwise_or(black_edges, color_edges)
    return res

def draw_region(image, mask):
    killer = 255 - mask
    black = np.ones((mask.shape[0], mask.shape[1], 1), np.uint8)
    color = [black*random.randint(0, 255) for _ in xrange(3)]
    color_edges = cv2.bitwise_and(cv2.merge(color), cv2.merge([mask, mask, mask]))
    black_edges = cv2.bitwise_and(image, cv2.merge([killer, killer, killer]))
    res = cv2.bitwise_or(black_edges, color_edges)
    return res