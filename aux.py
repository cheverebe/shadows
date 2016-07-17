import cv2
img = cv2.imread('img/kitti/umm_000080.png')
img = cv2.resize(img, (0, 0),fx=0.6, fy=0.6)
cv2.imwrite('img/umm_000080.png', img)