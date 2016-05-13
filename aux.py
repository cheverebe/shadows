import cv2
img = cv2.imread('img/road13.png')
img = cv2.resize(img, (0, 0),fx=0.6, fy=0.6)
cv2.imwrite('img/road13_rsz.png', img)