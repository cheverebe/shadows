import numpy as np
import cv2
name = 'balcon'
ext = 'png'
#img = cv2.imread('img/'+name+'.png')
img = cv2.imread('img/out/'+name+'/out2_151.png')
cv2.namedWindow('Original image', cv2.WINDOW_NORMAL)
cv2.imshow('Original image', img)

min_angle = 158

tmin2 = 20
tmax2 = 55
edges2 = cv2.Canny(img,tmin2,tmax2 )

#PRINT edges 2
cv2.namedWindow('Edges2', cv2.WINDOW_NORMAL)
cv2.imshow('Edges2', edges2)
cv2.imwrite('img/out/'+name+'/edges2.png', edges2)

cv2.waitKey(0)
cv2.destroyAllWindows()