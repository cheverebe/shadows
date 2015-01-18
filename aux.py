import numpy as np
import cv2
name = 'auto'
if
img = cv2.imread('img/'+name+'.png')
cv2.namedWindow('Original image', cv2.WINDOW_NORMAL)
cv2.imshow('Original image', img)

sp = 15
sr = 30
mshft2=cv2.pyrMeanShiftFiltering(img,sp,sr)

cv2.namedWindow('Mean Shifted 2', cv2.WINDOW_NORMAL)
cv2.imshow('Mean Shifted 2', mshft2)
cv2.imwrite('img/out/'+name+'/mshft2.png', mshft2)

tmin2 = 150
tmax2 = 250
edges2 = cv2.Canny(mshft2,tmin2,tmax2 )

#PRINT edges 2
cv2.namedWindow('Edges2', cv2.WINDOW_NORMAL)
cv2.imshow('Edges2', edges2)
cv2.imwrite('img/out/'+name+'/edges2.png', edges2)

cv2.waitKey(0)
cv2.destroyAllWindows()