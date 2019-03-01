import numpy as np
import cv2
from matplotlib import pyplot as plt

im1 = cv2.imread('piano4.jpg')
im2 = cv2.imread('piano5.jpg')
gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=100,nlevels=1,edgeThreshold=30)

corners1=cv2.cornerHarris(gray1,2,3,0.04)
corners2=cv2.cornerHarris(gray2,2,3,0.04)
kpCorners1=np.argwhere(corners1>0.01*corners1.max())
kpCorners2=np.argwhere(corners2>0.01*corners2.max())
kpCorners1 = [cv2.KeyPoint(pt[1],pt[0],3) for pt in kpCorners1]
kpCorners2 = [cv2.KeyPoint(pt[1],pt[0],3) for pt in kpCorners2]
kp1,des1=orb.compute(gray1,kpCorners1)
kp2,des2=orb.compute(gray2,kpCorners2)

# Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
im3 = cv2.drawMatches(im1,kp1,im2,kp2,matches[:10], outImg=None)

# Show
cv2.imshow('result',im3)
cv2.waitKey(0)
cv2.imwrite('result.png',im3)
