import numpy as np
import cv2
from matplotlib import pyplot as plt

im1 = cv2.imread('piano4.jpg')
im2 = cv2.imread('piano5.jpg')

orb = cv2.ORB_create(nfeatures=100,nlevels=1,edgeThreshold=30)
kp1, des1 = orb.detectAndCompute(im1,None)
kp2, des2 = orb.detectAndCompute(im2,None)

# Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
im3 = cv2.drawMatches(im1,kp1,im2,kp2,matches[11:20], outImg=None)

# Show
cv2.imshow('result',im3)
cv2.waitKey(0)
cv2.imwrite('results.png',im3)
