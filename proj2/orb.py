import numpy as np
import cv2
from matplotlib import pyplot as plt

def nothing(x):
    pass

img = cv2.imread('piano1.jpg')


cv2.namedWindow('params')
cv2.createTrackbar('nfeatures','params',500,1000,nothing)
cv2.createTrackbar('nlevels','params',8,20,nothing)
cv2.createTrackbar('edgeThreshold','params',31,100,nothing)

while (True):

  # Interactive sliders
  nfeatures=cv2.getTrackbarPos('nfeatures','params')
  nlevels=cv2.getTrackbarPos('nlevels','params')
  edgeThreshold=cv2.getTrackbarPos('edgeThreshold','params')
  
  orb = cv2.ORB_create(nfeatures=nfeatures,nlevels=nlevels,edgeThreshold=edgeThreshold)
  kp,des = orb.detectAndCompute(img, None)

  #Plot
  img2 = cv2.drawKeypoints(img,kp,color=(0,255,0),outImage=None)
  cv2.imshow('image',img2)
  if cv2.waitKey(1) & 0xFF == ord('s'):
    cv2.imwrite('result.png',img2)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cv2.destroyAllWindows()

