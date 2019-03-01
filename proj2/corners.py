import numpy as np
import cv2
import time

def nothing(x):
    pass

sigma=0.33

# Dynamic thresholds
cv2.namedWindow('params')
cv2.createTrackbar('blockSize','params',2,21,nothing)
cv2.createTrackbar('ksize*2+1','params',1,10,nothing)
cv2.createTrackbar('k*1000','params',40,1000,nothing)

# Capture device
cap = cv2.VideoCapture(0)

winname='video'
cv2.namedWindow(winname)
cv2.moveWindow(winname,40,30)
while(True):
  # Capture frame
  ret, frame = cap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Get slidebar value
  blockSize=cv2.getTrackbarPos('blockSize','params')
  ksize=cv2.getTrackbarPos('ksize*2+1','params')*2+1
  k=cv2.getTrackbarPos('k*1000','params')/1000

  # Corners
  corners=cv2.cornerHarris(gray,blockSize,ksize,k)

  # Superimpose
  corners=cv2.dilate(corners,None)
  frame[corners>0.05*corners.max()]=[0,0,255]

  # Display
  cv2.imshow(winname,frame)
  if cv2.waitKey(1) & 0xFF == ord('s'):
  # Save somes frames
    for i in range(0,4):
      ret, frame = cap.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      corners=cv2.cornerHarris(gray,blockSize,ksize,k)
      corners=cv2.dilate(corners,None)
      frame[corners>0.05*corners.max()]=[0,0,255]
      cv2.imwrite('table3_frame'+str(i)+'.png',frame)
    break

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


