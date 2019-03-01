import numpy as np
import cv2
import time

def nothing(x):
    pass

sigma=0.33

# Dynamic thresholds
cv2.namedWindow('thresholds')
cv2.createTrackbar('low','thresholds',0,255,nothing)
cv2.createTrackbar('high','thresholds',0,255,nothing)

# Capture device
cap = cv2.VideoCapture(0)

winname='video'
cv2.namedWindow(winname)
cv2.moveWindow(winname,40,30)
while(True):
  # Capture frame
  ret, frame = cap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Get slide bar value
  low=cv2.getTrackbarPos('low','thresholds')
  high=cv2.getTrackbarPos('high','thresholds')

  # Auto thresholding
  #m=np.median(gray)
  #low=int(max(0,(1.0-sigma)*m))
  #high=int(min(255,(1.0+sigma)*m))
  edges=cv2.Canny(gray,low,high)

  # Display
  cv2.imshow(winname,edges)
  if cv2.waitKey(1) & 0xFF == ord('s'):
  # Save somes frames
    cv2.imwrite('kitchen_hand_frame.png',frame)
    for i in range(0,4):
      ret, frame = cap.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      edges=cv2.Canny(gray,low,high)
      cv2.imwrite('kitchen_hand_frame'+str(i)+'.png',edges)
    break
    
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


