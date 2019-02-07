# 3. Spatial Filtering

import argparse
import requests
import cv2
import numpy as np
from matplotlib import pyplot as plt
dog_url='http://csweb.cs.wfu.edu/~pauca/csc391-CV/ImageData/DSC_9259.JPG'

# Input args for filter and image
parser = argparse.ArgumentParser()
parser.add_argument('--filter_size', dest='fsize', type=int, default=11, help='filter size')
parser.add_argument('--image_path', dest='impath', type=str, default='image.jpg', help='path to image')
parser.add_argument('--filter_type', dest='ftype', type=str, default='Gauss', help='filter type: Gaussian (default), median: \'median\'')
args = parser.parse_args()


# Get image from URL
r=requests.get(dog_url)
with open('image.jpg', 'wb') as f:
  f.write(r.content)

# Read in image and pad
imo=cv2.imread(args.impath)
im=np.copy(imo)

# Define and apply filter
if args.ftype=='Gauss':
  if im.shape[-1]==3: # Color image
    im[:,:,0]=cv2.GaussianBlur(im[:,:,0],(args.fsize,args.fsize),3)
    im[:,:,1]=cv2.GaussianBlur(im[:,:,1],(args.fsize,args.fsize),3)
    im[:,:,2]=cv2.GaussianBlur(im[:,:,2],(args.fsize,args.fsize),3)
  else: # Grayscale image
    im=cv2.GaussianBlur(im,(args.fsize,args.fsize),3)
elif args.ftype=='median':
  if im.shape[-1]==3: # Color image
    im[:,:,0]=cv2.medianBlur(im[:,:,0],args.fsize)
    im[:,:,1]=cv2.medianBlur(im[:,:,1],args.fsize)
    im[:,:,2]=cv2.medianBlur(im[:,:,2],args.fsize)
  else: # Grayscale image
    im=cv2.GaussianBlur(im,(args.fsize,args.fsize),3)

# Filter for display purposes
f=cv2.getGaussianKernel(args.fsize,3)
f=cv2.getGaussianKernel(args.fsize,3)
f=f*np.transpose(f)

# What???
plt.subplot(131),plt.imshow(cv2.cvtColor(imo,cv2.COLOR_BGR2RGB)),plt.title('Original')
plt.subplot(132),plt.imshow(f,cmap='gray'),plt.title('Filter')
plt.subplot(133),plt.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),plt.title('Filtered')
plt.show()

# Write filtered image
print(''.join(args.impath.split(".")[0:-1])+'_filtered'+'.'+args.impath.split(".")[-1])
cv2.imwrite(''.join(args.impath.split(".")[0:-1])+'_filtered'+'.'+args.impath.split(".")[-1], im)
