# 4. Frequency Analysis

import argparse
import requests
import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage as ski
dog_url='http://csweb.cs.wfu.edu/~pauca/csc391-CV/ImageData/DSC_9259.JPG'

# Input args for image
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', dest='impath', type=str, default='image.jpg', help='path to image')
args = parser.parse_args()

# Get image from URL
r=requests.get(dog_url)
with open('image.jpg', 'wb') as f:
  f.write(r.content)

# Read in image and pad
imo=cv2.imread(args.impath)
im=np.copy(imo)

# Convert to grayscale
if im.shape[-1]==3: # Color image
  im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# FFT2
ft=np.fft.fft2(im.astype(float))
mag=np.fft.fftshift(np.abs(ft)+1)
mag=mag/mag.max()
mag=ski.img_as_ubyte(mag)
logmag=np.fft.fftshift(np.log(np.abs(ft)+1))
logmag=logmag/logmag.max()
logmag=ski.img_as_ubyte(logmag)

# Display Fourier transform
plt.subplot(121),plt.imshow(mag,cmap='gray'),plt.title('magnitude')
plt.subplot(122),plt.imshow(logmag,cmap='gray'),plt.title('log(mag)')
plt.show()
