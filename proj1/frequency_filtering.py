# 5. Frequency Filtering

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
parser.add_argument('--cutoff', dest='cutoff', type=float, default='0.5', help='cutoff for filtering')
parser.add_argument('--order', dest='order', type=int, default=1, help='Butterworth filter order')
args = parser.parse_args()

# Get image from URL
r=requests.get(dog_url)
with open('image.jpg', 'wb') as f:
  f.write(r.content)

# Read in image and pad
im=cv2.imread(args.impath)

# Convert to grayscale
if im.shape[-1]==3: # Color image
  im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# FFT2
ft=np.fft.fft2(im.astype(float))
logmag=np.fft.fftshift(np.log(np.abs(ft)+1))
logmag=logmag/logmag.max()
logmag=ski.img_as_ubyte(logmag)

# I'm not sure what's going on here honestly
V=(np.linspace(-int(im.shape[0]/2), int(im.shape[0]/2)-1, im.shape[0]))
U=(np.linspace(-int(im.shape[1]/2), int(im.shape[1]/2)-1, im.shape[1]))
U,V=np.meshgrid(U,V)
D=np.sqrt(U*U + V*V)
D0=args.cutoff*D.max()

# Ideal lowpass and highpass (zeroing frequencies)
idealLowPass=(D<=D0)
idealHighPass=(D>=D0)
magLowPass=ft*np.fft.fftshift(idealLowPass)
magHighPass=ft*np.fft.fftshift(idealHighPass)
imLowPass=np.abs(np.fft.ifft2(magLowPass))
imHighPass=np.abs(np.fft.ifft2(magHighPass))
imLowPass=ski.img_as_ubyte(imLowPass/imLowPass.max())
imHighPass=ski.img_as_ubyte(imHighPass/imHighPass.max())

# Butterworth Filters
butterLowPass=1.0/(1+(np.sqrt(2)-1)*np.power(D/D0,2*args.order))
butterHighPass=butterLowPass.max()-butterLowPass
magButterLowPass=ft*np.fft.fftshift(butterLowPass)
magButterHighPass=ft*np.fft.fftshift(butterHighPass)
imButterLowPass=np.abs(np.fft.ifft2(magButterLowPass))
imButterHighPass=np.abs(np.fft.ifft2(magButterHighPass))
imButterLowPass=ski.img_as_ubyte(imButterLowPass/imButterLowPass.max())
imButterHighPass=ski.img_as_ubyte(imButterHighPass/imButterHighPass.max())

# Display results
plt.subplot(231),plt.imshow(im,cmap='gray'),plt.title('original')
plt.subplot(232),plt.imshow(imLowPass,cmap='gray'),plt.title('IdealLowPass')
plt.subplot(235),plt.imshow(imHighPass,cmap='gray'),plt.title('IdealHighPass')
plt.subplot(233),plt.imshow(imButterLowPass,cmap='gray'),plt.title('ButterworthLowPass')
plt.subplot(236),plt.imshow(imButterHighPass,cmap='gray'),plt.title('ButterworthHighPass')
plt.savefig('freqFilter.jpg',bbox_inches='tight')
plt.show()

# Compute coefficients for display purposes
logButterLP=np.fft.fftshift(np.log(np.abs(magButterLowPass)+1))
logButterLP=logButterLP/logButterLP.max()
logButterLP=ski.img_as_ubyte(logButterLP)
logButterHP=np.fft.fftshift(np.log(np.abs(magButterHighPass)+1))
logButterHP=logButterHP/logButterHP.max()
logButterHP=ski.img_as_ubyte(logButterHP)
plt.subplot(231),plt.imshow(logmag,cmap='gray'),plt.title('Original')
plt.subplot(232),plt.imshow(logButterLP,cmap='gray'),plt.title('Butter LP-Scaled Coef')
plt.subplot(235),plt.imshow(logButterHP,cmap='gray'),plt.title('Butter HP-Scaled Coef')
plt.subplot(233),plt.imshow(imButterLowPass,cmap='gray'),plt.title('Butter LP Image')
plt.subplot(236),plt.imshow(imButterHighPass,cmap='gray'),plt.title('Butter HP Image')
plt.savefig('freqFilterFig.jpg',bbox_inches='tight')
plt.show()
