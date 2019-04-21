import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
import h5py as h5
import random

random.seed(1)

# Configuration
plt.rcParams['font.size'] = 9
radius = 3
n_points = 8 * radius
METHOD = 'uniform'
river="data/train1/river.h5"
notriver="data/train1/not_river.h5"

# Read images (riv=river,nriv=not river)
riv=h5.File(river,'r')
riv=riv[list(riv.keys())[0]]
riv=np.array(riv)
nriv=h5.File(notriver,'r')
nriv=nriv[list(nriv.keys())[0]]
nriv=np.array(nriv)

# Okay
riv=np.swapaxes(np.swapaxes(riv,1,2),0,3)
nriv=np.swapaxes(np.swapaxes(nriv,1,2),0,3)

# lpb is the local binary pattern computed for each pixel in the image
def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')

# The Kullback-Leibler divergence is a measure of how one probability distribution
# is different from a second, reference probability distribution.
# These probability distributions are the histograms computed from the LBP
# KL(p,q) = 0 means p and q distributions are identical.
def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

# refs is an array reference LB patterns for various classes (brick, grass, wall)
# img is an input image
# match() gives the best match by comparing the KL divergence between the histogram
# of the img LBP and the histograms of the refs LBPs.
def match(refs, img):
    best_score = 10
    best_name = None
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    for name, ref in refs.items():
        ref_hist, _ = np.histogram(ref, density=True, bins=n_bins,
                                   range=(0, n_bins))
        score = kullback_leibler_divergence(hist, ref_hist)
        if score < best_score:
            best_score = score
            best_name = name
    return best_name

# Turn color images into grayscale using CIE 1931 standard
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    print(gray.shape)
    return gray

print(riv.shape)

# Pick samples
river1=rgb2gray(riv[:,:,:,int(random.random()*riv.shape[-1])])
river2=rgb2gray(riv[:,:,:,int(random.random()*riv.shape[-1])])
notriver1=rgb2gray(nriv[:,:,:,int(random.random()*nriv.shape[-1])])
notriver2=rgb2gray(nriv[:,:,:,int(random.random()*nriv.shape[-1])])
print(river1.shape)

# Pick random samples
refs= {
'river1': local_binary_pattern(river1,n_points,radius,METHOD),
'river2': local_binary_pattern(river2,n_points,radius,METHOD),
'notriver1': local_binary_pattern(notriver1,n_points,radius,METHOD),
'notriver2': local_binary_pattern(notriver2,n_points,radius,METHOD)
}

# Classify rotated textures
print('original: brick, rotated: 30deg, match result: ',
      match(refs, rotate(river1, angle=30, resize=False)))
print('original: brick, rotated: 70deg, match result: ',
      match(refs, rotate(river2, angle=70, resize=False)))
print('original: grass, rotated: 145deg, match result: ',
      match(refs, rotate(notriver1, angle=145, resize=False)))

# Plot histograms of LBP of textures
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3,
                                                       figsize=(9, 6))
plt.gray()

ax1.imshow(river1)
ax1.axis('off')
hist(ax4, refs['river1'])
ax4.set_ylabel('Percentage')

ax2.imshow(river2)
ax2.axis('off')
hist(ax5, refs['river2'])
ax5.set_xlabel('Uniform LBP values')

ax3.imshow(notriver1)
ax3.axis('off')
hist(ax6, refs['notriver1'])

plt.show()
