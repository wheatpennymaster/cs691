import matplotlib.pyplot as plt
import h5py as h5
import numpy as np
import math
from time import time
import pickle
from PIL import Image

from skimage.feature import hog
from skimage import data, exposure
from skimage import io

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Configuration
model="./rf12.sav"				# model path
image="data/test/DSC05745.JPG"	# image path
ps=64							# patch size
st=8							# stride

# Load the model and image
rf=pickle.load(open(model, 'rb'))
im=Image.open(image)
im=np.array(im)

# Compute coordinates
rows=range(0,im.shape[0]-ps+st,st)
cols=range(0,im.shape[1]-ps+st,st)
num=len(rows)*len(cols)

# Collect feature vectors
fds=np.zeros((num,rf.n_features_))
i=0
for r in range(0,len(rows)):
  for c in range(0,len(cols)):
    p=im[rows[r]:rows[r]+ps,cols[c]:cols[c]+ps,:]
    fd, hog_image = hog(p, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
    fds[i,:]=fd
    i=i+1

# Predictions
predictions=rf.predict_proba(fds)[:, 1]

# Put heatmap here
river_map1=np.zeros(im.shape[0:2])
river_map2=np.zeros(im.shape[0:2])
counter=np.zeros(im.shape[0:2])

# Loop through coordinates
i=0
for r in range(0,len(rows)):
  for c in range(0,len(cols)):
    river_map1[rows[r]:rows[r]+ps,cols[c]:cols[c]+ps]=river_map1[rows[r]:rows[r]+ps,cols[c]:cols[c]+ps]+predictions[i]
    river_map2[rows[r]:rows[r]+ps,cols[c]:cols[c]+ps]=river_map2[rows[r]:rows[r]+ps,cols[c]:cols[c]+ps]+round(predictions[i])
    counter[rows[r]:rows[r]+ps,cols[c]:cols[c]+ps]=counter[rows[r]:rows[r]+ps,cols[c]:cols[c]+ps]+1
    i=i+1

# Save results
rm1=Image.fromarray((255*np.divide(river_map1,counter)).astype(np.uint8))
rm1.save('rm1.png')
rm2=Image.fromarray((255*np.divide(river_map2,counter)).astype(np.uint8))
rm2.save('rm2.png')

