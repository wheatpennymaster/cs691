import matplotlib.pyplot as plt
import h5py as h5
import numpy as np
import math
from time import time
import pickle

from skimage.feature import hog
from skimage import data, exposure
from skimage import io

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import svm

# Configuration
river="data/train2/river.h5"			# for training
notriver="data/train2/not_river.h5"		# for training
river2="data/train1/river.h5"			# for validation
notriver2="data/train1/not_river.h5"	# for validation

################
### Training ###
################

# Read images (riv=river,nriv=not river)
riv=h5.File(river,'r')
riv=riv[list(riv.keys())[0]]
riv=np.array(riv)
nriv=h5.File(notriver,'r')
nriv=nriv[list(nriv.keys())[0]]
nriv=np.array(nriv)

# Flip dimensions (MATLAB writes in row-major)
riv=np.swapaxes(np.swapaxes(riv,1,2),0,3)
nriv=np.swapaxes(np.swapaxes(nriv,1,2),0,3)

# Get size
fd, hog_image = hog(riv[:,:,:,0], orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
print("Feature vector dims: " + str(fd.shape))
print("Num river patches: " + str(riv.shape[-1]))
print("Num not river patches: " + str(nriv.shape[-1]))

# Grab all of our feature vectors
riv_fds=np.zeros((riv.shape[-1],fd.shape[-1]))
riv_labels=np.ones((riv.shape[-1],1))
for i in range(0,riv.shape[-1]):
  fd, hog_image = hog(riv[:,:,:,i], orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
  riv_fds[i,:]=fd

nriv_fds=np.zeros((nriv.shape[-1],fd.shape[-1]))
nriv_labels=np.zeros((nriv.shape[-1],1))
for i in range(0,nriv.shape[-1]):
  fd, hog_image = hog(nriv[:,:,:,i], orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
  nriv_fds[i,:]=fd

# Concatenate
X_train=np.concatenate((riv_fds,nriv_fds),axis=0)
y_train=np.concatenate((riv_labels,nriv_labels),axis=0)

# Train RF
rf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                             max_features=int(round(math.sqrt(fd.shape[-1]))), n_jobs=-1, random_state=0)
t_start = time()
rf.fit(X_train, y_train.ravel())
time_full_train = time() - t_start
rf_auc_full_features = roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1])
print("AUC on training set: " + str(rf_auc_full_features))
confusion_matrix(y_train, np.round(rf.predict_proba(X_train)[:, 1]))

# Train linear SVM
lsvm = svm.SVC(kernel='linear',probability=True)
lsvm.fit(X_train,y_train.ravel())
lsvm_auc_full_features = roc_auc_score(y_train, lsvm.predict_proba(X_train)[:, 1])
print("AUC on training set: " + str(lsvm_auc_full_features))
confusion_matrix(y_train, np.round(lsvm.predict_proba(X_train)[:, 1]))

# Train radial SVM
rsvm = svm.SVC(kernel='rbf',gamma='auto',probability=True)
rsvm.fit(X_train,y_train.ravel())
rsvm_auc_full_features = roc_auc_score(y_train, rsvm.predict_proba(X_train)[:, 1])
print("AUC on training set: " + str(rsvm_auc_full_features))
confusion_matrix(y_train, np.round(rsvm.predict_proba(X_train)[:, 1]))

#############################################
### Validate on other image patches ###
#############################################

# Read images (riv=river,nriv=not river)
riv=h5.File(river2,'r')
riv=riv[list(riv.keys())[0]]
riv=np.array(riv)
nriv=h5.File(notriver2,'r')
nriv=nriv[list(nriv.keys())[0]]
nriv=np.array(nriv)

# Flip dimensions (MATLAB writes in row-major)
riv=np.swapaxes(np.swapaxes(riv,1,2),0,3)
nriv=np.swapaxes(np.swapaxes(nriv,1,2),0,3)

# Get size
fd, hog_image = hog(riv[:,:,:,0], orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
print("Feature vector dims: " + str(fd.shape))
print("Num river patches: " + str(riv.shape[-1]))
print("Num not river patches: " + str(nriv.shape[-1]))

# Grab all of our feature vectors
riv_fds=np.zeros((riv.shape[-1],fd.shape[-1]))
riv_labels=np.ones((riv.shape[-1],1))
for i in range(0,riv.shape[-1]):
  fd, hog_image = hog(riv[:,:,:,i], orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
  riv_fds[i,:]=fd

nriv_fds=np.zeros((nriv.shape[-1],fd.shape[-1]))
nriv_labels=np.zeros((nriv.shape[-1],1))
for i in range(0,nriv.shape[-1]):
  fd, hog_image = hog(nriv[:,:,:,i], orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
  nriv_fds[i,:]=fd

# Concatenate
X_val=np.concatenate((riv_fds,nriv_fds),axis=0)
y_val=np.concatenate((riv_labels,nriv_labels),axis=0)

auc_full_features = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
print("AUC on validation set: " + str(auc_full_features))
confusion_matrix(y_val, np.round(rf.predict_proba(X_val)[:, 1]))

lsvm_auc_full_features = roc_auc_score(y_val, lsvm.predict_proba(X_val)[:, 1])
print("AUC on validation set: " + str(lsvm_auc_full_features))
confusion_matrix(y_val, np.round(lsvm.predict_proba(X_val)[:, 1]))

rsvm_auc_full_features = roc_auc_score(y_val, rsvm.predict_proba(X_val)[:, 1])
print("AUC on validation set: " + str(rsvm_auc_full_features))
confusion_matrix(y_val, np.round(rsvm.predict_proba(X_val)[:, 1]))

# Save the model
pickle.dump(rf, open("rf.sav", "wb"))
pickle.dump(lsvm, open("lsvm.sav", "wb"))
pickle.dump(rsvm, open("rsvm.sav", "wb"))
