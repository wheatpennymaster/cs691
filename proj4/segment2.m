% Configuration
load('pix_svm0.mat');
im=imread('imt/sample.png');
mask=imread('imt/mask.png');

% Reshape into feature map
r=im(:,:,1);
g=im(:,:,2);
b=im(:,:,3);
teX=single(cat(2,r(:),g(:),b(:)));
teY=mask(:);

% Predict
pr=predict(l_svm,teX);
mm=reshape(pr,[size(mask,1) size(mask,2)]);

% Dice
dice = 2*nnz(mm&mask)/(nnz(mm) + nnz(mask));