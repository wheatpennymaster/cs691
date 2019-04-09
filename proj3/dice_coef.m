% Read
mask_ground_truth=imread('data/test/DSC05745.png');
mask_predict=imread('rm2.png');

% Threshold
mask_predict=imbinarize(rgb2gray(imread('data/test/DSC05745.JPG')));

% Dice
d=dice(mask_ground_truth',mask_predict);