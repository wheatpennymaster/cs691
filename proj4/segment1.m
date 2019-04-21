% Configuration
load('patch_svm0.mat');
im=imread('imt/sample.png');
mask=imread('imt/mask.png');

% Get patch features
teX=zeros(length(1:2:size(im,1))*length(1:2:size(im,2)),6,'single');
teY=zeros(length(1:2:size(im,1))*length(1:2:size(im,2)),1,'logical');
mm=zeros(size(mask));
a=1;
for i=1:2:size(im,1)
    for j=1:2:size(im,2)
        p=single(im(i:i+1,j:j+1,:));
        m=squeeze(mean(mean(p,1),2));
        s=squeeze(std(std(p,1,1),1,2));
        teX(a,:)=cat(1,m,s)';
        gt=round(mean(mean(mask(i:i+1,j:j+1))));
        teY(a)=gt;
        a=a+1;
    end
end

% Predict
pr=predict(l_svm,teX);

% Segment
a=1;
for i=1:2:size(im,1)
    for j=1:2:size(im,2)
        mm(i:i+1,j:j+1)=pr(a);
        a=a+1;
    end
end

% Dice
dice = 2*nnz(mm&mask)/(nnz(mm) + nnz(mask));
       