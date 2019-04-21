r_t=h5read('im1/river.h5','/patches');
nr_t=h5read('im2/not_river.h5','/patches');
r_v=h5read('im2/river.h5','/patches');
nr_v=h5read('im2/not_river.h5','/patches');

% Get features
p1=zeros(size(r_t,4),1);
for i=1:size(r_t,4)
    a=r_t(:,:,:,i);
    a=single(rgb2gray(a(25:40,25:40,:)));
    p1(i,:)=var(single(a(:)));
end

p0=zeros(size(nr_t,4),1);
for i=1:size(nr_t,4)
    a=nr_t(:,:,:,i);
    a=single(rgb2gray(a(25:40,25:40,:)));
    p0(i,:)=var(single(a(:)));
end

% Training set
trX=cat(1,p1,p0);
trY=cat(1,ones(size(r_t,4),1,'logical'),zeros(size(nr_t,4),1,'logical'));

% Get features
p1=zeros(size(r_v,4),1);
for i=1:size(r_v,4)
    a=r_v(:,:,:,i);
    a=single(rgb2gray(a(25:40,25:40,:)));
    p1(i,:)=var(single(a(:)));
end

p0=zeros(size(nr_v,4),1);
for i=1:size(nr_v,4)
    a=nr_v(:,:,:,i);
    a=single(rgb2gray(a(25:40,25:40,:)));
    p0(i,:)=var(single(a(:)));
end

% Validation set
vlX=cat(1,p1,p0);
vlY=cat(1,ones(size(r_v,4),1,'logical'),zeros(size(nr_v,4),1,'logical'));

% Model
[trIdx,C]=kmeans(trX,2);

% Acc
tr_acc=sum((trY==1&trIdx==1)|(trY==0&trIdx==2))./length(trY);
[~,vlIdx] = pdist2(C,vlX,'euclidean','Smallest',1);
vl_acc=sum((vlY==1&vlIdx'==1)|(vlY==0&vlIdx'==2))./length(vlY);