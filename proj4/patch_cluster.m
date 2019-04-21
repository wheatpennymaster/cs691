% repeat results
rng(1);

r_t=h5read('im1/river.h5','/patches');
nr_t=h5read('im1/not_river.h5','/patches');
r_v=h5read('im2/river.h5','/patches');
nr_v=h5read('im2/not_river.h5','/patches');

% Get features
m1=zeros(size(r_t,4),3);
s1=zeros(size(r_t,4),3);
for i=1:size(r_t,4)
    a=single(r_t(:,:,:,i));
    m1(i,:)=squeeze(mean(mean(a,1),2));
    s1(i,:)=squeeze(std(std(a,1,1),1,2));
end

m2=zeros(size(nr_t,4),3);
s2=zeros(size(nr_t,4),3);
for i=1:size(nr_t,4)
    a=single(nr_t(:,:,:,i));
    m2(i,:)=squeeze(mean(mean(a,1),2));
    s2(i,:)=squeeze(std(std(a,1,1),1,2));
end

% Training set
trX=cat(1,cat(2,m1,s1),cat(2,m2,s2));
trY=cat(1,zeros(size(r_t,4),1,'logical'),ones(size(nr_t,4),1,'logical'));

% Get features
m1=zeros(size(r_v,4),3);
s1=zeros(size(r_v,4),3);
for i=1:size(r_v,4)
    a=single(r_v(:,:,:,i));
    m1(i,:)=squeeze(mean(mean(a,1),2));
    s1(i,:)=squeeze(std(std(a,1,1),1,2));
end

m2=zeros(size(nr_v,4),3);
s2=zeros(size(nr_v,4),3);
for i=1:size(nr_v,4)
    a=single(nr_v(:,:,:,i));
    m2(i,:)=squeeze(mean(mean(a,1),2));
    s2(i,:)=squeeze(std(std(a,1,1),1,2));
end

% Validation set
vlX=cat(1,cat(2,m1,s1),cat(2,m2,s2));
vlY=cat(1,zeros(size(r_v,4),1,'logical'),ones(size(nr_v,4),1,'logical'));

% Model
[trIdx,C]=kmeans(trX,2);

% Acc
tr_acc=trIdx-double(trY);
tr_acc=sum(tr_acc==1)./length(tr_acc);
[~,vlIdx] = pdist2(C,vlX,'euclidean','Smallest',1);
vl_acc=vlIdx-double(vlY)';
vl_acc=sum(vl_acc==1)./length(vl_acc);

