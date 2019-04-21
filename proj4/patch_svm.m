r_t=h5read('im2/river.h5','/patches');
nr_t=h5read('im2/not_river.h5','/patches');
r_v=h5read('im1/river.h5','/patches');
nr_v=h5read('im1/not_river.h5','/patches');

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
trY=cat(1,ones(size(r_t,4),1,'logical'),zeros(size(nr_t,4),1,'logical'));

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
vlY=cat(1,ones(size(r_v,4),1,'logical'),zeros(size(nr_v,4),1,'logical'));

% Train models
r_svm=fitcsvm(trX,trY,'KernelFunction','rbf');
l_svm=fitcsvm(trX,trY,'KernelFunction','linear');

% Compute accuracies
[r_label_tr,~]=predict(r_svm,trX);
[l_label_tr,~]=predict(l_svm,trX);
[r_label_vl,~]=predict(r_svm,vlX);
[l_label_vl,~]=predict(l_svm,vlX);
tr_acc_r=sum(~xor(r_label_tr,trY))/size(trY,1);
tr_acc_l=sum(~xor(l_label_tr,trY))/size(trY,1);
vl_acc_r=sum(~xor(r_label_vl,vlY))/size(vlY,1);
vl_acc_l=sum(~xor(l_label_vl,vlY))/size(vlY,1);