% rng(1);
% 
% r_t=h5read('im1/river.h5','/patches');
% nr_t=h5read('im1/not_river.h5','/patches');
% r_v=h5read('im2/river.h5','/patches');
% nr_v=h5read('im2/not_river.h5','/patches');
% 
% Get features
% m1=zeros(size(r_t,4),3);
% s1=zeros(size(r_t,4),3);
% for i=1:size(r_t,4)
%     a=single(r_t(:,:,:,i));
%     a=a(25:40,25:40,:);
%     m1(i,:)=squeeze(mean(mean(a,1),2));
%     s1(i,:)=squeeze(std(std(a,1,1),1,2));
% end
% 
% m2=zeros(size(nr_t,4),3);
% s2=zeros(size(nr_t,4),3);
% for i=1:size(nr_t,4)
%     a=single(nr_t(:,:,:,i));
%     a=a(25:40,25:40,:);
%     m2(i,:)=squeeze(mean(mean(a,1),2));
%     s2(i,:)=squeeze(std(std(a,1,1),1,2));
% end
% 
% Training set
% trX=cat(1,cat(2,m1,s1),cat(2,m2,s2));
% trY=cat(1,ones(size(r_t,4),1,'logical'),zeros(size(nr_t,4),1,'logical'));
% 
% Get features
% m1=zeros(size(r_v,4),3);
% s1=zeros(size(r_v,4),3);
% for i=1:size(r_v,4)
%     a=single(r_v(:,:,:,i));
%     a=a(25:40,25:40,:);
%     m1(i,:)=squeeze(mean(mean(a,1),2));
%     s1(i,:)=squeeze(std(std(a,1,1),1,2));
% end
% 
% m2=zeros(size(nr_v,4),3);
% s2=zeros(size(nr_v,4),3);
% for i=1:size(nr_v,4)
%     a=single(nr_v(:,:,:,i));
%     a=a(25:40,25:40,:);
%     m2(i,:)=squeeze(mean(mean(a,1),2));
%     s2(i,:)=squeeze(std(std(a,1,1),1,2));
% end
% 
% Validation set
% vlX=cat(1,cat(2,m1,s1),cat(2,m2,s2));
% vlY=cat(1,ones(size(r_v,4),1,'logical'),zeros(size(nr_v,4),1,'logical'));
% 
% trX=cat(1,trX,vlX);
% trY=cat(1,trY,vlY);
% r=randperm(size(trX,1));
% trX=trX(r(1:5000),:);
% trY=trY(r(1:5000),:);
% 
% Model
% l_svm=fitcsvm(trX,trY,'KernelFunction','linear');

% Validate on external image
im=imread('./imt/image.JPG');
mask=imread('./imt/image.png')';
teX=zeros(length(1:16:size(im,1))*length(1:16:size(im,2)),6,'single');
teY=zeros(length(1:16:size(im,1))*length(1:16:size(im,2)),1,'logical');
mm=zeros(size(mask));
a=1;
for i=1:16:size(im,1)
    for j=1:16:size(im,2)
        p=single(im(i:i+15,j:j+15,:));
        m=squeeze(mean(mean(p,1),2));
        s=squeeze(std(std(p,1,1),1,2));
        teX(a,:)=cat(1,m,s)';
        gt=round(mean(mean(mask(i:i+15,j:j+15))));
        teY(a)=gt;
        a=a+1;
    end
end

% Predict
pr=predict(l_svm,teX);

% Segment
a=1;
for i=1:16:size(im,1)
    for j=1:16:size(im,2)
        mm(i:i+15,j:j+15)=pr(a);
        a=a+1;
    end
end
        
        
        
