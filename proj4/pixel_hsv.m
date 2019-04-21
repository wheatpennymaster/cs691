r_t=h5read('im2/river.h5','/patches');
nr_t=h5read('im2/not_river.h5','/patches');
r_v=h5read('im1/river.h5','/patches');
nr_v=h5read('im1/not_river.h5','/patches');

p=1;
ps=64;

% Get features
p1=zeros(size(r_t,4)*4,3,'single');
for i=1:size(r_t,4)
    a=r_t(:,:,:,i);
    a=single(rgb2hsv(a));
    a=a(ps/2-p+1:ps/2+p,ps/2-p+1:ps/2+p,:);
    g=size(a,1)*size(a,2);
    a=reshape(a,[g 3]);
    p1((i-1)*g+1:i*g,:)=a;
end

p0=zeros(size(nr_t,4)*4,3,'single');
for i=1:size(nr_t,4)
    a=nr_t(:,:,:,i);
    a=single(rgb2hsv(a));
    a=a(ps/2-p+1:ps/2+p,ps/2-p+1:ps/2+p,:);
    g=size(a,1)*size(a,2);
    a=reshape(a,[g 3]);
    p0((i-1)*g+1:i*g,:)=a;
end

% Training set
trX=cat(1,p1,p0);
trY=cat(1,ones(size(r_t,4)*g,1,'logical'),zeros(size(nr_t,4)*g,1,'logical'));

% Get features
p1=zeros(size(r_v,4)*4,3,'single');
for i=1:size(r_v,4)
    a=r_v(:,:,:,i);
    a=single(rgb2hsv(a));
    a=a(ps/2-p+1:ps/2+p,ps/2-p+1:ps/2+p,:);
    g=size(a,1)*size(a,2);
    a=reshape(a,[g 3]);
    p1((i-1)*g+1:i*g,:)=a;
end

p0=zeros(size(nr_v,4)*4,3,'single');
for i=1:size(nr_v,4)
    a=nr_v(:,:,:,i);
    a=single(rgb2hsv(a));
    a=a(ps/2-p+1:ps/2+p,ps/2-p+1:ps/2+p,:);
    g=size(a,1)*size(a,2);
    a=reshape(a,[g 3]);
    p0((i-1)*g+1:i*g,:)=a;
end

% Validation set
vlX=cat(1,p1,p0);
vlY=cat(1,ones(size(r_v,4)*g,1,'logical'),zeros(size(nr_v,4)*g,1,'logical'));

% Train model
m=fitcknn(trX,trY);

% Accuracy
[label_tr,~]=predict(m,trX);
[label_vl,~]=predict(m,vlX);
tr_acc=sum(~xor(label_tr,trY))/size(trY,1);
vl_acc=sum(~xor(label_vl,vlY))/size(vlY,1);