rng(1);

ps=64;
st=8;
n=10000;

if ~(exist('im'))
    im=imread('data/train2/IOH_03_R13C2.tif');
    m=imread('data/train2/IOH_03_R13C12_mask.png');
end

cols=1:st:size(im,1)-ps-st;
rows=1:st:size(im,2)-ps-st;
cols=cols(randperm(length(cols)));
rows=rows(randperm(length(rows)));

r=1;
t=1;
river=zeros(ps,ps,3,n,'uint8');
tree=zeros(ps,ps,3,n,'uint8');
for i=1:length(cols)
    for j=1:length(rows)
        p=im(rows(j):rows(j)+ps-1,cols(i):cols(i)+ps-1,:);
        b=m(rows(j):rows(j)+ps-1,cols(i):cols(i)+ps-1,:);
        if sum(b(:))>(0.95*(ps*ps)) && r<=n
            river(:,:,:,r)=p;
            r=r+1;
            r
        end
        if sum(b(:))<(0.01*(ps*ps)) && t<=n
            tree(:,:,:,t)=p;
            t=t+1;
        end
        if t>n && r>n
            break;
        end
    end
end

h5create('river.h5','/patches',size(river),'Datatype','uint8');
h5write('river.h5','/patches',river);
h5create('not_river.h5','/patches',size(tree),'Datatype','uint8');
h5write('not_river.h5','/patches',tree);
