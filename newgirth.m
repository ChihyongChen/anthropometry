rootDir = '/develop/czy/PycharmProjects/humanshape/';

%%load face point index
facesSM = '/develop/czy/PycharmProjects/humanshape/fitting/facesShapeModel.mat';
load(facesSM,'faces');

rot=[-0.7071 0.7071 0;0.7071 0.7071 0;0 0 1;]

keyPointIndex=[
    %jiaohuai +-10
    1350 1317 1384;
    %xigai +-5
    1847 1783 1859;
    %datui sum/2
    298 0 0;
    %yao +-10
    23 5426 2291;
    %xiong +-0
    639 5705 2547;
    %jian
    4138 0 0;
    %gebo
    3157 2679 3251;
    %shouwan
    2679 2919 2935
    ];

load([rootDir 'newbody.mat'],'points');
points=points*rot;
girth=computeGirth(points,faces,keyPointIndex);

girth

size(points)
maxz = max(points(:,3));
minz = min(points(:,3));
height1 = maxz - minz;
height1


