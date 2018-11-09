rootDir = '/home/a/Desktop/CZY/humanshape/';
fittedMeshsDir = [rootDir 'caesar-fitted-meshes/'];

%%load face point index
facesSM = '/home/a/Desktop/CZY/humanshape/fitting/facesShapeModel.mat';
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


meshes = dir([fittedMeshsDir '/*.mat']);
meshNames = {meshes.name};
girths=zeros(length(meshes):length(keyPointIndex));

for i = 1:length(meshes)
    load([fittedMeshsDir meshes(i).name],'points');
    points=points*rot;
    girths(i,:)=computeGirth(points,faces,keyPointIndex);
    i
    girths(i,:)
end
save([rootDir 'girths.mat'],'girths');
save([rootDir 'meshNames.mat'],'meshNames');
