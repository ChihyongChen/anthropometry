rootDir = '/home/a/Desktop/CZY/humanshape/';
fittedMeshsDir = [rootDir 'caesar-fitted-meshes/'];
csr4000a = [fittedMeshsDir 'csr4006a.mat'];
%csr4000a = [fittedMeshsDir 'CSR1717A.mat'];

load(csr4000a,'points');

expidx = 0;
p = expParams(expidx);
%show model
load(p.facesSM,'faces');
%clf;
%[X,Y] = meshgrid(-200:600:400,-300:700:400);
%Z = 0*X + 0*Y+5;
%fig = figure;
%surf(X,Y,Z);

hold on;
showmodel(points,faces,'g',[],0);
axis equal; view(45,22.5); %pause; 

         
    