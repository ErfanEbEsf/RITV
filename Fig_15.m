% This code will reproduce the results of the proposed method in the paper
% "Compressed MRI reconstruction exploiting a rotation-invariant
% total variation descretization".
% Preprint in arXiv and under review at Magnetic Resonance Imaging.
% Coded by Erfan Ebrahim Esfahani.
% Feel free to ask any questions or report bugs in the code via
% email: ebrahim.esfahani@ut.ac.ir
% Non-profit, non-commercial, scientific and personal use
% of the code is welcome.
%%      Figures 15      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters
addpath(genpath('./Utils'));

maxit = 100;
n = 2*128;
m = n;
rng('default');
param.mu = 0.7;
param.delta = 0.99;
param.beta = 0.1;
param.theta = 1;
param.tau = 16/sqrt(14);
param.eta = 30;
param.lambda_inv = 7000;

% BM3D params.
constt=200; ratt=200; 
final_noise=constt/ratt; %sigma/2; %0.1
param.sigma_bm3d=logspace(log10(constt),log10(final_noise),maxit); 

%% Load Data 
load Cart20   
load brain_data5
x0 = X(:,:,31);
x0 = abs(x0);
x0 = imresize(x0, [n m]);
x0 = max(0,real(x0));
figure(1);imshow((x0));title('Ground Truth');
b = mask.*fft2c(x0) + 0.1*randn(n,m);
zf = ifft2c(b);

disparityRange = [0 0.3];
figure(50)
map = abs(x0-zf);
imshow(map,disparityRange);
colormap(gca,jet) 
cb = colorbar; 
set(cb,'position',[.7955 .25 .02 .65])
set(cb,'YTick',[0:0.1:0.3]);


%% RITV solution
out = solver_RITV (maxit,b,mask,x0,param);

%zoom
x1 = 70;
x2 = 170;
y1 = 120;
y2 = 220;

u = out.sol;
SNR_proposed = out.SNR;
SSIM_proposed = out.SSIM;
HFEN_proposed = out.HFEN;

qq = insertText(  [u,imresize(u(x1:x2,y1:y2),[n,m]),ones(n)-abs(x0-u)] ,[1,1], ...
  ['SNR: ' num2str(SNR_proposed(end),'%0.2f')...
    '   SSIM: ' num2str(SSIM_proposed(end),'%0.2f')...
  '   HFEN: ' num2str(HFEN_proposed(end),'%0.2f')],'FontSize',14,'BoxColor','white');
qq = rgb2gray(qq);
figure(3);clf;imshow(qq);title('Proposed');

disparityRange = [0 0.3];
figure(5)
map = qq(:,513:end);
imshow(1-map,disparityRange);
colormap(gca,jet) 
cb = colorbar; 
set(cb,'position',[.7955 .25 .02 .65])
set(cb,'YTick',[0:0.1:0.3]);

