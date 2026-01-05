% This code will reproduce the results of the proposed method in the paper
% "Compressed MRI reconstruction exploiting a rotation-invariant
% total variation descretization".
% Preprint in arXiv and under review at Magnetic Resonance Imaging.
% Coded by Erfan Ebrahim Esfahani.
% Feel free to ask any questions or report bugs in the code via
% email: ebrahim.esfahani@ut.ac.ir
% Non-profit, non-commercial, scientific and personal use
% of the code is welcome.
%%   Figure 6 (RITV result)  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters
addpath(genpath('./Utils'));

maxit = 200;
n = 256;
m = n;
rotationDegree = 90;

%zoom
x1 = 20;
x2 = 100;
y1 = 106;
y2 = 150;

param.mu = 0.7;
param.delta = 0.99;
param.beta =1.7e-5;
param.theta = 1;
param.tau = 16/sqrt(14);

param.eta = 0;
param.lambda_inv = 7000;

% BM3D params.
constt=200; ratt=200; 
final_noise=constt/ratt; %sigma/2; %0.1
param.sigma_bm3d=logspace(log10(constt),log10(final_noise),maxit); 

%% Load Data
load brain_data5
x0 = X(:,:,35);
x0 = abs(x0);
x0 = imresize(x0, [n m]);
x0 = max(0,real(x0));
% x0 = imrotate(x0,90);
figure(1);imshow((x0));title('Ground Truth');
load Cart20
b = mask.*fft2c(x0);




%% RITV solution
out = solver_RITV_only (maxit,b,mask,x0,param);


u = out.sol;
SNR_proposed = out.SNR;
SSIM_proposed = out.SSIM;
HFEN_proposed = out.HFEN;


qq = insertText(  [u] ,[1,1], ...
  ['SNR: ' num2str(SNR_proposed(end),'%0.2f')...
    '   SSIM: ' num2str(SSIM_proposed(end),'%0.2f')...
  '   HFEN: ' num2str(HFEN_proposed(end),'%0.2f')],'FontSize',14,'BoxColor','white');
qq1 = rgb2gray(qq);

u_upright = u;


%% RITV rotated solution
x0 = imrotate(x0,rotationDegree);
mask = imrotate(mask,rotationDegree);
b = mask.*fft2c(x0);
zf = ifft2c(b);
out = solver_RITV (maxit,b,mask,x0,param);

%zoom

u_rot = out.sol;


SNR_proposed = out.SNR;
SSIM_proposed = out.SSIM;
HFEN_proposed = out.HFEN;



%%%%%%%%%%%%%%%

qq2 = insertText(  [u_rot] ,[1,1], ...
  ['SNR: ' num2str(SNR_proposed(end),'%0.2f')...
    '   SSIM: ' num2str(SSIM_proposed(end),'%0.2f')...
  '   HFEN: ' num2str(HFEN_proposed(end),'%0.2f')],'FontSize',14,'BoxColor','white');
qq2 = rgb2gray(qq2);


u_rot_rot = imrotate(u_rot,-rotationDegree);
RITV_rot = [qq1 qq2 imresize(u_upright(x1:x2,y1:y2),[n m/2]) imresize(u_rot_rot(x1:x2,y1:y2),[n m/2])];
figure(25);clf;imshow(RITV_rot);title('RITV');
% imwrite(RITV_rot,'RITV_rot.png');

