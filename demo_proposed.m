% This code will reproduce the results of the proposed method in the paper
% "Compressed MRI reconstruction exploiting a rotation-invariant
% total variation descretization".
% Preprint in arXiv and under review at Magnetic Resonance Imaging.
% Coded by Erfan Ebrahim Esfahani.
% Feel free to ask any questions or report bugs in the code via
% email: ebrahim.esfahani@ut.ac.ir
% Non-profit, non-commercial, scientific and personal use
% of the code is welcome.
%%  Figures 10, 11, 13  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters
addpath(genpath('./Utils'));

maxit = 200;
n = 2*128;
m = n;

param.mu = 0.7;
param.delta = 0.99;
param.beta = 0.016;
param.theta = 1;
param.tau = 16/sqrt(14);
param.eta = 0.2;
param.lambda_inv = 7000;

% BM3D params.
constt=200; ratt=200; 
final_noise=constt/ratt; %sigma/2; %0.1
param.sigma_bm3d=logspace(log10(constt),log10(final_noise),maxit); 

%% Load Data 
load('Spiral.mat'); mask = (Q1);           %Fig. 10 
%mask = LineMaskLimitedAngle(44,n,pi,0);   %Fig. 11
%load Cart20                               %Fig. 13
load Head    %Fig. 10
%load Knee   %Fig. 11
%load Brain  %Fig. 13
x0 = abs(x0);
x0 = imresize(x0, [n m]);
x0 = max(0,real(x0));
figure(1);imshow((x0));title('Ground Truth');
b = mask.*fft2c(x0);
zf = ifft2c(b);

disparityRange = [0 0.3];
figure(50)
map = abs(x0-zf);
imshow(map,disparityRange);title('ZF error');
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
imshow(1-map,disparityRange);title('Proposed error');
colormap(gca,jet) 
cb = colorbar; 
set(cb,'position',[.7955 .25 .02 .65])
set(cb,'YTick',[0:0.1:0.3]);

%% Graph of SNR
mm = maxit;
SNRzf = out.SNR0;
figure(10); clf;
h=plot(1:1:mm,SNR_proposed,'k'); 
set(h,'LineWidth',2);
hold on;
h=plot(1:1:mm,SNRzf,'--'); 
set(h,'LineWidth',2);
set(gca,'FontSize',14);
h=xlabel('Iterations');
set(h,'FontSize',14);
h=ylabel('SNR');
set(h,'FontSize',14);
legend({'Proposed','ZF'}); 
grid on

%% Graph of HFEN
HFENzf = out.HFEN0;
figure(11); clf;
h=plot(1:1:mm,HFEN_proposed ,'k'); 
set(h,'LineWidth',2);
hold on;
h=plot(1:1:mm,HFENzf,'--'); 
set(h,'LineWidth',2);
set(gca,'FontSize',14);
h=xlabel('Iterations');
set(h,'FontSize',14);
h=ylabel('HFEN');
set(h,'FontSize',14);
legend({'Proposed','ZF'}); 
grid on

%% Graph of SSIM
SSIMzf = out.SSIM0;
figure(12); clf;
h=plot(1:1:mm,SSIM_proposed,'k'); 
set(h,'LineWidth',2);
hold on;
h=plot(1:1:mm,SSIMzf,'--'); 
set(h,'LineWidth',2);
set(gca,'FontSize',14);
h=xlabel('Iterations');
set(h,'FontSize',14);
h=ylabel('SSIM');
set(h,'FontSize',14);
legend({'Proposed','ZF'}); 
grid on

