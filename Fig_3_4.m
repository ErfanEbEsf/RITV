% This code will reproduce the results of the proposed method in the paper
% "Compressed MRI reconstruction exploiting a rotation-invariant
% total variation descretization".
% Preprint in arXiv and under review at Magnetic Resonance Imaging.
% Coded by Erfan Ebrahim Esfahani.
% Feel free to ask any questions or report bugs in the code via
% email: ebrahim.esfahani@ut.ac.ir
% Non-profit, non-commercial, scientific and personal use
% of the code is welcome.

%%          Figures 3 & 4          %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters
addpath(genpath('./Utils'));

maxit = 300;%Should be common between Malitsky-Pock & GADMM (due to BM3D settings) 
n = 2*128;
m = n;

param.mu = 0.7;
param.delta = 0.99;
param.beta = 0.016;
param.theta = 1;
param.tau = 16/sqrt(14);
param.eta = 0.2;
param.lambda_inv = 7000;
mu = 1e+4; %GADMM mu
% BM3D params.
constt=200; ratt=200; 
final_noise=constt/ratt; %sigma/2; %0.1
param.sigma_bm3d=logspace(log10(constt),log10(final_noise),maxit); 

%% Load Data
x0 = phantom;
x0 = abs(x0);
x0 = imresize(x0, [n m]);
x0 = max(0,real(x0));
figure(1);imshow((x0));title('Ground Truth');
mask = LineMaskLimitedAngle(12,n,pi,0); %Radial mask in Fig. 3.
b = mask.*fft2c(x0);
zf = ifft2c(b);

figure(2); imshow(mask);

%% Proposed solution
out = solver_RITV (maxit,b,mask,x0,param);

time_MP = out.Runtime
u = out.sol;
SNR_proposed = out.SNR;
SSIM_proposed = out.SSIM;
HFEN_proposed = out.HFEN;

qq = insertText(  [u] ,[1,1], ...
  ['SNR: ' num2str(SNR_proposed(end),'%0.2f')...
    '   SSIM: ' num2str(SSIM_proposed(end),'%0.2f')...
  '   HFEN: ' num2str(HFEN_proposed(end),'%0.2f')],'FontSize',14,'BoxColor','white');
qq = rgb2gray(qq);
figure(3);clf;imshow(qq);title('Malitsky-pock');


%% Graph of HFEN
mm = maxit;
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
legend({'Malitsky-pock','ZF'}); 
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
legend({'Malitsky-pock','ZF'}); 
grid on


Fig_3_4_GADMM(mu,maxit);
