% This code will reproduce the results of the proposed method in the paper
% "Compressed MRI reconstruction exploiting a rotation-invariant
% total variation descretization".
% Preprint at arXiv and under review at Magnetic Resonance Imaging.
% Coded by Erfan Ebrahim Esfahani.
% Feel free to ask any questions or report bugs in the code via
% email: ebrahim.esfahani@ut.ac.ir
% Non-profit, non-commercial, scientific and personal use
% of the code is welcome.
%%           Table 1           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters
addpath(genpath('./Utils'));

maxit = 100;
n = 128;
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

%% Load Data (select dataset and mask according to Table1)
%load Spiral_10_128;     %spiral 10 mask
%load Spiral_20_128;     %spiral 20 mask
load Spiral_30_128;      %spiral 30 mask
mask = Q1;         
%load brain_data5        %brain dataset (50 slices)
load knee_data           %knee dasetset (50 slices)
Size = size(X,3);
SNR = zeros(1,Size);
SSIM = zeros(1,Size);
HFEN = zeros(1,Size);
for i =1:Size

x0 = abs(X(:,:,i));
x0 = imresize(x0, [n m]);
x0 = max(0,real(x0));
figure(1);imshow((x0));title('Ground Truth');

b = mask.*fft2c(x0);
zf = ifft2c(b);


% Proposed solution
out = solver_RITV (maxit,b,mask,x0,param);

SNR_proposed = out.SNR;
SSIM_proposed = out.SSIM;
HFEN_proposed = out.HFEN;

SNR(i) = SNR_proposed(end);
SSIM(i) = SSIM_proposed(end);
HFEN(i) = HFEN_proposed(end);
end
 %out.SamplingRate
mean_SNR = mean(SNR)
std_SNR = std(SNR)
mean_SSIM = mean(SSIM)
std_SSIM = std(SSIM)
mean_HFEN = mean(HFEN)
std_HFEN = std(HFEN)