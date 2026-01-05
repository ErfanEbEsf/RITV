% This code will reproduce the results of the proposed method in the paper
% "Compressed MRI reconstruction exploiting a rotation-invariant
% total variation descretization".
% Preprint in arXiv and under review at Magnetic Resonance Imaging.
% Coded by Erfan Ebrahim Esfahani.
% Feel free to ask any questions or report bugs in the code via
% email: ebrahim.esfahani@ut.ac.ir
% Non-profit, non-commercial, scientific and personal use
% of the code is welcome.
%%           Table 3           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters
addpath(genpath('./Utils'));

maxit = 100;
n = 2*128;
m = n;
param.mu = 0.7;
param.delta = 0.99;
%param.beta = 0.016;
param.theta = 1;
param.tau = 16/sqrt(14);
param.eta = 0.2;
param.lambda_inv = 7000;

% BM3D params.
constt=200; ratt=200; 
final_noise=constt/ratt; %sigma/2; %0.1
param.sigma_bm3d=logspace(log10(constt),log10(final_noise),maxit); 

%% Load Data (select dataset and mask according to Table1)
load Spiral; mask = Q1; %spiral 16 mask
figure(99);imshow(mask);    
load knee_data          
X = X(:,:,31:50);
Size = size(X,3);
SNR_proposed_max = zeros(1,Size);
SNR_proposed_final = zeros(1,Size);
SNR_RITV_only_max = zeros(1,Size);
SNR_RITV_only_final = zeros(1,Size);
SNR_BM3D_only_max = zeros(1,Size);
SNR_BM3D_only_final = zeros(1,Size);
for i =1:Size

x0 = abs(X(:,:,i));
x0 = imresize(x0, [n m]);
x0 = max(0,real(x0));
figure(100);imshow((x0));title('Ground Truth');

b = mask.*fft2c(x0);
zf = ifft2c(b);


% Proposed solution
param.beta = 0.016;
out = solver_RITV (maxit,b,mask,x0,param);

SNR_proposed = out.SNR;
SNR_proposed_final(i) = SNR_proposed(end);
SNR_proposed_max(i) = max(SNR_proposed);

% RITV-only solution
param.beta = 1e-5;
out = solver_RITV_only (maxit,b,mask,x0,param);

SNR_RITV_only = out.SNR;
SNR_RITV_only_final(i) = SNR_RITV_only(end);
SNR_RITV_only_max(i) = max(SNR_RITV_only);

% BM3D_only solution
param.beta = 0.016;
out = solver_BM3D_only (maxit,b,mask,x0,param);

SNR_BM3D_only = out.SNR;
SNR_BM3D_only_final(i) = SNR_BM3D_only(end);
SNR_BM3D_only_max(i) = max(SNR_BM3D_only);


%% Graph of SNR
figure(i); clf;
h=plot(1:1:maxit,SNR_proposed,'k'); 
set(h,'LineWidth',2);
hold on;
h=plot(1:1:maxit,SNR_BM3D_only,'green'); 
set(h,'LineWidth',2);
hold on;
h=plot(1:1:maxit,SNR_RITV_only,'red'); 
set(h,'LineWidth',2);
hold on;
set(gca,'FontSize',14);
h=xlabel('Iterations');
set(h,'FontSize',14);
h=ylabel('SNR');
set(h,'FontSize',14);
legend({'MP:BM3D+RITV (Algorithm 1)','MP:BM3D alone','MP:RITV alone'},'Location','SouthEast'); 
grid on

end

mean_SNR_proposed_final = mean(SNR_proposed_final)
mean_SNR_proposed_max = mean(SNR_proposed_max)


mean_SNR_RITV_only_final = mean(SNR_RITV_only_final)
mean_SNR_RITV_only_max = mean(SNR_RITV_only_max)

mean_SNR_BM3D_only_final = mean(SNR_BM3D_only_final)
mean_SNR_BM3D_only_max = mean(SNR_BM3D_only_max)

proposed_SNR_drop =-mean_SNR_proposed_final + mean_SNR_proposed_max
RITV_only_SNR_drop = -mean_SNR_RITV_only_final + mean_SNR_RITV_only_max
BM3D_only_SNR_drop = -mean_SNR_BM3D_only_final + mean_SNR_BM3D_only_max



