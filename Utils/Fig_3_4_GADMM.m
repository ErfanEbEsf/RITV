

function out = Fig_3_4_GADMM
addpath(genpath('./Utils'));

%% Setting parameters
global b m n D R;
maxit = 300;
mu = 1e+6;
tau = 1/(mu+8);
gamma = 1/4;
eta = 0.2;
lambda =  1/7000;
n = 2*128;
m = n;

%% zoom
x1 = 70;
x2 = 170;
y1 = 120;
y2 = 220;
%%
constt=200; ratt=200; 
final_noise=constt/ratt; 
sigma_bm3d=logspace(log10(constt),log10(final_noise),maxit); 

%% Loadibg data

x0 = phantom(n);
figure(1);imshow((x0));title('Ground Truth');
mask = LineMaskLimitedAngle(12,n,pi,0);
figure(2);imshow(mask);title('Sampling Mask');
% imwrite(mask,'SM.png', 'png');
b = mask.*fft2c(x0);
zf = ifft2c(b);



%% Defining operators
R =@(x) mask.*fft2c(x);
R_adj = @(r) ifft2c(mask.*r);
mysnr = @(x,x0) 20*log10(norm(x0,'fro')/norm(x-x0,'fro')); 
LoG_filter = fspecial('log',[15 15], 1.5);
LoG0 = imfilter(x0, LoG_filter,'symmetric', 'conv'); normLoG0 = norm(LoG0,'fro');
HFEN = @(u) norm(imfilter(u, LoG_filter, 'symmetric', 'conv') - LoG0,'fro')/normLoG0;
D = @(x) cat(3,[diff(x,1,1);zeros(1,size(x,2))],[diff(x,1,2) zeros(size(x,1),1)]);
DT = @(u) -[u(1,:,1);diff(u(:,:,1),1,1)]-[u(:,1,2) diff(u(:,:,2),1,2)];
prox_alphatauG= @(t,sigma) t-bsxfun(@rdivide, t,...
max(sqrt(sum(t.^2,3))/(sigma),1));

%% solution by GADMM

u = zf;   
v = zeros(m,n,2,4);
ksi = zeros(m,n,2);
SNR_proposed = zeros(maxit,1);
SSIM_proposed = zeros(maxit,1);
HFEN_proposed = zeros(maxit,1);
% Main iterations
% figure(1);
tic;
for mm = 1:maxit
    
    tmp = -LT(v)+mu*ksi;
    u = u-tau*DT(D(u)+tmp)-tau*mu*R_adj(R(u)-b);
    [~,u] = BM3D_MRI_denoise2(eta*tau*mu,1,u,sigma_bm3d(mm));  % BM3D denoising
    v = v + gamma*L(D(u)+tmp);
    v = prox_alphatauG(v,gamma*mu*lambda);
    ksi = ksi + (1/mu)*(D(u)-LT(v));
    
    SNR_proposed(mm)=mysnr( abs(u), abs (x0));
    HFEN_proposed(mm)= HFEN(abs(u));
    SSIM_proposed(mm)=ssim( abs(u), abs(x0));
    fprintf('iteration= %d   SNR= %.2f   SSIM= %.4f   HFEN= %.4f\n',...
    mm,SNR_proposed(mm),SSIM_proposed(mm),HFEN_proposed(mm));
end
time_TGVW = toc


qq = insertText(  [u] ,[1,1], ...
  ['SNR: ' num2str(SNR_proposed(mm),'%0.2f')...
    '   SSIM: ' num2str(SSIM_proposed(mm),'%0.2f')...
  '   HFEN: ' num2str(HFEN_proposed(mm),'%0.2f')],'FontSize',14,'BoxColor','white');

figure(3);clf;imshow(qq);title('GADMM');
% imwrite(rgb2gray(qq),'.\GADMM\SL_mu6.jpg','jpg');


% save('.\GADMM\SNR_SL_mu6.mat','SNR_proposed')
% save('.\GADMM\SSIM_SL_mu6.mat','SSIM_proposed')
% save('.\GADMM\HFEN_SL_mu6.mat','HFEN_proposed')


%% Graph of HFEN
HFENzf = HFEN(zf)*ones(mm,1);
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
legend({'GADMM','ZF'}); 
grid on

%% Graph of SSIM
SSIMzf = ssim(abs(zf),abs(x0))*ones(mm,1);
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
legend({'GADMM','ZF'}); 
grid on

%% Outputs
out.IterationsCount = mm;
out.SamplingRate = numel(find(mask))/numel(mask);

out.TGVW_Runtime = time_TGVW;

% out.SNR_zf = SNRzf(end);
out.TGVW_SNR = SNR_proposed(end);

out.SSIM_zf = SSIMzf(end);
out.TGVW_SSIM = SSIM_proposed(end);

out.HFEN_zf = HFENzf(end);
out.TGVW_HFEN = HFEN_proposed(end);


function t = L(u)
	t=zeros([size(u) 4]);
	t(:,:,1,1)=u(:,:,1); 
	t(1:end-1,2:end,2,1)=(u(2:end,1:end-1,2)+u(1:end-1,1:end-1,2)+...
		u(2:end,2:end,2)+u(1:end-1,2:end,2))/4;
	t(1:end-1,1,2,1)=(u(1:end-1,1,2)+u(2:end,1,2))/4;
	t(:,:,2,2)=u(:,:,2);
	t(2:end,1:end-1,1,2)=(u(2:end,1:end-1,1)+u(1:end-1,1:end-1,1)+...
		u(2:end,2:end,1)+u(1:end-1,2:end,1))/4;
	t(1,1:end-1,1,2)=(u(1,1:end-1,1)+u(1,2:end,1))/4;
	t(2:end,:,1,3) = (u(2:end,:,1)+u(1:end-1,:,1))/2;
	t(1,:,1,3) = u(1,:,1)/2;
	t(:,2:end,2,3) = (u(:,2:end,2)+u(:,1:end-1,2))/2;
	t(:,1,2,3) = u(:,1,2)/2;
	t(1:end-1,1:end-1,1,4) = (u(1:end-1,1:end-1,1)+u(1:end-1,2:end,1))/2;
	t(1:end-1,1:end-1,2,4) = (u(1:end-1,1:end-1,2)+u(2:end,1:end-1,2))/2;

function u = LT(t)
	[height,width,d,c]=size(t);
	u=zeros(height,width,2);
	u(1:end-1,2:end,1)=t(1:end-1,2:end,1,1)+(t(1:end-1,2:end,1,2)+...
		t(1:end-1,1:end-1,1,2)+t(2:end,2:end,1,2)+t(2:end,1:end-1,1,2))/4+...
		(t(1:end-1,2:end,1,3)+t(2:end,2:end,1,3))/2+...
		(t(1:end-1,1:end-1,1,4)+t(1:end-1,2:end,1,4))/2;
	u(1:end-1,1,1)=t(1:end-1,1,1,1)+(t(1:end-1,1,1,2)+t(2:end,1,1,2))/4+...
		(t(1:end-1,1,1,3)+t(2:end,1,1,3))/2+t(1:end-1,1,1,4)/2;
	u(2:end,1:end-1,2)=t(2:end,1:end-1,2,2)+(t(2:end,1:end-1,2,1)+...
		t(1:end-1,1:end-1,2,1)+t(2:end,2:end,2,1)+t(1:end-1,2:end,2,1))/4+...
		(t(2:end,1:end-1,2,3)+t(2:end,2:end,2,3))/2+...
		(t(1:end-1,1:end-1,2,4)+t(2:end,1:end-1,2,4))/2;
	u(1,1:end-1,2)=t(1,1:end-1,2,2)+(t(1,1:end-1,2,1)+t(1,2:end,2,1))/4+...
		(t(1,1:end-1,2,3)+t(1,2:end,2,3))/2+t(1,1:end-1,2,4)/2;
    