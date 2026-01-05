% This code will reproduce the results of the proposed method in the paper
% "Compressed MRI reconstruction exploiting a rotation-invariant
% total variation descretization".
% Preprint in arXiv and under review at Magnetic Resonance Imaging.
% Coded by Erfan Ebrahim Esfahani.
% Feel free to ask any questions or report bugs in the code via
% email: ebrahim.esfahani@ut.ac.ir
% Non-profit, non-commercial, scientific and personal use
% of the code is welcome.
%%    Figure 6 (TV & TGV)   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Fig_6_TV_TGV

addpath(genpath('./Utils'));
global m n 
n = 256;
m = n;
rotationDegree = 90;
maxit = 500;
sigmaTGV = 1/sqrt(12);
tauTGV = sigmaTGV;
sigmaTVW = 1/sqrt(10);
tauTVW = sigmaTVW;
lambda = 8e-5;
alpha0 = 2;
alpha1 = 1;
alpha = 0.001;
x1 = 20;
x2 = 100;
y1 = 106;
y2 = 150;

%% Load Data
load brain_data5
x0 = X(:,:,35);
x0 = abs(x0);
x0 = imresize(x0, [n m]);
x0 = max(0,real(x0));
x00 = x0;
figure(1);imshow((x0));title('Ground Truth');
load Cart20
b = mask.*fft2c(x0);
mask00 = mask;
zf = ifft2c(b);
%% operators
R =@(x) mask.*fft2c(x);
R_adj = @(r) ifft2c(mask.*r);
prox2_sigma = @(r,lambda,sigma) r/(lambda*sigma+1);
mysnr = @(x,x0) 20*log10(norm(x0,'fro')/norm(x-x0,'fro')); 
D = @(u) cat(3,dxp(u),dyp(u));
div_1 = @(p) dxm(p(:,:,1)) + dym(p(:,:,2)); 



%% solution by TV
u =zf;
u_tild = zeros(m,n);
p = D(u);
r = zeros(m,n);
s = zeros(m,n);
counter=0;
SNR_TVW = zeros(maxit,1);
SSIM_TVW = zeros(maxit,1);

HFENvec_TVW = zeros(maxit,1);
%%% Main Iterations
tic;
 for j=1:maxit
     counter = counter+1;
    p = projP(p + sigmaTVW*(D(u_tild)),alpha);
    r = prox2_sigma (r + sigmaTVW*(R(u_tild) - b),1,sigmaTVW);
    u_old = u;
    u = u + tauTVW*(div_1(p) - R_adj(r));
    u = max(0,real(u));
    u_tild = 2*u - u_old;
    
    x = u;
    SNR_TVW(counter)=mysnr(abs(x),abs(x0));
    HFENvec_TVW(counter)= HFEN(abs(x),abs(x0));
    SSIM_TVW(counter)=ssim(abs(x),abs(x0));
   
fprintf('iteration= %d\n',counter);   
 end

qq1 = insertText(  [ u] ,[1,1], ...
  ['SNR: ' num2str(SNR_TVW(end),'%0.2f')...
    '   SSIM: ' num2str(SSIM_TVW(end),'%0.2f')...
  '   HFEN: ' num2str(HFENvec_TVW(end),'%0.2f')],'FontSize',14,'BoxColor','white');

u_upright = u;
figure(1); imshow([x0 mask]);
%% solution by TV rotated
x0 = imrotate(x00,rotationDegree);
mask = imrotate(mask00,rotationDegree);
b = mask.*fft2c(x0);
zf = ifft2c(b);

u =zf;
u_tild = zeros(m,n);
p = D(u);
r = zeros(m,n);
counter=0;
SNR_TVW = zeros(maxit,1);
SSIM_TVW = zeros(maxit,1);
HFENvec_TVW = zeros(maxit,1);
%%% Main Iterations
tic;
 for j=1:maxit
     counter = counter+1;
    p = projP(p + sigmaTVW*(D(u_tild)),alpha);
    r = prox2_sigma (r + sigmaTVW*(R(u_tild) - b),1,sigmaTVW);
    u_old = u;
    u = u + tauTVW*(div_1(p) - R_adj(r));
    u = max(0,real(u));
    u_tild = 2*u - u_old;
    
    x = u;
    SNR_TVW(counter)=mysnr(abs(x),abs(x0));
    HFENvec_TVW(counter)= HFEN(abs(x),x0);
    SSIM_TVW(counter)=ssim(abs(x),abs(x0));
    
  
fprintf('iteration= %d\n',counter);end


u_rot = u;
qq2 = insertText(  [ u_rot] ,[1,1], ...
  ['SNR: ' num2str(SNR_TVW(end),'%0.2f')...
    '   SSIM: ' num2str(SSIM_TVW(end),'%0.2f')...
  '   HFEN: ' num2str(HFENvec_TVW(end),'%0.2f')],'FontSize',14,'BoxColor','white');



qq1 = rgb2gray(qq1);
qq2 = rgb2gray(qq2);
u_rot_rot = imrotate(u_rot,-rotationDegree);
TV_rot = [qq1 qq2 imresize(u_upright(x1:x2,y1:y2),[n m/2]) imresize(u_rot_rot(x1:x2,y1:y2),[n m/2])];
figure(3);clf;imshow(TV_rot);title('TV');
%imwrite(TV_rot,'TV_rot.png');


figure(2); imshow([x0 mask]);
%% solution by TGV 
x0 = x00;
b = mask00.*fft2c(x0);
zf = ifft2c(b);

u = zf;        
u_tild = zeros(m,n);
v = D(u);
v_tild = zeros(m,n,2);
p = zeros(m,n,2);
q = zeros(m,n,3);
r = zeros(m,n);
counter=0;
SNR_TGV = zeros(maxit,1);
SSIM_TGV = zeros(maxit,1);
HFENvec_TGV = zeros(maxit,1);

% Main iterations
tic;
 for j=1:maxit
    counter = counter + 1;
    p = projP(p + sigmaTGV*(D(u_tild)-v_tild),alpha1);
    q = projQ(q + sigmaTGV*E(v_tild),alpha0);
    r = prox2_sigma(r + sigmaTGV*(R(u_tild) - b),lambda,sigmaTVW);
    u_old = u;
    u = u + tauTGV*(div_1(p) - R_adj(r));
    u = max(0,real(u));
    u_tild = 2*u - u_old;
    v_old = v;
    v = v + tauTGV*(p + div_2(q));
    v_tild = 2*v - v_old;

    SNR_TGV(counter)=mysnr(abs(u),abs(x0));
    HFENvec_TGV(counter)= HFEN(abs(u),abs(x0));
    SSIM_TGV(counter)=ssim(abs(u),abs(x0));
    
fprintf('iteration= %d\n',counter); end

 qq1 = insertText(  [ u] ,[1,1], ...
  ['SNR: ' num2str(SNR_TGV(end),'%0.2f')...
    '   SSIM: ' num2str(SSIM_TGV(end),'%0.2f')...
  '   HFEN: ' num2str(HFENvec_TGV(end),'%0.2f')],'FontSize',14,'BoxColor','white');

 
u_upright = u;
% imwrite([x0 mask00],'orietation1.png');

%% solution by TGV rotated
x0 = imrotate(x00,rotationDegree);
mask = imrotate(mask00,rotationDegree);
b = mask.*fft2c(x0);
zf = ifft2c(b);


u = zf;        
u_tild = zeros(m,n);
v = D(u);
v_tild = zeros(m,n,2);
p = zeros(m,n,2);
q = zeros(m,n,3);
r = zeros(m,n);
counter=0;
SNR_TGV = zeros(maxit,1);
SSIM_TGV = zeros(maxit,1);
HFENvec_TGV = zeros(maxit,1);

% Main iterations
tic;
 for j=1:maxit
    counter = counter + 1;
    p = projP(p + sigmaTGV*(D(u_tild)-v_tild),alpha1);
    q = projQ(q + sigmaTGV*E(v_tild),alpha0);
    r = prox2_sigma(r + sigmaTGV*(R(u_tild) - b),lambda,sigmaTVW);
    u_old = u;
    u = u + tauTGV*(div_1(p) - R_adj(r));
    u = max(0,real(u));
    u_tild = 2*u - u_old;
    v_old = v;
    v = v + tauTGV*(p + div_2(q));
    v_tild = 2*v - v_old;

    SNR_TGV(counter)=mysnr(abs(u),abs(x0));
    HFENvec_TGV(counter)= HFEN(abs(u),abs(x0));
    SSIM_TGV(counter)=ssim(abs(u),abs(x0));
  
fprintf('iteration= %d\n',counter); end
 u_rot = u;


qq2 = insertText(  [ u_rot] ,[1,1], ...
  ['SNR: ' num2str(SNR_TGV(end),'%0.2f')...
    '   SSIM: ' num2str(SSIM_TGV(end),'%0.2f')...
  '   HFEN: ' num2str(HFENvec_TGV(end),'%0.2f')],'FontSize',14,'BoxColor','white');


qq1 = rgb2gray(qq1);
qq2 = rgb2gray(qq2);
u_rot_rot = imrotate(u_rot,-rotationDegree);
TGV_rot = [qq1 qq2 imresize(u_upright(x1:x2,y1:y2),[n m/2]) imresize(u_rot_rot(x1:x2,y1:y2),[n m/2])];
figure(330);clf;imshow(TGV_rot);title('TGV');
% imwrite(TGV_rot,'TGV_rot.png');
% imwrite([x0 mask],'orientation2.png');

function z = E(p)
global m n
z = zeros(m,n,3);
z(:,:,1) = dxm(p(:,:,1));
z(:,:,2) = dym(p(:,:,2));
z(:,:,3) = (dym(p(:,:,1)) + dxm(p(:,:,2)))/2;

function r = div_2(z)
global m n
r = zeros(m,n,2);
r(:,:,1) = dxp(z(:,:,1)) + dyp(z(:,:,3));
r(:,:,2) = dxp(z(:,:,3)) + dyp(z(:,:,2));

function u = projC(u,c)
absu = norm(u,'fro');
denom = max(1,absu/c);
u = u./denom;

% function x = projC2(x,bb)
% a = x(:);
% Pbox = min(max(a,0),255);
% aTPbox = a'*Pbox;
% if aTPbox <= bb
%     x = Pbox;
% else
%     lstar = ;
%     x = min(max(a*(1-mustar),0),255);
% end

function p = projP(p,alpha1)

  absp = sqrt(abs(p(:,:,1)).^2 + abs(p(:,:,2)).^2);
  denom = max(1,absp/alpha1);
  p(:,:,1) = p(:,:,1)./denom;
  p(:,:,2) = p(:,:,2)./denom;  

function q = projQ(q,alpha0)
  absq = sqrt(abs(q(:,:,1)).^2 + abs(q(:,:,2)).^2 + 2*abs(q(:,:,3)).^2);
  denom = max(1,absq/alpha0);
  q(:,:,1) = q(:,:,1)./denom;
  q(:,:,2) = q(:,:,2)./denom;
  q(:,:,3) = q(:,:,3)./denom;
  
    function HFEN = HFEN(u,x0)
  LoG_filter = fspecial('log',[15 15], 1.5);
LoG0 = imfilter(x0, LoG_filter,'symmetric', 'conv'); normLoG0 = norm(LoG0,'fro');
HFEN = norm(imfilter(u, LoG_filter, 'symmetric', 'conv') - LoG0,'fro')/normLoG0;
