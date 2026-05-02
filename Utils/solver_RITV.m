% This code will reproduce the results of the proposed method in the paper
% "Compressed MRI reconstruction exploiting a rotation-invariant
% total variation descretization".
% Preprint submitted to arXiv.
% Written by Erfan Ebrahim Esfahani.
% Feel free to ask any questions or report bugs in the code via
% email: ebrahim.esfahani@ut.ac.ir
% Non-profit, non-commercial, scientific and personal use
% of the code is welcome.

function out = solver_RITV (maxit,b,mask,x0,param)

% Input data:
% maxit: Number of iterations (e.g. 100)
% b: Undersampled k-sapce 
% mask: Undersampling pattern
% x0: Ground truth.

%% Setting parameters
mu = param.mu ;
delta = param.delta ;
beta = param.beta ;
theta = param.theta ;
tau = param.tau ;
eta = param.eta ;
lambda_inv = param.lambda_inv ;
sigma_bm3d = param.sigma_bm3d;
[m,n] = size(b);
 

%% Zero-filling solution
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
prox2 = @(r,sigma) (r-sigma*b)/(sigma+1);

%% Initialization

u = zf;   
v = zeros(m,n,2,4);
r = zeros(m,n);
h = zeros(m,n,2);
SNR_proposed = zeros(maxit,1);
SSIM_proposed = zeros(maxit,1);
HFEN_proposed = zeros(maxit,1);

%% Main iterations
tic;
for mm = 1:maxit
    
    u_old = u;
    KTy_u = -DT(h) + R_adj(r); 
    u = u - tau*(KTy_u);
    [~,u] = BM3D_MRI_denoise2(eta*tau,1,u,sigma_bm3d(mm));  % BM3D denoising
%      u = max(0,real(u));
    v_old = v;
    KTy_v = L(h);
    v = prox_alphatauG(v-tau*(KTy_v),tau/lambda_inv);
    tau_old = tau;
    tau = tau_old*sqrt(1+theta);
    r_old = r;
    h_old = h;
    while 1
        theta= tau/tau_old;
        ubar = u + theta*(u - u_old);
        vbar = v + theta*(v - v_old);
        betatau = beta*tau;
        r = prox2(r_old + betatau*R(ubar),betatau);
        h = h_old + betatau*(LT(vbar) - D(ubar));
    if sqrt(beta)*tau*normX(-DT(h)+R_adj(r) ...
            -KTy_u,L(h)-KTy_v) <= delta*normY(h-h_old,r-r_old)
      break;
    else
        tau = tau*mu;
    end
    end
    SNR_proposed(mm)=mysnr( abs(u), abs (x0));
    HFEN_proposed(mm)= HFEN(abs(u));
    SSIM_proposed(mm)=ssim( abs(u), abs(x0));
    fprintf('iteration= %d   SNR= %.2f   SSIM= %.4f   HFEN= %.4f\n',...
    mm,SNR_proposed(mm),SSIM_proposed(mm),HFEN_proposed(mm));
end
runtime = toc;


%% Outputs
out.sol = abs(u);
out.IterationsCount = mm;
out.SamplingRate = numel(find(mask))/numel(mask);
out.Runtime = runtime;
out.SNR = SNR_proposed;
out.SSIM = SSIM_proposed;
out.HFEN = HFEN_proposed;
out.SNR0 = mysnr(abs(zf),abs(x0))*ones(maxit,1);
out.SSIM0 = ssim(abs(zf),abs(x0))*ones(maxit,1);
out.HFEN0 = HFEN(zf)*ones(maxit,1);


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
    
    function a = normX(u,v)
        u = abs(u(:)); v = abs(v(:));
        a = sqrt(sum(u.^2) + sum(v.^2));
%         u = abs(u);
%         v = abs(v);
%         norm2v = 0;
%         for i=1:size(v,4)
%            norm2v = norm2v + sum(sum(sum(v(:,:,:,i).^2,3)));
%         end
%         a = sqrt(sum(sum(u.^2)) + norm2v);  
        
        
        function a = normY(h,r)
    h = abs(h(:)); r = abs(r(:));
    a = sqrt(sum(r.^2) + sum(h.^2));
%             h = abs(h);
%             r = abs(r);
%             a = sqrt(sum(sum(sum(h.^2,3))) + sum(sum(r.^2))) ;

         
 