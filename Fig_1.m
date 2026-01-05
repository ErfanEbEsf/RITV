
function Fig_1
global D m n div_1
a = 13;
alpha1 = 1;
alpha0 = 2;
Nbiter = 200;
D = @(x) cat(3,[diff(x,1,1);zeros(1,size(x,2))],[diff(x,1,2) zeros(size(x,1),1)]);
div_1 = @(p) dxm(p(:,:,1)) + dym(p(:,:,2)); 
TV = @(x) sum(sum(sqrt(sum(D(x).^2,3))));
x = phantom(250); 

m = size(x,1);
n = size(x,2);


tv = TV(x);
tgv = TGV(x,Nbiter);
ritv = RITV(x,Nbiter);

x2 = imrotate(x,90);
tv2 = TV(x2);
tgv2 = TGV(x2,Nbiter);
ritv2 = RITV(x2,Nbiter);



z = insertText(  x ,[1,1], ...
   ['TV:' num2str(tv,'%0.1f')...
      '  TGV:' num2str(tgv,'%0.1f')...
   '  RITV:' num2str(ritv,'%0.1f')],'FontSize',a,'BoxColor','white');

z2 = insertText(  x2 ,[1,1], ...
   ['TV:' num2str(tv2,'%0.1f')...
     '  TGV:' num2str(tgv2,'%0.1f')...
   '  RITV:' num2str(ritv2,'%0.1f')],'FontSize',a,'BoxColor','white');
% TV_error = abs(tv-tv2)
% TGV_error = abs(tgv-tgv2)
% RITV_error = abs(ritv-ritv2)
% imwrite([z z2 ], 'fig1.png', 'png')
imshow([z z2 ]);

%     function tgv = TGV(u,Nbiter)
%         global D div_1
%         tau = 1/sqrt(8);
%         sigma = tau;
%         p = 0;
%         q = 0;
%         u = 0;
%         ubar = 0;
%         v = 0;
%         vbar = 0;
%         prox_G= @(t,sigma) bsxfun(@rdivide, t,...
%         max(sqrt(sum(t.^2,3))/(sigma),1));
%     for i=1:Nbiter
%        p = projP (p+sigma*(D(ubar)-vbar),alpha1);
%        q = projQ(q+sigma*E(vbar),alpha0);
%        uold = u;
%        u = u + tau*div_1(p);
%        ubar = 2*u - uold;
%        vold = v;
%        v = v+tau*(p+div_2(q));
%        vbar = 2*v - vold;
%        
%        pp = D(u) - v;
%        qq = E(v);
%        tgv = sum(sum(sqrt(abs(pp(:,:,1)).^2 + abs(pp(:,:,2)).^2)))...
%       + sum(sum(sqrt(abs(qq(:,:,1)).^2 + abs(qq(:,:,2)).^2 + 2*abs(qq(:,:,3)).^2)));
%    fprintf('iteration: %d   TGV: %f \n',i,tgv);
%     end

    function tgv = TGV(u,Nbiter)
        global D
        tau = 1/sqrt(8);
        sigma = tau;
        theta = 1;
        v0 = D(u);
        v = zeros([size(u) 2]);
        vbar = zeros([size(u) 2]);
        q = zeros([size(u) 3]);
        prox_G= @(t,sigma) t-bsxfun(@rdivide, t,...
        max(sqrt(sum(t.^2,3))/(sigma),1));
    for i=1:Nbiter
        q = projQ(q + sigma*E(vbar),1);
        v = prox_G(v +tau*div_2(q)-v0,tau)+v0;
        v_old = v;
        vbar = v + theta*(v-v_old);
        pp = v - D(u);
        qq = E(v);
       tgv = sum(sum(sqrt(abs(pp(:,:,1)).^2 + abs(pp(:,:,2)).^2)))...
      + sum(sum(sqrt(abs(qq(:,:,1)).^2 + abs(qq(:,:,2)).^2 + 2*abs(qq(:,:,3)).^2)));
   fprintf('iteration: %d   \n',i);
    end
 
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
    
    
    function RITV = RITV(x,Nbiter)
        global D
      v = zeros([size(x) 2 4]);  
      u = zeros([size(x) 2]);
      mu = 1;  
      sigma = 0.99/9; 
      prox_mu_sigma_g = @(t) t-bsxfun(@rdivide, t, max(sqrt(sum(t.^2,3))/(mu*sigma),1));
       
    for iter = 1:Nbiter
		%x = prox_mu_tau_f(x+tau*opDadj(-opD(x)+opLadj(v)-mu*u));
		v = prox_mu_sigma_g(v-sigma*L(-D(x)+LT(v)-mu*u));
		u = u-(-D(x)+LT(v))/mu;
% 		if mod(iter,40)==0
% 			%we display the primal and dual cost functions, which reach equal values at convergence
% 			fprintf('%d %f %f\n',iter,sum(sum((x-y).^2))/2+lambda*sum(sum(sum(sqrt(sum(v.^2,3))))),...
% 				-sum(sum((y-opDadj(u)).^2-y.^2))/2);
% 		
%         end 
        RITV = sum(sum(sum(sqrt(sum(v.^2,3)))));
        fprintf('iteration: %d  \n',iter);
    end

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
    

