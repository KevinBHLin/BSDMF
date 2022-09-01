function [ FusionImage, BlurVersion, ResOutline, ResOutFrame, val_RMSE, RMSE_ITER, ConvegenIter, H, U_avr, W_avr, V_avr ] = BSDMF( HSI, MSI, F, Sflag, Htype, dimen, rank, InitIter, MaxIter, REF, display, upsample_method)
% HSI---------------------m*n*L,HSI
% MSI-------------------- M*N*l,MSI
% F---------------------l*L, spectral transponse matrix
% Sflag-----------------1 when considering residual, otherwise 0
% Htype-----------------the method to obtain the spectral subspace matrix
% dimen*rank------------the size of U
% InitIter--------------the iteration number of initilization
% MaxIter---------------the iteration number of variational Bayes method
% REF-------------------the reference hyperspectral image
% display---------------1 when displaying the process, otherwise 0
% upsample_method-------upsampling method of low resolution HSI

%% initial the parameter

if nargin<12
    upsample_method = 'bicubic';
end


max_HSI = max(max(max(HSI)));
HSI = HSI./max_HSI;
MSI = MSI./max_HSI;
[M, N, l]   = size(MSI);
[m, n, L]   = size(HSI);

% HSI(HSI<0) = 0;
% MSI(MSI<0) = 0;

REF_edge = REF(M/m+1:M-M/m,N/n+1:N-N/n,:);
REF_edge = reshape(REF_edge,size(REF_edge,1)*size(REF_edge,2),size(REF_edge,3))';
REF = reshape(REF,size(REF,1)*size(REF,2),size(REF,3))';
RMSE_ITER = zeros(InitIter+InitIter,1);

param.ax = 1e-6;
param.bx = 1e-6;
param.ay = 1e-6;
param.by = 1e-6;
param.ar = 1e-6;
param.br = 1e-6;
param.au = 1e-6;
param.bu = 1e-6;
param.aw = 1e-6;
param.bw = 1e-6;
param.av = 1e-6;
param.bv = 1e-6;
param.InitialIter = InitIter;
param.iter  = MaxIter;
d           = dimen;
r           = rank;


%% Pro-treating

X           = imresize(HSI,[M N],upsample_method);
X           = reshape(X,M*N,L)';

Xori        = HSI;
Xori        = reshape(Xori,m*n,L)';

Y           = MSI;
Y           = reshape(Y,M*N,l)';

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                       %
% II. Subspace learning.                                                %
% ----------------------                                                %
%                                                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
switch  Htype 
    case 'VCA'
%     Find endmembers with VCA (pick the one with smallest volume from 20 
%     runs of the algorithm)
    max_vol = 0;
    vol = zeros(1, 20);
    for idx_VCA = 1:20
        E_aux = VCA(Xori,'Endmembers',d,'SNR',0,'verbose','off');
        vol(idx_VCA) = abs(det(E_aux'*E_aux));
        if vol(idx_VCA) > max_vol
            H = E_aux;
            max_vol = vol(idx_VCA);
        end   
    end
    
    case 'SFIM'
    Out_sfim = SFIM(HSI,MSI);
    Out_sfim = reshape(Out_sfim, M*N, L)';
    Ry = Out_sfim*Out_sfim';
    [H, ~] = svds(Ry,d);
    
    case 'PCA'
    [coeff1]= pca(X');
    H = coeff1(:,1:d);
    
    case  'SVD'
%     Xtmp = reshape(HSI,m*n,L)';
    Ry = X*X';
    [H, ~] = svds(Ry,d);
    
    case 'TEST'
    Ry = REF*REF';
    [H, ~] = svds(Ry,d);
    
    case 'Eye'
    H = eye(L);
    d   =   L;
    r   =   rank;
    
end

FH        = F*H;
HFFH        = FH'*FH;

alphax_avr  = param.ax/param.bx;
alphay_avr  = param.ay/param.by;
alphar_avr  = param.ar/param.br;
alphau_avr  = param.au/param.bu;
alphaw_avr  = param.aw/param.bw;
alphav_avr  = param.av/param.bv;

%% Initialization

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                       %
% III. Initialization.                                                  %
% ----------------------                                                %
%                                                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

alpha_x     = param.ax/param.bx;
alpha_y     = param.ay/param.by;
alpha_r     = param.ar/param.br;
alpha_u     = param.au/param.bu;
alpha_w     = param.aw/param.bw;
alpha_v     = param.av/param.bv;

W           = ones(r,M*N)*0.005;              %the average of Theta
V           = ones(r,M*N)*0.005;              %the average of Theta
R           = zeros(L,M*N);
R_avr       = R;
T           = W+V;

FusionImage       = zeros(L,M*N);

for t=1:param.InitialIter
        
    %sample U
    [Lambda_U, TmpU, TmpS] = Mypinv(kron(eye(d),W*W')*alpha_x + kron(HFFH,T*T')*alpha_y + alpha_u*eye(d*r));
    Mu_U = Lambda_U*reshape(alpha_x*W*(H'*(X-R))' + alpha_y*T*(FH'*(Y-F*R))',d*r,1);
    U_avr = reshape(Mu_U,r,d);
    U_vec = TmpU*TmpS*randn(d*r,1)+Mu_U;
    U = reshape(U_vec,r,d);

    %sample W
    [SigmaW, TmpU, TmpS] = Mypinv(alpha_w*eye(r) + alpha_x*U*U' + alpha_y*U*HFFH*U');
    W_avr = SigmaW*(alpha_x*U*(H'*(X-R)) + alpha_y*U*FH'*(Y-FH*U'*V-F*R));
    W = TmpU*TmpS*randn(size(W_avr))+W_avr;   
    
    %sample V
    [SigmaV, TmpU, TmpS] = Mypinv(alpha_v*eye(r) + alpha_y*U*HFFH*U');
    V_avr = alpha_y*SigmaV*U*(FH'*(Y-FH*U'*W-F*R));
    V = TmpU*TmpS*randn(size(V_avr))+V_avr;
    
    T = W + V;
    
%     if Sflag~=0
        %sample R
        [SigmaR, TmpU, TmpS] = Mypinv((alpha_r+alpha_x)*eye(L) + alpha_y*F'*F);
        R_avr = SigmaR*(alpha_y*F'*(Y-FH*U'*T)+alpha_x*(X-H*U'*W));
        R = TmpU*TmpS*randn(size(R_avr))+R_avr;
    
        %sample alpha_r
        ar = param.ar + 0.5*numel(R);
        br = param.br + 0.5*sum(sum(R.^2));
        alpha_r = gamrnd(ar,1./br);
        alphar_avr = ar./br;
%     end
    
    %sample alpha_x
    ax = param.ax + 0.5*numel(X);
    bx = param.bx + 0.5*sum(sum((X - H*U'*W).^2));
    alpha_x = gamrnd(ax,1./bx);
    alphax_avr = ax./bx;

    %sample alpha_y
    ay = param.ay + 0.5*numel(Y);
    by = param.by + 0.5*sum(sum((Y - FH*U'*T).^2));
    alpha_y = gamrnd(ay,1./by);
    alphay_avr = ay./by;

    %sample alpha_u
    au = param.au + 0.5*numel(U);
    bu = param.bu + 0.5*sum(sum(U.^2));
    alpha_u = gamrnd(au,1./bu);
    alphau_avr = au./bu;

    %sample alpha_s
    aw = param.aw + 0.5*numel(W);
    bw = param.bw + 0.5*sum(sum(W.^2));
    alpha_w = gamrnd(aw,1./bw);
    alphaw_avr = aw./bw;

    %sample alpha_t
    av = param.av + 0.5*numel(V);
    bv = param.bv + 0.5*sum(sum(V.^2));
    alpha_v = gamrnd(av,1./bv);
    alphav_avr = av./bv;
    
    if display==1
        FusionImage = H*U'*T+R;
        Z = FusionImage*max_HSI;

        RMSE_ITER(t) = sqrt((norm((REF - Z),'fro')).^2/(M*N*L));

        disp(['Initial Iteration = ' num2str(t) ', RMSE = ',num2str(RMSE_ITER(t)) ', alpha_x = ',num2str(alpha_x)...
            ', alpha_y = ',num2str(alpha_y) ', alpha_u = ',num2str(alpha_u) ', alpha_v = ',num2str(alpha_v) ', alpha_w = ',num2str(alpha_w)])
    elseif display==2
        FusionImage = H*U'*T+R;
        Z = FusionImage*max_HSI;
        Z = reshape(Z',M,N,L);Z = Z(M/m+1:M-M/m,N/n+1:N-N/n,:);Z = reshape(Z,size(Z,1)*size(Z,2),size(Z,3))';
        RMSE_ITER(t) = sqrt((norm((REF_edge - Z),'fro')).^2/(size(Z,1)*size(Z,2)*size(Z,3)));

        disp(['Initial Iteration = ' num2str(t) ', RMSE = ',num2str(RMSE_ITER(t)) ', alpha_x = ',num2str(alpha_x)...
            ', alpha_y = ',num2str(alpha_y) ', alpha_u = ',num2str(alpha_u) ', alpha_v = ',num2str(alpha_v) ', alpha_w = ',num2str(alpha_w)])
    end
    
end

%% the variational bayesian algorithm

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                       %
% IV. The Variational Bayesian Algorithm.                               %
% ---------------------------------------                               %
%                                                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

EWW     = W_avr*W_avr' + size(W_avr,2)*SigmaW;
EVV     = V_avr*V_avr' + size(V_avr,2)*SigmaV;
ETT     = EWW + EVV + V_avr*W_avr' + W_avr*V_avr';
T_avr   = W_avr + V_avr;
ERR     = zeros(L,L);
if Sflag==0
    R_avr=zeros(L,M*N);
end
Lambda_U = mat2cell(Lambda_U, ones(1,d)*r, ones(1,d)*r);
Lambda_U = reshape(Lambda_U, 1,d*d);
Lambda_U = cellfun(@(x)(reshape(x,r*r,1)),Lambda_U,'UniformOutput',false);
Lambda_U = cell2mat(Lambda_U);
EUU = sum(Lambda_U,2); EUU = reshape(EUU, r, r) + U_avr*U_avr';
EUFFU = sum(Lambda_U.*repmat(HFFH(:)',r*r,1),2); EUFFU = reshape(EUFFU, r, r) + U_avr*HFFH*U_avr';

ConvegenIter = -1;

for t=1:param.iter
    %% VBE-step
    
    %update the parameter of distribution U
    Lambda_U = Mypinv( kron(eye(d),EWW)*alphax_avr + kron(HFFH,ETT)*alphay_avr + alphau_avr*eye(d*r) );
    Mu_U = Lambda_U*reshape(alphax_avr*W_avr*(H'*(X-R_avr))' + alphay_avr*T_avr*(FH'*(Y-F*R_avr))',d*r,1);
    U_avr = reshape(Mu_U,r,d);
    Lambda_U = mat2cell(Lambda_U, ones(1,d)*r, ones(1,d)*r);
    Lambda_U = reshape(Lambda_U, 1,d*d);
    Lambda_U = cellfun(@(x)(reshape(x,r*r,1)),Lambda_U,'UniformOutput',false);
    Lambda_U = cell2mat(Lambda_U);
    EUU = sum(Lambda_U,2); EUU = reshape(EUU, r, r) + U_avr*U_avr';
    EUFFU = sum(Lambda_U.*repmat(HFFH(:)',r*r,1),2); EUFFU = reshape(EUFFU, r, r) + U_avr*HFFH*U_avr';

    %update the parameter of distribution W
    Namtas = Mypinv(alphaw_avr*eye(r) + alphax_avr*EUU + alphay_avr*EUFFU);
    W_avr = Namtas*(alphax_avr*U_avr*H'*(X-R_avr) + alphay_avr*U_avr*FH'*(Y-F*R_avr)-alphay_avr*EUFFU*V_avr);
    EWW = W_avr*W_avr' + size(W_avr,2)*Namtas;
    
    %update the parameter of distribution V
    Namtat = Mypinv(alphav_avr*eye(r) + alphay_avr*EUFFU);
    V_avr = alphay_avr*Namtat*(U_avr*FH'*(Y-F*R_avr)-EUFFU*W_avr);
    EVV = V_avr*V_avr' + size(V_avr,2)*Namtat;

    T_avr = V_avr + W_avr;
    ETT     = EWW + EVV + V_avr*W_avr' + W_avr*V_avr';
    
    if Sflag~=0
        %sample R
        NamtaR = Mypinv((alphar_avr+alphax_avr)*eye(L) + alphay_avr*F'*F);
        R_avr = NamtaR*(alphay_avr*F'*(Y-FH*U_avr'*T_avr)+alphax_avr*(X-H*U_avr'*W_avr));
        ERR   = R_avr*R_avr' + size(R_avr,2)*NamtaR;

        %sample alphar
        alphar_avr = (param.ar + 0.5*numel(R_avr))./(param.br + 0.5*(trace(ERR)));
    end
    
    %% VBM-step
    
    %update the parameter of distribution alphax
    alphax_avr = (param.ax + 0.5*numel(X))/(param.bx + 0.5*( sum(sum((X-H*(U_avr'*W_avr)-R_avr).^2)) + trace(EUU*EWW) - trace(U_avr*(U_avr'*W_avr*W_avr')) + trace(ERR)-trace(R_avr*R_avr') ));

    %update the parameter of distribution alphay
    alphay_avr = (param.ay + 0.5*numel(Y))/(param.by + 0.5*( sum(sum((Y-FH*(U_avr'*T_avr)-F*R_avr).^2)) + trace(EUFFU*ETT) - trace(U_avr*HFFH*(U_avr'*T_avr*T_avr')) + trace(F*ERR*F')-trace(F*R_avr*R_avr'*F') ));

    %update the parameter of distribution alphap
    alphau_avr = (param.au + 0.5*numel(U_avr))./(param.bu + 0.5*(trace(EUU)));
    
    %update the parameter of distribution alphas
    alphaw_avr = (param.aw + 0.5*numel(W_avr))./(param.bw + 0.5*trace(EWW));
    
    %update the parameter of distribution alphat
    alphav_avr = (param.av + 0.5*numel(V_avr))./(param.bv + 0.5*trace(EVV));
    
    Out_new = H*U_avr'*(T_avr)+R_avr;

    delta = sqrt(sum(sum( (FusionImage-Out_new).^2 ))/(M*N*L));

    FusionImage = Out_new;
    
    if(delta<0.0001 && ConvegenIter<0)
        ConvegenIter = t;
    end
%     if(criton==1 && delta<0.0001)
%         break;
%     end

    if display==1
        Z = FusionImage*max_HSI;

        RMSE_ITER(t+param.InitialIter) = sqrt((norm((REF - Z),'fro')).^2/(M*N*L));

        disp(['Initial Iteration = ' num2str(t) ', RMSE = ',num2str(RMSE_ITER(t+param.InitialIter)) ', alpha_x = ',num2str(alphax_avr)...
            ', alpha_y = ',num2str(alphay_avr) ', alpha_u = ',num2str(alphau_avr) ', alpha_v = ',num2str(alphav_avr) ', alpha_w = ',num2str(alphaw_avr)])
    elseif display==2
        Z = FusionImage*max_HSI;
        Z = reshape(Z',M,N,L);Z = Z(M/m+1:M-M/m,N/n+1:N-N/n,:);Z = reshape(Z,size(Z,1)*size(Z,2),size(Z,3))';
        RMSE_ITER(t+param.InitialIter) = sqrt((norm((REF_edge - Z),'fro')).^2/(size(Z,1)*size(Z,2)*size(Z,3)));

        disp(['Initial Iteration = ' num2str(t) ', RMSE = ',num2str(RMSE_ITER(t+param.InitialIter)) ', alpha_x = ',num2str(alphax_avr)...
            ', alpha_y = ',num2str(alphay_avr) ', alpha_u = ',num2str(alphau_avr) ', alpha_v = ',num2str(alphav_avr) ', alpha_w = ',num2str(alphaw_avr)])
    end
    
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                       %
% V. Evaluation.                                                        %
% ---------------------------------------                               %
%                                                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

FusionImage = FusionImage.*max_HSI;
val_RMSE = sqrt(mean((FusionImage(:)-REF(:)).^2));

FusionImage = reshape(FusionImage',M,N,L);
BlurVersion = H*U_avr'*W_avr;
BlurVersion = reshape(BlurVersion',M,N,L).*max_HSI;
ResOutline = H*U_avr'*V_avr;
ResOutline = reshape(ResOutline',M,N,L).*max_HSI;
ResOutFrame = reshape(R_avr',M,N,L).*max_HSI;

W_avr = reshape(W_avr',M,N,r);
V_avr = reshape(V_avr',M,N,r);

end


function [X, Ux, sqrtSx] = Mypinv(A)

%    A = (A+A')./2;
   [m,n] = size(A);

   [V,S] = svd(A);
%    S    = fliplr(S); %×óÓÒ·­×ª
%    S    = S';
%    S    = fliplr(S);
   if m > 1, s = diag(S);
      elseif m == 1, s = S(1);
      else s = 0;
   end
   tol = max(m,n) * eps(max(s));
   r = sum(s > tol);
%    if r<size(s)
%        disp(['Low rank', num2str(r)]);
%    end
   if (r == 0)
      X = zeros(size(A'),class(A));
      Ux = zeros(size(A',1),size(A',1));
      sqrtSx = zeros(size(A'));
      disp(['Low rank', num2str(r)]);
   else
      sqrtSx = zeros(size(s));
      sqrtSx(1:r) = ones(r,1)./sqrt(s(1:r));
      sqrtSx = diag(sqrtSx);
      s = diag(ones(r,1)./s(1:r));
      X = V(:,1:r)*s*V(:,1:r)';
      Ux = [V(:,1:r) zeros(size(V,1),size(V,2)-r)];
   end
%    X = (X' + X)./2;

end


