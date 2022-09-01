clear all
close all
clc
addpath(genpath('functions'))
%--------------------------------------------------------------
% Simulated Nikon D700 spectral response T = [R;G;B], 
% Please note, the same transformation must be used between 'the ground
% truth and RGB image' and the leanred and the transformed dictionary.
%--------------------------------------------------------------
T1 = [0.005 0.007 0.012 0.015 0.023 0.025 0.030 0.026 0.024 0.019 0.010 0.004   0     0      0    0     0     0     0     0     0     0     0     0     0     0     0     0    0     0       0];
T2 = [0.000 0.000 0.000 0.000 0.000 0.001 0.002 0.003 0.005 0.007 0.012 0.013 0.015 0.016 0.017 0.02 0.013 0.011 0.009 0.005  0.001  0.001  0.001 0.001 0.001 0.001 0.001 0.001 0.002 0.002 0.003 ];
T3 = [0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.001 0.003 0.010 0.012  0.013  0.022  0.020 0.020 0.018 0.017 0.016 0.016 0.014 0.014 0.013];
F  = [T1;T2;T3];


img_path = 'data'; % the path of the CAVE dataset
FileList = dir(img_path);
folderlength = 0;

for i=1:length(FileList)
    if ( FileList(i).isdir==1 && ~strcmp(FileList(i).name,'.') && ~strcmp(FileList(i).name,'..') )
        folderlength = folderlength + 1;
        FileFolder{folderlength} = [FileList(i).name];
    end
end

Time = zeros(folderlength,1);
for t = 1:folderlength
%--------------------------------------------------------------
% Read the input image
%--------------------------------------------------------------    
    for j=1:31
        str=strcat(img_path, '\', FileFolder{t}, '\', FileFolder{t}, '\',...
                FileFolder{t}, '_', num2str(floor(j/10)), num2str(mod(j,10)), '.png');
            ddd = imread(str);
        I_REF(:,:,j) = ddd(:,:,1);
    end
    
if t==32
    I_REF = double(I_REF)./255;
else
    I_REF = double(I_REF)./65535;%max(double(I_REF(:)));%im2double(I_REF);
end

ratio = 2; % difference of GSD between MS and HS
size_kernel=[2*ratio-1 2*ratio-1];

sig = (1/(2*(2.7725887)/ratio^2))^0.5;
start_pos(1)=1; % The starting point of downsampling
start_pos(2)=1; % The starting point of downsampling

KerBlu = fspecial('Gaussian',[size_kernel(1) size_kernel(2)],sig);
I_HS=imfilter(I_REF, KerBlu, 'circular');
I_HS=I_HS(start_pos(1):ratio:end, start_pos(2):ratio:end,:);

I_MS = reshape((F*reshape(I_REF,[],size(I_REF,3))')',size(I_REF,1),size(I_REF,2),[]);

snrMSdB = 40; snrHSdB = 35;
%add noise
I_HSn = I_HS; I_MSn = I_MS;
sigma_hsi = zeros(size(I_HSn,3),1);
sigma_msi = zeros(size(I_MSn,3),1);
snrHS = 10^(snrHSdB/20);
snrMS = 10^(snrMSdB/20);
for i = 1:size(I_HSn,3)
    mu = mean(reshape(I_HSn(:,:,i),1,[]));
    sigma_hsi(i) = mu/snrHS;
    I_HSn(:,:,i) = I_HSn(:,:,i) + randn(size(I_HSn,1),size(I_HSn,2))*sigma_hsi(i);
end
for i = 1:size(I_MSn,3)
    mu = mean(reshape(I_MSn(:,:,i),1,[]));
    sigma_msi(i) = mu/snrMS;
    I_MSn(:,:,i) = I_MSn(:,:,i) + randn(size(I_MSn,1),size(I_MSn,2))*sigma_msi(i);
end

Truth = hyperConvert2d(I_REF);
[M,N,L] = size(I_REF);

t0=clock;
[Z6,blur,residual,Simage] = BSDMF( I_HSn, I_MSn, F, 0, 'SVD', 16, 25, 15, 20, I_REF,0);
Time(t,1)=etime(clock,t0);

% Perform the evaluation
[psnr6,rmse6, ergas6, sam6, uiqi6,ssim6,DD6,CC6] = quality_assessment(I_REF.*255, Z6.*255, 0, 1.0/ratio);


t

end


