clc;clear;
a =rand(7895,7274);
b = rand(1234,3540);
%% 【CPU】 matlab conv2d
[ma,na]=size(a);[mb,nb]=size(b);
a_zp = zeros(ma+mb-1, na+nb-1);
b_zp = zeros(ma+mb-1, na+nb-1);
a_zp(1:ma, 1:na)=a;
b_zp(1:mb, 1:nb)=b;
matlab_fft = ifft2(fft2(a_zp).*fft2(b_zp));
%% 【CPU】 FFTW lib using C++
fftw_fft = fft_conv2d(a,b);
%% 【CPU】org conv. C++
post_conv = conventional_conv2d_full(a,b);
