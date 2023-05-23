clc;clear;
a =rand(300,100);
b = rand(90,70);

fftw_fft = fft_conv2d(a,b);

%% 【CPU】org conv.
%post_conv = conventional_conv2d_full(a,b);
%%
[ma,na]=size(a);[mb,nb]=size(b);
a_zp = zeros(ma+mb-1, na+nb-1);
b_zp = zeros(ma+mb-1, na+nb-1);
a_zp(1:ma, 1:na)=a;
b_zp(1:mb, 1:nb)=b;
matlab_fft = ifft2(fft2(a_zp).*fft2(b_zp));
sum(sum(fftw_fft)) - sum(sum(matlab_fft))
%% 【CPU】FFT conv.
[a_r,a_c] = fft_2d(a_zp);
[b_r,b_c] = fft_2d(b_zp);
a_com = complex(a_r,a_c);
b_com = complex(b_r,b_c);
c = a_com.*b_com;
c_r = real(c);
c_c = imag(c);
d = ifft_2d(c_r,c_c);
%% 【CPU】fast FFT conv.
[a_r,a_c] = FastFT_2d(a_zp);
[b_r,b_c] = FastFT_2d(b_zp);
a_com = complex(a_r,a_c);
b_com = complex(b_r,b_c);
c = a_com.*b_com;
c_r = real(c);
c_c = imag(c);
d = iFastFT_2d(c_r,c_c);
%% 【CPU】FFT conv. same function