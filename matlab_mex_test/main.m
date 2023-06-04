clc;clear;figure;
%mexcuda  mexGPUExample.cu
idx = 1;
for i = [10 50 100 200 300]
    dim = i;
    a =rand(dim,dim) ;
    b = rand(dim,dim);    
    [ma,na]=size(a);[mb,nb]=size(b);
    a_zp = zeros(ma+mb-1, na+nb-1);
    b_zp = zeros(ma+mb-1, na+nb-1);
    a_zp(1:ma, 1:na)=a;
    b_zp(1:mb, 1:nb)=b;
    %% 【convolution base】
    %【CPU】
    tic;
    matlab_conv2 = conv2(a,b,"full");
    matlab_conv_time = toc;
    %【CPU】org conv. C++
    tic;
    org_conv = conventional_conv2d_full(a,b);
    org_conv_time = toc;
    %【CPU omp】org conv. C omp
    tic;
    org_conv_omp = conventional_conv2d_full_omp(a,b);
    org_conv_time_omp = toc;
    %【GPU】conv2 CUDA
    tic;
    org_conv_GPU = conventional_conv2d_full_cuda(a,b);
    org_conv_GPU_time = toc;
    %% 【FFT base】
    %【CPU】 FFTW lib using C++
    tic;
    fftw_fft = fft_conv2d(a,b);
    fftw_cpu_time = toc;
    %【CPU】 fft matlab conv2d
    tic;
    matlab_fft = ifft2(fft2(a_zp).*fft2(b_zp));
    fft_matlab_time = toc;
    %【GPU】cuda cufft 
    tic;
    cufft_conv2d_ans = cufft_conv2d(a_zp,b_zp);
    fftcufft_time = toc;
    %% plot figure
    semilogy([matlab_conv_time org_conv_time org_conv_time_omp  org_conv_GPU_time...
       fftw_cpu_time fft_matlab_time fftcufft_time],'-o')
    xticks([1 2 3 4 5 6 7]);xticklabels({'conv2(matlab)','conv2(C)','conv2(C OMP)','conv2(cuda)', ...
        'fftw(C)','fft(matlab)','fft(cuda)'});
    set(gca,'FontSize',15,'FontName','Times New Roman');xlabel('Method');ylabel('Time(s)');
    legend_cell{idx} =  ['dimension' num2str(dim) 'X' num2str(dim)];hold on;
    idx = idx + 1;
    % semilogy([fftw_cpu_time fft_matlab_time fftcufft_time],'-o')
    % xticks([1 2 3 4 5 6]);xticklabels({'fftw(C)','fft(matlab)','fft(cuda)'});
    % set(gca,'FontSize',15,'FontName','Times New Roman');xlabel('Method');ylabel('Time(s)');
    % legend(['dimension' num2str(dim) 'X' num2str(dim)]);hold on;
end
legend(legend_cell);