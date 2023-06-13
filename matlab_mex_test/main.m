clc;clear;%figure;
cpu_fft_conv2d = py.importlib.import_module('fft_convolve2d');
gpu_fft_conv2d = py.importlib.import_module('gpu_fft_convolve2D');
idx = 1;
dev= gpuDevice();
path = 'C:\Users\chliu\Downloads\';
for i = 1000:1000:5000
    fig = figure;
    dim = i;
    a =rand(dim,dim) ;
    b = rand(dim,dim);    
    [ma,na]=size(a);[mb,nb]=size(b);
    a_zp = zeros(ma+mb-1, na+nb-1);
    b_zp = zeros(ma+mb-1, na+nb-1);
    a_zp(1:ma, 1:na)=a;
    b_zp(1:mb, 1:nb)=b;
    %% 【convolution base】
    %【CPU】matlab conv2d
    % tic;
    % matlab_conv2 = conv2(a,b,"full");
    % matlab_conv_time = toc;
    % %【CPU】org conv. C++
    % tic;
    % org_conv = conventional_conv2d_full(a,b);
    % org_conv_time = toc;
    % %【CPU omp】org conv. C omp
    % tic;
    % org_conv_omp = conventional_conv2d_full_omp(a,b);
    % org_conv_time_omp = toc;
    % %【GPU】conv2 CUDA
    % tic;
    % org_conv_GPU = conventional_conv2d_full_cuda(a,b);
    % org_conv_GPU_time = toc;

    %% 【FFT base】
    % %【CPU】 FFTW lib using C++
    % tic;
    % fftw_fft = fft_conv2d(a,b);
    % fftw_cpu_time = toc;
    % %【CPU】 fft matlab conv2d
    % tic;
    % matlab_fft = ifft2(fft2(a_zp).*fft2(b_zp));
    % fft_matlab_time = toc;
    % %【CPU】python fft conv
    % tic;
    % py_result_cpu_fft = cpu_fft_conv2d.fft_conv2d(a_zp, b_zp);
    % py_result_cpu_fft_time = toc;
    % sh = double(py.array.array('d',py_result_cpu_fft.shape));
    % npary1 = double(py.array.array('d',py.numpy.nditer(py_result_cpu_fft)));
    % result_cpu_fft_mat = reshape(npary1,fliplr(sh))';  % matlab 2d array
    %【GPU】python fft conv library-cupy
    tic;
    py_result_gpu_fft = gpu_fft_conv2d.gpu_fft_convolve2D(a_zp, b_zp);
    py_result_gpu_fft_time = toc;
    sh = double(py.array.array('d',py_result_gpu_fft.shape));
    npary1 = double(py.array.array('d',py.numpy.nditer(py_result_gpu_fft)));
    result_gpu_fft_mat = reshape(npary1,fliplr(sh))';  % matlab 2d array
    %【GPU】cuda cufft 
    tic;
    cufft_conv2d_ans = cufft_conv2d(a_zp,b_zp);
    fftcufft_time = toc;
    %% plot figure
    % semilogy([matlab_conv_time org_conv_time org_conv_time_omp  org_conv_GPU_time...
    %    fftw_cpu_time fft_matlab_time fftcufft_time py_result_cpu_fft_time py_result_gpu_fft_time],'-o')
    % xticks([1 2 3 4 5 6 7 8 9]);xticklabels({'conv2(matlab)','conv2(C)','conv2(C OMP)','conv2(cuda)', ...
    %     'fft(cpu fftw)','fft(cpu matlab)','fft(gpu cufft)','fft(python cpu)','fft(python gpu cupy)'});
    % set(gca,'FontSize',20,'FontName','Times New Roman');xlabel('Method');ylabel('Time(s)');
    % legend_cell{idx} =  ['dimension ' num2str(dim) 'x' num2str(dim)];hold on;
    % idx = idx + 1;

    % semilogy([fftw_cpu_time fft_matlab_time py_result_cpu_fft_time fftcufft_time py_result_gpu_fft_time],'-o')
    % xticks([1 2 3 4 5]);xticklabels({'fft(cpu fftw)','fft(cpu matlab)','fft(python cpu)','fft(gpu cufft)','fft(python gpu cupy)'});
    % set(gca,'FontSize',20,'FontName','Times New Roman');xlabel('Method');ylabel('Time(s)');
    % legend_cell{idx} =  ['dimension ' num2str(dim) 'x' num2str(dim)];hold on;
    % idx = idx + 1;

    semilogy([fftcufft_time py_result_gpu_fft_time],'-o')
    xticks([1 2 3 4 5]);xticklabels({'fft(gpu cufft)','fft(python gpu cupy)'});
    set(gca,'FontSize',20,'FontName','Times New Roman');xlabel('Method');ylabel('Time(s)');
    legend_cell{idx} =  ['dimension ' num2str(dim) 'x' num2str(dim)];hold on;
    idx = idx + 1;
    
    fig_path = [path 'fig_' num2str(i) '.fig'];
    saveas(fig,fig_path)
    close(fig);
    dev.reset;
end
legend(legend_cell);
dev.reset;
