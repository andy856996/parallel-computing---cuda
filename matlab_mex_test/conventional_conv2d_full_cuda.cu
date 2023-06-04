/*----------------------------------------------
create by LAI,WEI-LIN mod by Ding Ze An(apply CUDA)
Date : 2023 5 23
National Taitung Unv. IPGIT
Email : andy856996@gamil.com(Ding Ze An)
Email : 10822112@gm.nttu.edu.tw(LAI,WEI-LIN)
----------------------------------------------*/
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include "mex.h"
__global__ static void convolution2D_CUDA
(double *mask, double *kernel,double* outfinal, int m_row, int m_col, int k_row, int k_col){
	int m, n;
	const int j = blockDim.x * blockIdx.x+threadIdx.x;
	const int i = blockDim.y * blockIdx.y+threadIdx.y;
    int o_row = m_row + k_row - 1;
    int o_col = m_col + k_col - 1;
	if(i<o_row&&j<o_col){
		double temp = 0.0;
		for (m = 0; m < k_row; m++) {
			for (n = 0; n < k_col; n++) {
				int mask_row = i - m;
				int mask_col = j - n;
				if (mask_row >= 0 && mask_row < m_row && mask_col >= 0 && mask_col < m_col) {
					int mask_index = mask_row * m_col + mask_col;
					int kernel_index = m * k_col + n;
					temp += mask[mask_index] * kernel[kernel_index];
				}
			}
		}
		outfinal[j * o_row + i] = temp;
	}
	
}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    double *mask;
    double *kernal;
    double *output;
    int mrows, mcols, krows, kcols;

    mask = mxGetPr(prhs[0]);
    kernal = mxGetPr(prhs[1]);

    mrows = mxGetM(prhs[0]);
    mcols = mxGetN(prhs[0]);

    krows = mxGetM(prhs[1]);
    kcols = mxGetN(prhs[1]);
	
	int output_dim_R = mrows + krows - 1;
	int output_dim_C = mcols + kcols - 1;
	
    plhs[0] = mxCreateDoubleMatrix(mrows + krows - 1, mcols + kcols - 1, mxREAL);
    output = mxGetPr(plhs[0]);
	
	double* mask_gpu;double* kernal_gpu;double* output_gpu;
	double* outputFinal_gpu;
	cudaMalloc((void**)&mask_gpu, mrows * mcols * sizeof(double));
	cudaMalloc((void**)&kernal_gpu, krows * kcols* sizeof(double));
	cudaMalloc((void**)&outputFinal_gpu, output_dim_R * output_dim_C * sizeof(double));
	
	cudaMemcpy(mask_gpu, mask, mrows * mcols * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(kernal_gpu, kernal, krows * kcols* sizeof(double), cudaMemcpyHostToDevice);
	
	const int threadNum = 32;
	dim3 threads(threadNum, threadNum, 1); 
	dim3 grid((output_dim_C + threads.x - 1) / threads.x, (output_dim_R + threads.y - 1) / threads.y, 1);
	convolution2D_CUDA<<<grid,threads>>>(mask_gpu, kernal_gpu,outputFinal_gpu, mrows, mcols, krows, kcols);
	
	cudaFree(mask_gpu);
	cudaFree(kernal_gpu);
	
	cudaMemcpy(output, outputFinal_gpu,output_dim_R * output_dim_C * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(outputFinal_gpu);
}