/*----------------------------------------------
create by Ding Ze An
Date : 2023 5 23
National Taitung Unv. IPGIT
Email : andy856996@gamil.com
----------------------------------------------*/
#include <iostream>
#include <cufft.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"
using namespace std;

void print_mat(double *inMatrix,int R,int C){
	for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            cout <<  inMatrix[i*C+j]<<" " ;
        }
		cout<<endl;
    }
}
 __global__ static void mult_complex_arr(cufftDoubleComplex* array1,cufftDoubleComplex* array2,cufftDoubleComplex* result,int size){
	int const i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i<size){
		result[i].x = array1[i].x * array2[i].x - array1[i].y * array2[i].y;  // Real part of result
		result[i].y = array1[i].x * array2[i].y + array1[i].y * array2[i].x;  // Imaginary part of result
		//printf("i=%d,",i,);
	}
}
 __global__ static void gpuComplexArr2Matlab_output(double* array1,cufftDoubleComplex* array2,int R,int C){
	 
	const int j = blockDim.x * blockIdx.x+threadIdx.x;
	const int i = blockDim.y * blockIdx.y+threadIdx.y;
	
	if (j < C && i < R) {
        array1[i*C+j] = array2[i+j*R].x/(R*C);
    }
}
void put_data_in_comMat(double *input,double *output,int R,int C){
	for (int i = 0; i < R; i++) {
		for (int j = 0;j<C;j++){
			output[(C*2)*i+j*2] = input[i+j*R];
			output[(C*2)*i+j*2+1] = 0.0;
		}
    }
}
void double2complex_arrary(double * realArray,cufftDoubleComplex* complexArray,int R,int C){
	for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
			complexArray[i*C+j].x = realArray[(C*2)*i+j*2];
			complexArray[i*C+j].y = 0.0;
			//cout<<complexArray[i*C+j].x<<"+"<<complexArray[i*C+j].y<<"i ";
        }
		//cout<<endl;
    }
}

void print_complex_mat(cufftDoubleComplex* complexArray,int R,int C){
	for (int i = 0; i < R; i++) {
		for (int j = 0;j<C;j++){
			cout<<complexArray[i*C+j].x<<"+"<<complexArray[i*C+j].y<<"i ";
		}
		cout<<endl;
    }
}
void put_ComplexMat_toMATLAB_mat(cufftDoubleComplex *in,double *out,int R,int C){
	for (int i = 0; i < R; i++) {
		for (int j = 0;j<C;j++){
			out[i*C+j] = (in[i+j*R].x)/(R*C);
		}
    }
}
 __global__ static void KERNAL_covertMATLABDoubleMatrix2Complex(const double* realArray, cufftDoubleComplex* complexArray, int R, int C)
{
	const int col = blockDim.x * blockIdx.x+threadIdx.x;
	const int row = blockDim.y * blockIdx.y+threadIdx.y;
	
	
	int idx_realArray = row + col * R;
    int idx_complexArray = row * C + col;
    if (col < C && row < R) {
        complexArray[idx_complexArray].x = realArray[idx_realArray];
        complexArray[idx_complexArray].y = 0.0;
    }
}

void covertMATLABDoubleMatrix2Complex(double* realArray, cufftDoubleComplex* complexArray, int R, int C)
{
    for (int i = 0; i < R; i++) {
		for (int j = 0;j<C;j++){
			complexArray[i * C + j].x = realArray[i + j * R];
			complexArray[i * C + j].y = 0.0;
		}
    }
	
}
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
	char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
	char const * const errMsg = "mask dim must be same as kernal dim";
	
    double *mask;
    double *kernel;

    mask = mxGetPr(prhs[0]);
    kernel = mxGetPr(prhs[1]);

    int R = mxGetM(prhs[0]);
    int C = mxGetN(prhs[0]);
	
	int R_k = mxGetM(prhs[1]);
    int C_k = mxGetN(prhs[1]);
	
	if (R != R_k || C != C_k){
		mexErrMsgIdAndTxt(errId, errMsg);
	}

    mxInitGPU();
	
	double* mask_gpu;
	double* kernel_gpu;
	cudaMalloc((void**)&mask_gpu, R * C * sizeof(double));
	cudaMalloc((void**)&kernel_gpu, R * C * sizeof(double));
	cudaMemcpy(mask_gpu, mask, sizeof(double) * R * C, cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_gpu, kernel, sizeof(double) * R * C, cudaMemcpyHostToDevice);

    cufftDoubleComplex* mask_complex_gpu;
    cufftDoubleComplex* kernal_complex_gpu;
    cufftDoubleComplex* result_complex_gpu;

    cudaMalloc((void**)&mask_complex_gpu, R * C * sizeof(cufftDoubleComplex));
    cudaMalloc((void**)&kernal_complex_gpu, R * C * sizeof(cufftDoubleComplex));
    cudaMalloc((void**)&result_complex_gpu, R * C * sizeof(cufftDoubleComplex));
	
	const int BLOCK_SIZE = 30;
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1); 
	dim3 grid((C + threads.x - 1) / threads.x, (R + threads.y - 1) / threads.y, 1);
	
	KERNAL_covertMATLABDoubleMatrix2Complex<<<grid, threads>>>(mask_gpu,mask_complex_gpu,R,C);
	KERNAL_covertMATLABDoubleMatrix2Complex<<<grid, threads>>>(kernel_gpu,kernal_complex_gpu,R,C);
	
	
    cufftHandle mask_plan;
    cufftResult mask_result = cufftPlan2d(&mask_plan,R,C, CUFFT_Z2Z);
    cufftHandle kernal_plan;
    cufftResult kernal_result = cufftPlan2d(&kernal_plan, R,C, CUFFT_Z2Z);
    cufftHandle resultifft_plan;
    cufftResult resultifft_result = cufftPlan2d(&resultifft_plan, R,C, CUFFT_Z2Z);
	
    mask_result = cufftExecZ2Z(mask_plan, mask_complex_gpu, mask_complex_gpu, CUFFT_FORWARD);
    kernal_result = cufftExecZ2Z(kernal_plan, kernal_complex_gpu, kernal_complex_gpu, CUFFT_FORWARD);
	
	int const threadsPerBlock = 1024;
    int blocksPerGrid = (R*C + threadsPerBlock - 1) / threadsPerBlock;;
    mult_complex_arr<<<blocksPerGrid, threadsPerBlock>>>(mask_complex_gpu, kernal_complex_gpu,result_complex_gpu, R * C);
	
	/*free memroy*/
	cufftDestroy(mask_result);
	cufftDestroy(kernal_result);
	cudaFree(mask_complex_gpu);
	cudaFree(kernal_complex_gpu);
	cudaFree(mask_gpu);
	cudaFree(kernel_gpu);
	/*free memroy*/
	
    resultifft_result = cufftExecZ2Z(resultifft_plan, result_complex_gpu, result_complex_gpu, CUFFT_INVERSE);
	
	double * result_output_gpu;
	cudaMalloc((void**)&result_output_gpu, R * C * sizeof(double));
	
	gpuComplexArr2Matlab_output<<<grid, threads>>>(result_output_gpu,result_complex_gpu, R,C);
	
    plhs[0] = mxCreateDoubleMatrix(R, C, mxREAL);
	cudaMemcpy(mxGetPr(plhs[0]), result_output_gpu, R * C * sizeof(double), cudaMemcpyDeviceToHost);
	
	/*free memroy*/
	cufftDestroy(resultifft_plan);
	cudaFree(result_complex_gpu);
	cudaFree(result_output_gpu);
	/*free memroy*/
}