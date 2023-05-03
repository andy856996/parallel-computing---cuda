#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#define ITERATION 500
#define THREAD_NUM 10
__global__ static void  gd_cu(float x_data[], float y_data[], float b_grad[], float w_grad[], float b_w[],float lr_b_w[])
{
	const float lr = 1;
	const int tid = threadIdx.x;
	b_grad[tid] =  - 2.0 * (y_data[tid] - b_w[0] - (b_w[1] * x_data[tid])) * 1.0;
	w_grad[tid] =  - 2.0 * (y_data[tid] - b_w[0] - (b_w[1] * x_data[tid])) * x_data[tid];
	__syncthreads();
	if (tid == 0) {
		for (int i = 1; i < THREAD_NUM; i++) {
			b_grad[0] += b_grad[i];
			w_grad[0] += w_grad[i];
		}
		lr_b_w[0] = lr_b_w[0] + b_grad[0] * b_grad[0];
		lr_b_w[1] = lr_b_w[1] + w_grad[0] * w_grad[0];
		b_w[0] = b_w[0] - lr / sqrt(lr_b_w[0]) * b_grad[0];
		b_w[1] = b_w[1] - lr / sqrt(lr_b_w[1]) * w_grad[0];
	}
	//AdaGrad
}
int main() {
	float x_data[10] = { 338, 333, 328, 207,226, 25, 179, 60, 208, 606 };
	float y_data[10] = { 640, 633, 619, 393,428, 27, 193, 66, 226, 1591 };
	float b;
	float w;
	b = -120;
	w = -4;
	int lr = 1;
	float lr_b = 0;
	float lr_w = 0;
	int size_ = 10;
	float b_w_host[] = {-120.0,-4.0};
	float lr_b_w_host[] = {0,0};
	float* x_cuda, * y_cuda, * b_grad, * w_grad,* b_w,* lr_b_w;
	cudaMalloc((void**)&x_cuda, sizeof(float) * size_);
	cudaMalloc((void**)&y_cuda, sizeof(float) * size_);
	cudaMalloc((void**)&b_grad, sizeof(float) * size_);
	cudaMalloc((void**)&w_grad, sizeof(float) * size_);
	cudaMalloc((void**)&b_w, sizeof(float) * 2);
	cudaMalloc((void**)&lr_b_w, sizeof(float) * 2);

	cudaMemcpy(x_cuda, x_data, sizeof(float) * size_,
		cudaMemcpyHostToDevice);
	cudaMemcpy(y_cuda, y_data, sizeof(float) * size_,
		cudaMemcpyHostToDevice);
	cudaMemcpy(b_w, b_w_host, sizeof(float) * 2,
		cudaMemcpyHostToDevice);
	cudaMemcpy(lr_b_w, lr_b_w_host, sizeof(float) * 2,
		cudaMemcpyHostToDevice);

	float ans_host[2];
	for (int i = 0; i < ITERATION; i++) {
		gd_cu << <1, THREAD_NUM, 0 >> > (x_cuda, y_cuda,  b_grad,  w_grad,  b_w,  lr_b_w);
	}
	cudaMemcpy(&ans_host, b_w, sizeof(float) * 2,
		cudaMemcpyDeviceToHost);
	b = ans_host[0];
	w = ans_host[1];
	printf("=====GPU=====\n");
	printf("w=%f\n", w);
	printf("b=%f\n", b);
	printf("==============\n");
	return 0;
}
