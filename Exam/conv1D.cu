
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <algorithm>
#include<iostream>
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) < (Y)) ? (Y) : (X))

#define DATA_SIZE 55000
#define BLOCK_NUM  64
#define THREAD_NUM 1024
bool InitCUDA()
{
    int count;
    cudaGetDeviceCount(&count);
    if (count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }
    int i;
    for (i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) {
                break;
            }
        }
    }
    if (i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }
    cudaSetDevice(i);
    return true;
}
float* convolve(float h[], float x[], int lenH, int lenX, int* lenY)
{
    int nconv = lenH + lenX - 1;
    (*lenY) = nconv;
    int i, j, h_start, x_start, x_end;
    float* y = (float*)calloc(nconv, sizeof(float));
    for (i = 0; i < nconv; i++)
    {
        x_start = MAX(0, i - lenH + 1);
        x_end = MIN(i + 1, lenX);
        h_start = MIN(i, lenH - 1);
        for (j = x_start; j < x_end; j++)
        {
            y[i] += h[h_start--] * x[j];
        }
    }
    return y;
}
int isequal(float* a, float* b, int size) {
    for (int i = 0; i < size; i++){
        if (a[i] != b[i]) {
            return 0;
        }
    }
    return 1;
}
void printArr_float(float* p, int size) {
    printf("-------printArrStart------\n");
    for (int i = 0; i < size; i++) {
        printf("%0.f ", p[i]);
    }
    printf("\n-----printArrEnd--------\n\n");
}
void GenerateNumbers(float *number, int size)
{
    for(int i = 0; i < size; i++) {
        number[i] = rand() % 10;
    }
}
__global__ static void convolve_cu(float *h, float *x,float *y)
{
    int nconv = DATA_SIZE * 2 - 1;
    int j, h_start, x_start, x_end;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int size = nconv / THREAD_NUM;
    
    for (int i = bid*THREAD_NUM + tid; i < nconv; i += THREAD_NUM*BLOCK_NUM)
    //for (int i = tid; i < nconv; i += THREAD_NUM)
    {
        x_start = MAX(0, i - DATA_SIZE + 1);
        x_end = MIN(i + 1, DATA_SIZE);
        h_start = MIN(i, DATA_SIZE - 1);
        for (j = x_start; j < x_end; j++)
        {
            y[i] += h[h_start--] * x[j];
            
        }
    }
}

int main(int argc, char** argv)
{
    if (!InitCUDA()) {
        return 0;
    }

    printf("CUDA initialized.\n");

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("using Device %d: %s\n\n", dev, deviceProp.name);
    printf("**************************\n");
    printf("**************************\n");

    float data1[DATA_SIZE] = { 0 };
    float data2[DATA_SIZE] = { 0 };
    
    //產生隨機數字
    GenerateNumbers(data1, DATA_SIZE);
    GenerateNumbers(data2, DATA_SIZE);
    //印出數字
    //printArr_float(data1,DATA_SIZE);
    //printArr_float(data2,DATA_SIZE);

    // compute w/ CPU
    int lenY;
    clock_t t1, t2;
    t1 = clock();
    float* y = convolve(data1, data2, DATA_SIZE, DATA_SIZE, &lenY);
    t2 = clock();
    printf("Convolution 1D using CPU : %lf(s)\n", (t2 - t1) / (double)(CLOCKS_PER_SEC));

    /*for (int i = 0; i < lenY; i++) {
        printf("%0.f ", y[i]);
    }
    puts("");*/
    

    //compute w/ GPU
    float* gpudata1, * gpudata2, * result;

    cudaMalloc((void**)&gpudata1, sizeof(float) * DATA_SIZE);
    cudaMalloc((void**)&gpudata2, sizeof(float) * DATA_SIZE);
    cudaMalloc((void**)&result, sizeof(float) * (DATA_SIZE * 2 - 1));

    cudaMemcpy(gpudata1, data1, sizeof(float) * DATA_SIZE,
        cudaMemcpyHostToDevice);
    cudaMemcpy(gpudata2, data2, sizeof(float) * DATA_SIZE,
        cudaMemcpyHostToDevice);
    t1 = clock();
    convolve_cu <<<BLOCK_NUM, THREAD_NUM, 0 >>> (gpudata1, gpudata2, result);
    

    float ans[DATA_SIZE * 2 - 1];
    cudaMemcpy(&ans, result, sizeof(float) * (DATA_SIZE * 2 - 1) , 
        cudaMemcpyDeviceToHost);
    t2 = clock();
    printf("Convolution 1D using GPU : %lf(s)\n", (t2 - t1) / (double)(CLOCKS_PER_SEC));
    
    //printArr_float(ans,DATA_SIZE * 2 - 1);
    printf("is CPU & GPU answer equal ? : %s\n\n", (isequal(ans, y, (DATA_SIZE * 2 - 1)) ? "true" : "false"));
    free(y);
    return 0;
}

