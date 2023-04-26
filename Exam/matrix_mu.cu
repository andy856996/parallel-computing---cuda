
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>
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
int* matrixMultiplication(int a[], int b[], int a_R, int a_C, int b_R, int b_C)
{
    int t = 0;
    int nconv = (a_R * b_C);
    int* y = (int*)calloc(nconv, sizeof(int));

    for (int i = 0; i < a_R; i++) {
        for (int j = 0; j < b_C; j++) {
            t = 0;
            for (int k = 0; k < a_C; k++)
            {
                t += a[i * a_C + k] * b[b_C * k + j];
            }
            y[i * b_C + j] = t;
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
void GenerateNumbers1D(float *number, int size)
{
    for(int i = 0; i < size; i++) {
        number[i] = rand() % 10;
    }
}
void print_matrix(int a[], int R, int C) {
    printf("\n-----------\n");
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            printf("%d ,", a[C * i + j]);
        }
        printf("\n");
    }
    printf("-----------\n");
}
int* GenerateNumbers2D_int(int R, int C)
{
    int* number = (int*)calloc(R * C, sizeof(int));
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++)
        {
            number[C * i + j] = rand() % 10;
        }
    }
    return number;
}
__global__ static void matrixMultiplication_cu(int a[], int b[], int y[], int a_R, int a_C, int b_R, int b_C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < a_R && col < b_C) {
        int t = 0;
        for (int k = 0; k < a_C; k++)
        {
            t += a[row * a_C + k] * b[b_C * k + col];
        }
        y[row * b_C + col] = t;
    }
}





__global__ static void convolve_cu(float *h, float *x,float *y)
{
    int nconv = DATA_SIZE * 2 - 1;
    int j, h_start, x_start, x_end;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    //const int size = nconv / THREAD_NUM;
    
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

    int a_R = 40;
    int a_C = 30;
    int b_R = 30;
    int b_C = 50;
    int* a = GenerateNumbers2D_int(a_R, a_C);
    int* b = GenerateNumbers2D_int(b_R, b_C);
    int* y = matrixMultiplication(a, b, a_R, a_C, b_R, b_C);
    print_matrix(y, a_R, b_C);



    int nconv = (a_R * b_C);
    int* y_cuda_out = (int*)calloc(nconv, sizeof(int));

    int* a_cuda, * b_cuda, * y_cuda_intput;

    cudaMalloc((void**)&a_cuda, sizeof(int) * (a_R * a_C));
    cudaMalloc((void**)&b_cuda, sizeof(int) * (b_R * b_C));
    cudaMalloc((void**)&y_cuda_intput, sizeof(int) * (a_R * b_C));

    
    cudaMemcpy(a_cuda, a, sizeof(int) * (a_R * a_C),
        cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda, b, sizeof(int) * (b_R * b_C),
        cudaMemcpyHostToDevice);

    dim3 dimBlock(16,16);
    dim3 dimGrid((b_C + dimBlock.x -1)/dimBlock.x,(a_R + dimBlock.y - 1)/dimBlock.y);

    matrixMultiplication_cu << < dimGrid, dimBlock >> > (a_cuda, b_cuda, y_cuda_intput, a_C, a_R, b_R, b_C);

    cudaMemcpy(&y_cuda_out, y_cuda_intput, sizeof(int) * nconv,
        cudaMemcpyDeviceToHost);
    
    print_matrix(y_cuda_out, a_R, b_C);

    cudaFree(a_cuda);
    cudaFree(b_cuda);
    cudaFree(y_cuda_intput);
    //

    /*print_matrix(y, a_R, b_C);
    printf("a : ");
    print_matrix(a, a_R,a_C);
    printf("\nb : ");
    print_matrix(b, b_R,b_C);*/


    //float data1[DATA_SIZE] = { 0 };
    //float data2[DATA_SIZE] = { 0 };
    
    //產生隨機數字
    //GenerateNumbers1D(data1, DATA_SIZE);
    //GenerateNumbers1D(data2, DATA_SIZE);
    //印出數字
    //printArr_float(data1,DATA_SIZE);
    //printArr_float(data2,DATA_SIZE);

    // compute w/ CPU
    /*int lenY;
    clock_t t1, t2;
    t1 = clock();
    float* y = convolve(data1, data2, DATA_SIZE, DATA_SIZE, &lenY);
    t2 = clock();
    printf("Convolution 1D using CPU : %lf(s)\n", (t2 - t1) / (double)(CLOCKS_PER_SEC));*/
    


    /*for (int i = 0; i < lenY; i++) {
        printf("%0.f ", y[i]);
    }
    puts("");*/
    

    //compute w/ GPU
    /*float* gpudata1, * gpudata2, * result;

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
    free(y);*/
    

    return 0;
}

