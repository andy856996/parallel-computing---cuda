
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <sys/time.h>
#define BLOCK_NUM  64
#define THREAD_NUM 1000
double seconds(){
    struct timeval tp; struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec+(double)tp.tv_usec*1.e-6);
}
int isequal(int* a, int* b, int size) {
    for (int i = 0; i < size; i++){
        if (a[i] != b[i]) {
            return 0;
        }
    }
    return 1;
}
void print_matrix(int a[], int R, int C,int isprint) {
    if (isprint ==0){
        return ;
    }
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
            number[C * i + j] = rand() % 100;
        }
    }
    return number;
}

int* matrixTranspose(int a[],int R, int C)
{
    int* matrix_T = (int*)calloc(C*R, sizeof(int));
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            matrix_T[j*R + i] = a[i*C+j]; 
        }
    }
    return matrix_T;
}
int* matrixSum(int a[],int b[],int R, int C)
{
    int* sum = (int*)calloc(R*C, sizeof(int));
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++)
        {
            sum[C * i + j] =a[C*i +j]+b[C*i + j];
        }
    }
    return sum;
}
__global__ static void  matrixSum_cu(int a[],int b[],int y[],int R, int C)
{
    int total_size = R*C;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    for (int i = bid*THREAD_NUM + tid; i < total_size; i += THREAD_NUM*BLOCK_NUM){
       y[i] = a[i] + b[i];
    }
}
__global__ static void matrixTranspose_cu(int a[],int y[],int R, int C)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    int idx_b = row*C + col;
    int idx_y = col*R + row;
    if (col < C && row < R){
        y[idx_y] = a[idx_b];
    }
}
int main(int argc, char** argv)
{
    double iStart_CPU, iElaps_CPU,iStart_GPU, iElaps_GPU;
    int isprint = 0;
    int R = atoi(argv[1]);
    int C = atoi(argv[2]);

    printf("R:%d ,C:%d\n",R,C);

    int total_size = R*C;
    int* a = GenerateNumbers2D_int(R, C);
    int* b = GenerateNumbers2D_int(R, C);
    print_matrix(a, R, C,isprint);
    print_matrix(b, R, C,isprint);  

    //CPU
    iStart_CPU = seconds();
    int* matrixSum_out = matrixSum(a,b,R, C);
    print_matrix(matrixSum_out, R, C,isprint);  
    int* y_cpu = matrixTranspose(matrixSum_out,R,C);
    iElaps_CPU = seconds() - iStart_CPU;
    printf("CPU Time : %lf(msec)\n",1000*iElaps_CPU);
    print_matrix(y_cpu,C,R,isprint);  
    
    // GPU
    int* a_cuda, * b_cuda, * y_cuda_intput ,*y_Transpose_cu;
    cudaMalloc((void**)&a_cuda, sizeof(int) * total_size);
    cudaMalloc((void**)&b_cuda, sizeof(int) *total_size);
    cudaMalloc((void**)&y_cuda_intput, sizeof(int) *total_size);
    cudaMalloc((void**)&y_Transpose_cu, sizeof(int) *total_size);

    
    cudaMemcpy(a_cuda, a, sizeof(int) * total_size,
        cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda, b, sizeof(int) * total_size,
        cudaMemcpyHostToDevice);

    // 定義 grid ,block
    int dimx22v2 = 10; int dimy22v2 = 10;
    dim3 block22v2(dimx22v2, dimy22v2);
    dim3 grid22v2((C+block22v2.x-1)/block22v2.x, (R+block22v2.y-1)/block22v2.y);

    iStart_GPU = seconds();
    // GPU sum matrix
    matrixSum_cu <<<BLOCK_NUM, THREAD_NUM, 0 >>> (a_cuda, b_cuda,y_cuda_intput, R,C);
    printf("function 1 :pass\n");
    matrixTranspose_cu <<< grid22v2, block22v2 >>>(y_cuda_intput,y_Transpose_cu,R,C);
        printf("function 1 :pass\n");
    iElaps_GPU = seconds() - iStart_GPU;

    //int host_y[R*C] = {0};
    int host_y_T[R*C] = {0};
    int host_y[R*C] = {0};
    cudaMemcpy(&host_y_T, y_Transpose_cu, sizeof(int) *total_size , 
              cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_y, y_cuda_intput, sizeof(int) *total_size , 
        cudaMemcpyDeviceToHost);


    printf("GPU Time : %7.3f(msec)\n",1000*iElaps_GPU);
    print_matrix(host_y, R,C,isprint);
    printf("(sum matrix )is two arrary equal(CPU/GPU) ? : %d\n",isequal(host_y,matrixSum_out,R*C));
    printf("(Transpose matrix )is two arrary equal(CPU/GPU) ? : %d\n",isequal(host_y_T,y_cpu,R*C));
    
    return 0;
}

