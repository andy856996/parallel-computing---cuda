#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void add(int *a, int *b, int *c, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

void random_ints(int* a, int n)
{
    for (int i = 0; i < n; ++i)
        a[i] = rand();
}

int main()
{
    int n = (20480 * 20480);
    int threads_per_block = 512;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    int size = n * sizeof(int);
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    a = (int*)malloc(size);
    random_ints(a, n);
    b = (int*)malloc(size);
    random_ints(b, n);
    c = (int*)malloc(size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    add<<<(n + threads_per_block - 1)/threads_per_block, threads_per_block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // for (int i = 0; i < n; ++i) {
    //     cout << c[i] << ",";
    // }
    cout << endl;
    cout << cudaGetErrorString(cudaGetLastError()) << endl;

    free(a);
    free(b);
    free(c);

    return 0;
}