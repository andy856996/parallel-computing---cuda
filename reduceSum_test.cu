#include <cuda_runtime.h>

#include <stdio.h>
//#include <sys/time.h>
int max(int x[],int k)
{
	int t,i;
	t=x[0];
	for(i=1;i<k;i++)
        {
		if(x[i]>t)
			t=x[i];
	}
	return(t);
}
// double seconds() {
//   struct timeval tp;
//   struct timezone tzp;
//   int i = gettimeofday( & tp, & tzp);
//   return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
// }
//Recursive CPU function of Interleaved Pair
int recursiveReduce(int * data, int const size) {
  if (size == 1) return data[0]; // renew the stride
  int const stride = size / 2;
  // in-place reduction
  for (int i = 0; i < stride; i++) {
    if(data[i] < data[i + stride]){
        data[i] = data[i + stride];
    }
    //data[i] += data[i + stride];
  }
  return recursiveReduce(data, stride); // call recursively
}

// kernel 1: Neighbored pair with divergence
__global__ void reduceNeighbored(int * g_idata, int * g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //用輸入的陣列起始位址＋區塊位址偏移量來產生thread所屬區塊的起始位址
  int * idata = g_idata + blockIdx.x * blockDim.x;
  if (idx >= n) return; // 邊界檢查
  // in-place reduction
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0)
      //idata[tid] += idata[tid + stride];

		if(idata[tid] < idata[tid + stride]){
		    idata[tid] = idata[tid + stride];
		}

    // 同一block內的threads同步，即先到者等待所有其它threads抵達
    __syncthreads();
  }
  //將此block計算的結果寫到輸出陣列
  if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char ** argv) {
  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties( & deviceProp, dev);
  printf("%s starting reduction at ", argv[0]);
  printf("device %d: %s ", dev, deviceProp.name);
  cudaSetDevice(dev);
  bool bResult = false;
  // total number of elements to reduce
  int size = 1 << 24;
  printf("with array size %d ", size);
  int blocksize = 512; // initial block size
  if (argc > 1) blocksize = atoi(argv[1]);
  dim3 block(blocksize, 1);
  dim3 grid((size + block.x - 1) / block.x, 1);
  printf("grid %d block %d\n", grid.x, block.x);

  // allocate host memory
  size_t bytes = size * sizeof(int);
  int * h_idata = (int * ) malloc(bytes);
  int * h_odata = (int * ) malloc(grid.x * sizeof(int));
  int * tmp = (int * ) malloc(bytes);

  // initialize the array
  for (int i = 0; i < size; i++)
    //mask off high 2 bytes to force max num to 255
    h_idata[i] = (int)(rand() & 0xFF);
  memcpy(tmp, h_idata, bytes);
  double iStart, iElaps;
  int gpu_sum = 0;
  // allocate device memory
  int * d_idata = NULL;
  int * d_odata = NULL;
  cudaMalloc((void ** ) & d_idata, bytes);
  cudaMalloc((void ** ) & d_odata, grid.x * sizeof(int));

  int *tmp_value = tmp;
  printf(" %d\n",max(tmp_value,size));

  // cpu reduction
  //iStart = seconds();
  int cpu_sum = recursiveReduce(tmp, size);
  //iElaps = seconds() - iStart;
  iElaps = 3;
  printf("cpu reduce elapsed %7.3f msec cpu_sum: %d\n", 1000 * iElaps, cpu_sum);
  
  //Invoking kernel 1: reduceNeighbored with divergence
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  //iStart = seconds();
  reduceNeighbored << < grid, block >>> (d_idata, d_odata, size);
  cudaDeviceSynchronize();
  //iElaps = seconds() - iStart;
  cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  //在此將kernel分別計算出的每個block總和再加總
  gpu_sum = 0;
  int max_num = 0;
  for (int i = 0; i < grid.x; i++)
    if(max_num<h_odata[i]){
		max_num =h_odata[i];
	}
  gpu_sum = max_num;
  printf("gpu Neighbored elapsed %7.3f msec gpu_sum: %d <<<grid %d block %d>>>\n", 1000 * iElaps, gpu_sum, grid.x, block.x);

  
  free(h_idata);
  free(h_odata);
  cudaFree(d_idata);
  cudaFree(d_odata);
  cudaDeviceReset();
  bResult = (gpu_sum == cpu_sum);
  if (!bResult) printf("Test failed!\n");
  return EXIT_SUCCESS;
}
