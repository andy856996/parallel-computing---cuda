#include "convolve.h"
// convolution 1D int//
int* convolve_int(int h[], int x[], int lenH, int lenX, int* lenY)
{
  int nconv = lenH+lenX-1;
  (*lenY) = nconv;
  int i,j,h_start,x_start,x_end;

  int *y = (int*) calloc(nconv, sizeof(int));

  for (i=0; i<nconv; i++)
  {
    x_start = MAX(0,i-lenH+1);
    x_end   = MIN(i+1,lenX);
    h_start = MIN(i,lenH-1);
    for(j=x_start; j<x_end; j++)
    {
      y[i] += h[h_start--]*x[j];
    }
  }
  return y;
}
// convolution 1D float//
float* convolve_float(float h[], float x[], int lenH, int lenX, int* lenY)
{
  int nconv = lenH+lenX-1;
  (*lenY) = nconv;
  int i,j,h_start,x_start,x_end;

  float *y = (float*) calloc(nconv, sizeof(float));

  for (i=0; i<nconv; i++)
  {
    x_start = MAX(0,i-lenH+1);
    x_end   = MIN(i+1,lenX);
    h_start = MIN(i,lenH-1);
    for(j=x_start; j<x_end; j++)
    {
      y[i] += h[h_start--]*x[j];
    }
  }
  return y;
}
// convolution 2D int//
int* convolve2D_int(int mask[], int kernal[],int m_R,int m_C,int k_R,int k_C)
{
	int nconv_out = m_R*m_C;
	int *y_out = (int*) calloc(nconv_out, sizeof(int));
	int m = k_R;
	int n = k_C;
	int hx = m / 2;
	int hy = n / 2;
	for (int i = 0; i < m_R; ++i)  // rows
	{
	    for (int j = 0; j < m_C; ++j)  // columns
	    {
	        for (int x = 0; x < m; ++x)  // kernel rows
	        {
	            for (int y = 0; y < n; ++y)  // kernel columns
	            {
	                int ii = i + x - hx;
	                int jj = j + y - hy;
	                // ignore input samples which are out of bound
	                if (ii >= 0 && ii < m_R && jj >= 0 && jj <= m_C)
	                    y_out[i*m_C+ j] += kernal[(m - x - 1) * k_C+ (n - y - 1) ] * mask[ii*m_C+ jj];
	            }
	        }
	    }
	}
	return y_out;
}
// convolution 2D float//
float* convolve2D_float(float mask[], float kernal[],int m_R,int m_C,int k_R,int k_C)
{
	int nconv_out = m_R*m_C;
	float *y_out = (float*) calloc(nconv_out, sizeof(float));
	int m = k_R;
	int n = k_C;
	int hx = m / 2;
	int hy = n / 2;
	for (int i = 0; i < m_R; ++i)  // rows
	{
	    for (int j = 0; j < m_C; ++j)  // columns
	    {
	        for (int x = 0; x < m; ++x)  // kernel rows
	        {
	            for (int y = 0; y < n; ++y)  // kernel columns
	            {
	                int ii = i + x - hx;
	                int jj = j + y - hy;
	                // ignore input samples which are out of bound
	                if (ii >= 0 && ii < m_R && jj >= 0 && jj <= m_C)
	                    y_out[i*m_C+ j] += kernal[(m - x - 1) * k_C+ (n - y - 1) ] * mask[ii*m_C+ jj];
	            }
	        }
	    }
	}
	return y_out;
}
int isequal(float* a, float* b, int size) {
    for (int i = 0; i < size; i++){
        if (a[i] != b[i]) {
            return 0;
        }
    }
    return 1;
}
int* GenerateNumbers2D_int(int R, int C,int MAX_RNAD_NUM)
{
    int* number = (int*)calloc(R * C, sizeof(int));
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++)
        {
            number[C * i + j] = rand() % MAX_RNAD_NUM;
        }
    }
    return number;
}
float* GenerateNumbers2D_float(int R, int C,float MAX_RNAD_NUM)
{
    float* number = (float*)calloc(R * C, sizeof(float));
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++)
        {
            number[C * i + j] =  (float) rand() / (RAND_MAX + 1.0) * MAX_RNAD_NUM;
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
void printArr_float(float* p, int size) {
    printf("-------printArrStart------\n");
    for (int i = 0; i < size; i++) {
        printf("%0.f ", p[i]);
    }
    printf("\n-----printArrEnd--------\n\n");
}
void printArr_int(int* p, int size) {
    printf("-------printArrStart------\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", p[i]);
    }
    printf("\n-----printArrEnd--------\n\n");
}
void GenerateNumbers(float *number, int size)
{
    for(int i = 0; i < size; i++) {
        number[i] = rand() % 10;
    }
}
void print_matrix_int(int a[], int R, int C,int isprint) {
    if (isprint ==0){
        return ;
    }
    printf("\n-----------\n");
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            printf("%d ", a[C * i + j]);
        }
        printf(";\n");
    }
    printf("-----------\n");
}
void print_matrix_float(float a[], int R, int C,int isprint) {
    if (isprint ==0){
        return ;
    }
    printf("\n-----------\n");
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            printf("%f ", a[C * i + j]);
        }
        printf(";\n");
    }
    printf("-----------\n");
}