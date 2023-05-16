#include "convolve.h"
#include <string.h>
#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) < (Y)) ? (Y) : (X))
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
// convolution 2D full//
int* convolve2D_full_int(int mask[], int kernal[],int m_R,int m_C,int k_R,int k_C)
{
	int *mask_pad = (int*)malloc(sizeof(int) * (m_R + (k_R-1) * 2) * (m_C + (k_C-1) * 2));
	int *kernal_pad = (int*)malloc(sizeof(int) * (k_R + (m_R-1) * 2) * (k_C + (m_C-1) * 2) );
	// fill zero
	memset(mask_pad, 0, sizeof(int) * (m_R + (k_R-1) * 2) * (m_C + (k_C-1) * 2));
	memset(kernal_pad, 0, sizeof(int) * (k_R + (m_R-1) * 2) * (k_C + (m_C-1) * 2));
	
	const int mask_pad_R = (m_R + (k_R-1) * 2);const int mask_pad_C = (m_C + (k_C-1) * 2);
	const int kernal_pad_R = (k_R + (m_R-1) * 2);const int kernal_pad_C = (k_C + (m_C-1) * 2);

	int mp_idx_R = 0;int mp_idx_C = 0;
	for(int i = k_R-1;i<=(k_R-1)+(m_R)-1;i++){
		mp_idx_C = 0;
		for(int j = k_C-1;j<=(k_C-1)+(m_C)-1;j++){
			mask_pad[i*mask_pad_C+j] = mask[mp_idx_R*m_C+mp_idx_C];
			mp_idx_C ++;
		}
		mp_idx_R++;
	}
	print_matrix_int(mask_pad, mask_pad_R,mask_pad_C,1);
	int ker_idx_R = 0;int ker_idx_C = 0;
	for(int i = m_R-1;i<=(m_R-1)+(k_R)-1;i++){
		ker_idx_C = 0;
		for(int j = m_C-1;j<=(m_C-1)+(k_C)-1;j++){
			kernal_pad[i*kernal_pad_C+j] = kernal[ker_idx_R*k_C+ker_idx_C];
			ker_idx_C ++;
		}
		ker_idx_R++;
	}
	print_matrix_int(kernal_pad, kernal_pad_R,kernal_pad_C,1);
	int *y_out = (int*)malloc(sizeof(int) * (m_R+k_R-1) * (m_C+k_C-1));
	memset(y_out, 0, sizeof(int) * (m_R+k_R-1) * (m_C+k_C-1));
	int sum = 0;
	int row_start,row_end,col_start,col_end;
	for (int i = 0; i < (m_R+k_R-1); ++i)  // rows
	{
	    for (int j = 0; j < (m_C+k_C-1); ++j)  // columns
	    {
	    	row_start = max(i-m_R, 0);
	    	row_end = min(i,k_R-1);
            col_start = max(j-m_C, 0);
            col_end = min(j,k_C-1);
            
	    	sum= 0;
	        for (int k = row_start; k <= row_end; ++k)  // kernel rows
	        {
	            for (int l = col_start; l <= col_end; ++l)  // kernel columns
	            {
	            	sum += mask_pad[(i-k+k_R)*mask_pad_C+ (j-l+k_C)] * kernal_pad[k*kernal_pad_C+l];
	            	printf("mp : %d\n",mask_pad[(i-k+k_R)*mask_pad_C+ (j-l+k_C)]);
	            	printf("ker : %d\n",kernal_pad[k*kernal_pad_C+l]);
	            	//y_out[i*m_C+j] += mask_pad[(i+x)*(m_C+k_C-1)+j+y]*kernal_pad[x*k_C+y];
	            }
				y_out[i*(m_C+k_C-1)+j] = sum;
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