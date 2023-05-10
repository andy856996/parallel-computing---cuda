#include <stdio.h>
#include "convolve.c"
int main(int argc, char *argv[])
{
	float MAX_RNAD_NUM = 10;
	const int isprint = 1;
	int m_R = 600;
    int m_C = 600;
    
    int k_R = 200;
    int k_C = 200;
    
	//conv 2D
	float* mask = GenerateNumbers2D_float(m_R, m_C,MAX_RNAD_NUM);
	float* kernal = GenerateNumbers2D_float(k_R, k_C,MAX_RNAD_NUM);

	float* output_2D = convolve2D_float(mask, kernal,m_R,m_C,k_R,k_C);
	
	//print_matrix_float(mask, m_R, m_C,isprint);
	//print_matrix_float(kernal, k_R, k_C,isprint);
	
	//print_matrix_float(output_2D, m_R, m_C,isprint);
	return 0;
}
