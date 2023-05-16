#include <stdio.h>
#include "convolve.c"
int main(int argc, char *argv[])
{
	float MAX_RNAD_NUM = 10;
	const int isprint = 1;
	int m_R = 10;
    int m_C = 10;
    
    int k_R = 5;
    int k_C = 5;
    
	//conv 2D
//	float* mask = GenerateNumbers2D_float(m_R, m_C,MAX_RNAD_NUM);
//	float* kernal = GenerateNumbers2D_float(k_R, k_C,MAX_RNAD_NUM);
	int* mask = GenerateNumbers2D_int(m_R, m_C,MAX_RNAD_NUM);
	int* kernal = GenerateNumbers2D_int(k_R, k_C,MAX_RNAD_NUM);
	
	print_matrix_int(mask, m_R, m_C,isprint);
	print_matrix_int(kernal, k_R, k_C,isprint);
	
	int* output_2D = convolve2D_full_int(mask, kernal,m_R,m_C,k_R,k_C);
	
	print_matrix_int(output_2D, (m_R+k_R-1), (m_C+k_C-1),isprint);
	//print_matrix_float(output_2D, m_R, m_C,isprint);
	return 0;
}
