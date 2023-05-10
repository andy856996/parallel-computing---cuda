#include <stdio.h>
#include "convolve.c"
#define ARR_SIZE 100000
int main(int argc, char *argv[])
{
	float h[] = {1, 7, 1, 1, 5 ,2 ,7 ,6};
	float x[] = {1 ,4 ,2 ,3};
//	GenerateNumbers(h, 8);
//	GenerateNumbers(x, 4);
	
	int lenY;
	float *y = convolve_float(h,x,8,4,&lenY);
	
	printArr_float(y, lenY);
	
	printf("\nis equal ? : %d \n",isequal(y,y,lenY));
	free(y);
	return 0;
}
