#include <stdlib.h>
#include <stdio.h>

// helper functions to get the min and max of two numbers
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) < (Y)) ? (Y) : (X))

/**
 the convolve function will have as input two arrays h and x.
 I will return a pointer to a new array, as well as,
 set the length of that array in lenY.
 The length of h and x must be specified as inputs.
*/
float* convolve(float h[], float x[], int lenH, int lenX, int* lenY);
int isequal(float* a, float* b, int size);
void printArr_float(float* p, int size);

void GenerateNumbers(float *number, int size);
int* GenerateNumbers2D_int(int R, int C);
float* GenerateNumbers2D_float(int R, int C);

int* matrixTranspose(int a[],int R, int C);
int* matrixSum(int a[],int b[],int R, int C);

void print_matrix_float(float a[], int R, int C,int isprint);
void print_matrix_int(int a[], int R, int C,int isprint);