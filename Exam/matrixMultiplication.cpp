#include <iostream>
using namespace std;
void printArr_float(int* p, int size) {
    printf("-------printArrStart------\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", p[i]);
    }
    printf("\n-----printArrEnd--------\n\n");
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

int* GenerateNumbers2D_int(int R,int C)
{
    int* number = (int*)calloc(R*C, sizeof(int));
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++)
        {
            number[C*i + j] = rand() % 10;
        }
    }
    return number;
}
void print_matrix(int a[],int R,int C) {
	printf("\n-----------\n");
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++){
            printf("%d ,", a[C * i + j]);
        }
        printf("\n");
    }
    printf("-----------\n");
}
int main(){
	int a_R = 400;
	int a_C = 300;
	int b_R = 300;
	int b_C = 500;
	int* a = GenerateNumbers2D_int(a_R,a_C);
	int* b = GenerateNumbers2D_int(b_R,b_C);
	int* y = matrixMultiplication(a, b, a_R, a_C, b_R, b_C);
//	print_matrix(y, a_R,b_C);
//	printf("a : ");
//	print_matrix(a, a_R,a_C);
//	printf("\nb : ");
//	print_matrix(b, b_R,b_C);
	return 0;
}