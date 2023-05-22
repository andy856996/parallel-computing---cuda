#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>
#include <ctime>

#include "fftw3.h"
#include "mex.h"

using namespace std;

void fft2D(double* in, double* out, int R, int C)
{
    fftw_plan plan;
    fftw_complex* inComplex = reinterpret_cast<fftw_complex*>(in);
    fftw_complex* outComplex = reinterpret_cast<fftw_complex*>(out);

    // Create the plan for the forward transform
    plan = fftw_plan_dft_2d(R, C, inComplex, outComplex, FFTW_FORWARD, FFTW_ESTIMATE);

    // Execute the forward transform
    fftw_execute(plan);

    // Destroy the plan
    fftw_destroy_plan(plan);
}

void print_mat(double *inMatrix,int R,int C){
	for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            cout <<  inMatrix[i*C+j]<<" " ;
        }
		cout<<endl;
    }
}

void print_arr(double *inMatrix,int len){
	for (int i = 0; i < len; ++i) {
        cout <<  inMatrix[i]<<" " ;
    }
}
void put_data_in_comMat(double *input,double *output,int R,int C){
	for (int i = 0; i < R; i++) {
		for (int j = 0;j<C;j++){
			output[(C*2)*i+j*2] = input[i+j*R];
			output[(C*2)*i+j*2+1] = 0.0;
		}
    }
}
void put_mat_toMATLAB_mat(double *in,double *out,int R,int C){
	for (int i = 0; i < R; i++) {
		for (int j = 0;j<C;j++){
			out[i*C+j] = in[i+j*R];
		}
    }
}
void mexFunction(int nOutputs, mxArray *output_pointers[], int nInputs, const mxArray *input_pointers[]) {

    // M: number of rows
    // N: number of columns
    int R = mxGetM(input_pointers[0]);
    int C = mxGetN(input_pointers[0]);
	double *inMatrix;
    inMatrix = mxGetPr(input_pointers[0]);
	
	//double *data = mxCreateDoubleMatrix(R, C, mxREAL);
	
	double *data = (double *) malloc(sizeof(double) * R * C * 2);
	double *output = (double *) malloc(sizeof(double) * R * C * 2);
	
	//print_arr(inMatrix, R*C);
	
	//print_mat(inMatrix, R, C);
	put_data_in_comMat(inMatrix,data,R, C);
	//print_mat(data, R, C*2);
	
	fft2D(data, output, R, C);
	print_mat(output,R,C*2);
	
	output_pointers[0] = mxCreateDoubleMatrix(R, C*2, mxREAL);
	put_mat_toMATLAB_mat(output,mxGetPr(output_pointers[0]),R,C*2);
}