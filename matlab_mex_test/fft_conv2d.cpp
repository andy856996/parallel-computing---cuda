/*----------------------------------------------
create by Ding Ze An
Date : 2023 5 23
National Taitung Unv.
Email : andy856996@gamil.com
----------------------------------------------*/
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

void complex_mat_multiplication(fftw_complex *matrix1, fftw_complex *matrix2,fftw_complex *result , int len){
	for (int i = 0; i < len; i++)
	{
		// Perform complex multiplication
		result[i][0] = matrix1[i][0] * matrix2[i][0] - matrix1[i][1] * matrix2[i][1]; // Real part
		result[i][1] = matrix1[i][0] * matrix2[i][1] + matrix1[i][1] * matrix2[i][0]; // Imaginary part
	}
}
void print_mat(double *inMatrix,int R,int C){
	for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            cout <<  inMatrix[i*C+j]<<" " ;
        }
		cout<<endl;
    }
}
void print_mat_complex(double *inMatrix,int R,int C){
	for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
			if (j%2==0){
				cout <<  inMatrix[i*C+j];
			}else{
				cout <<"+"<<inMatrix[i*C+j]<<" ";
			}
            
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
			//out[i*C+j] = in[i+j*R];
			out[i+j*R] = in[i*C+j];
		}
    }
}
void padding_FFT_conv(double *Mat,double *paddingMat,int R,int C,int R_padding,int C_padding){
	for (int i = 0; i < R; i++) {
		for (int j = 0;j<C;j++){
			paddingMat[i*C_padding+j] = Mat[i*C+j];
		}
    }
}
void take_realVar_matCom_devideN(double*input_complex,double *output,int R,int C){
	for (int i = 0; i < R; i++) {
		for (int j = 0;j<C;j++){
			output[i*C+j] = input_complex[i*C*2+j*2]/(R*C);
		}
    }
}
void print_mat_real(double *inMatrix,int R,int C){
	cout<<"R*C="<<R*C<<endl;
	for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
			//cout << inMatrix[i*C*2+j*2]/(R*C)<<" ";
			cout << inMatrix[i*C*2+j*2]<<" ";
        }
		cout<<endl;
    }
}
void fft_conv2d(double* mask, double* kernal,double *result, int R, int C)
{
    fftw_plan maskPlan;
	fftw_plan kernalPlan;
	fftw_plan resultPlan;
    fftw_complex* maskComplex = reinterpret_cast<fftw_complex*>(mask);
    fftw_complex* kernalComplex = reinterpret_cast<fftw_complex*>(kernal);
	fftw_complex* resultComplex = reinterpret_cast<fftw_complex*>(result);
	
    // Create the plan for the forward transform
    maskPlan = fftw_plan_dft_2d(R, C, maskComplex, maskComplex, FFTW_FORWARD, FFTW_ESTIMATE);
	kernalPlan = fftw_plan_dft_2d(R, C, kernalComplex, kernalComplex, FFTW_FORWARD, FFTW_ESTIMATE);
	
	fftw_execute(maskPlan);
	fftw_execute(kernalPlan);
	
	complex_mat_multiplication(maskComplex,kernalComplex,resultComplex,R*C);
	
	resultPlan = fftw_plan_dft_2d(R, C, resultComplex, resultComplex, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(resultPlan);
	
	//print_mat_real(result,R,C);
	
    fftw_destroy_plan(maskPlan);
	fftw_destroy_plan(kernalPlan);
	fftw_destroy_plan(resultPlan);
}
void mexFunction(int nOutputs, mxArray *output_pointers[], int nInputs, const mxArray *input_pointers[]) {


    int R_mask = mxGetM(input_pointers[0]);
    int C_mask = mxGetN(input_pointers[0]);
	double *mask;
    mask = mxGetPr(input_pointers[0]);
	
	int R_kernal = mxGetM(input_pointers[1]);
    int C_kernal = mxGetN(input_pointers[1]);
	double *kernal;
    kernal = mxGetPr(input_pointers[1]);
	//double *data = mxCreateDoubleMatrix(R, C, mxREAL);
	
	double *mask_doubleArr_complex = (double *) malloc(sizeof(double) * R_mask * C_mask * 2);
	double *kernal_doubleArr_complex = (double *) malloc(sizeof(double) * R_kernal * C_kernal * 2);
	
	put_data_in_comMat(mask,mask_doubleArr_complex,R_mask, C_mask);
	put_data_in_comMat(kernal,kernal_doubleArr_complex,R_kernal, C_kernal);
	
	double *maskPadding_doubleArr_complex = (double *) malloc(sizeof(double) * (R_mask+R_kernal-1)* (C_mask+C_kernal-1) * 2);
	double *kernalPadding_doubleArr_complex = (double *) malloc(sizeof(double) * (R_mask+R_kernal-1)* (C_mask+C_kernal-1) * 2);
	double *result_doubleArr_complex = (double *) malloc(sizeof(double) * (R_mask+R_kernal-1)* (C_mask+C_kernal-1) * 2);
	
	std::fill(maskPadding_doubleArr_complex, maskPadding_doubleArr_complex+(R_mask+R_kernal-1)* (C_mask+C_kernal-1) * 2, 0);
	std::fill(kernalPadding_doubleArr_complex, kernalPadding_doubleArr_complex+(R_mask+R_kernal-1)* (C_mask+C_kernal-1) * 2, 0);
	std::fill(result_doubleArr_complex, result_doubleArr_complex+(R_mask+R_kernal-1)* (C_mask+C_kernal-1) * 2, 0);
	
	padding_FFT_conv(mask_doubleArr_complex,maskPadding_doubleArr_complex,R_mask,C_mask*2,(R_mask+R_kernal-1),(C_mask+C_kernal-1) * 2);
	padding_FFT_conv(kernal_doubleArr_complex,kernalPadding_doubleArr_complex,R_kernal,C_kernal*2,(R_mask+R_kernal-1),(C_mask+C_kernal-1) * 2);

	fft_conv2d(maskPadding_doubleArr_complex, kernalPadding_doubleArr_complex,result_doubleArr_complex,(R_mask+R_kernal-1), (C_mask+C_kernal-1));
	
	output_pointers[0] = mxCreateDoubleMatrix((R_mask+R_kernal-1),(C_mask+C_kernal-1), mxREAL);
	
	double *result = (double *) malloc(sizeof(double) * (R_mask+R_kernal-1)* (C_mask+C_kernal-1));

	take_realVar_matCom_devideN(result_doubleArr_complex,result,(R_mask+R_kernal-1),(C_mask+C_kernal-1));
	
	put_mat_toMATLAB_mat(result,mxGetPr(output_pointers[0]),(R_mask+R_kernal-1),(C_mask+C_kernal-1));
}
