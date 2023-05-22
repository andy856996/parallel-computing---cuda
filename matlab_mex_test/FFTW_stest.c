#include <stdio.h>
#include <stdlib.h>

#include "fftw3.h"
#include "mex.h"

void fft2D(double* in, double* out, int width, int height)
{
    fftw_plan plan;
    fftw_complex* inComplex = reinterpret_cast<fftw_complex*>(in);
    fftw_complex* outComplex = reinterpret_cast<fftw_complex*>(out);

    // Create the plan for the forward transform
    plan = fftw_plan_dft_2d(width, height, inComplex, outComplex, FFTW_FORWARD, FFTW_ESTIMATE);

    // Execute the forward transform
    fftw_execute(plan);

    // Destroy the plan
    fftw_destroy_plan(plan);
}

void vector2matrix(double *input, int nRows, int nCols, double **output) {
    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            output[i][j] = input[j * nRows + i];
        }
    }
}

void matrix2vector(double **input, int nRows, int nCols, double *output) {
    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            output[j * nRows + i] = input[i][j];
        }
    }
}

void to_real_img_arr(fftw_complex **in,double **real,double **img,int M,int N){
	for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            real[i][j] = in[i][j][0];
			img[i][j] = in[i][j][1];
        }
    }
}
void put_data_in_comMat(double *input,double *output,int len){
	for (int i = 0; i < len; ++i) {
		output[2*i] = input[i];
    }
}
void mexFunction(int nOutputs, mxArray *output_pointers[], int nInputs, const mxArray *input_pointers[]) {
    if (nInputs != 1) {
        std::cerr << "The number of input parameters must be exactly 1 (only for the real signal)!" << std::endl;
        return;
    }
    
    if (nOutputs != 2) {
        std::cerr << "The number of output parameters must be exactly 2 (real and imaginary parts)!" << std::endl;
        return;
    }
	
    
	
    // M: number of rows
    // N: number of columns
    int height = mxGetM(input_pointers[0]);
    int width = mxGetN(input_pointers[0]);
	
	double *inMatrix = mxGetDoubles(input_pointers[0]);
	
	double *data = (double *) malloc(sizeof(double) * width * height * 2);
	double *output = (double *) malloc(sizeof(double) * width * height * 2);
	
	put_data_in_comMat(inMatrix,data,width * height);
	
	fft2D(inMatrix, output, width, height);
	
	
	
    for (int i = 0; i < width * height; i++) {
        std::cerr << "(" << data[2 * i] << ", " << data[2 * i + 1] << ")" << std::endl;
    }
	
    output_pointers[0] = mxCreateDoubleMatrix(height, width, mxREAL);
	output_pointers[1] = mxCreateDoubleMatrix(height, width, mxREAL);
    //matrix2vector(img, M, N, mxGetPr(output_pointers[1]));
}