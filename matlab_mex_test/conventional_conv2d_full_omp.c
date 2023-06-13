/*
Developer : LAI,WEI-LIN
National Taitung Unv. IPGIT
Date : 2023/5/30
Email : 10822112@gm.nttu.edu.tw
*/
#include <stdio.h>
#include <string.h>
#include "mex.h"

#include <omp.h>

void convolution2D(double* mask, double* kernel, double* output, int m_row, int m_col, int k_row, int k_col) {
    int i, j, m, n;
    int o_row = m_row + k_row - 1;
    int o_col = m_col + k_col - 1;

    #pragma omp parallel for private(i, j, m, n) shared(mask, kernel, output)
    for (i = 0; i < o_row; i++) {
        for (j = 0; j < o_col; j++) {
            double temp = 0.0;
            for (m = 0; m < k_row; m++) {
                for (n = 0; n < k_col; n++) {
                    int mask_row = i - m;
                    int mask_col = j - n;

                    if (mask_row >= 0 && mask_row < m_row && mask_col >= 0 && mask_col < m_col) {
                        int mask_index = mask_row * m_col + mask_col;
                        int kernel_index = m * k_col + n;
                        temp += mask[mask_index] * kernel[kernel_index];
                    }
                }
            }
            output[i * o_col + j] = temp;
        }
    }
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    double *mask;
    double *kernel;
    double *output;
    int mrows, mcols, krows, kcols;

    mask = mxGetPr(prhs[0]);
    kernel = mxGetPr(prhs[1]);

    mrows = mxGetM(prhs[0]);
    mcols = mxGetN(prhs[0]);

    krows = mxGetM(prhs[1]);
    kcols = mxGetN(prhs[1]);

    plhs[0] = mxCreateDoubleMatrix(mrows + krows - 1, mcols + kcols - 1, mxREAL);

    output = mxGetPr(plhs[0]);

    convolution2D(mask, kernel, output, mrows, mcols, krows, kcols);

    int i, j;
    int o_row = mrows + krows - 1;
    int o_col = mcols + kcols - 1;
    double *temp = malloc(o_row * o_col * sizeof(double));
    for (i = 0; i < o_row; i++) {
        for (j = 0; j < o_col; j++) {
            temp[j * o_row + i] = output[i * o_col + j];
        }
    }
    memcpy(output, temp, o_row * o_col * sizeof(double));
    free(temp);
}
