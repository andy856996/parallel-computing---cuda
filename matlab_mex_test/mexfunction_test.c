/*==========================================================
 * conv2D.c
 * creater : Ding Ze An
 * National Taitung Unv.
 *========================================================*/

#include "mex.h"

void convolution2D(double *mask, double *kernel, double *output, mwSize m_row,mwSize m_col, mwSize k_row,mwSize k_col) {
    int i, j, m, n;
    double sum;
    
    for (i = 0; i < m_row; i++) {
        for (j = 0; j < m_col; j++) {

            sum = 0;

            for (m = 0; m < k_row; m++) {
                for (n = 0; n < k_col; n++) {
                    if (i + m >= k_row / 2 && i + m < m_row + k_row / 2 &&
                        j + n >= k_col / 2 && j + n < m_col + k_col / 2) {
                        sum += mask[(i + m - k_row / 2)*m_col+(j + n - k_col / 2)] * kernel[m*k_col+n];
                    }
                }
            }
            output[i*m_col+j] = sum;
        }
    }
}
/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double *mask;
    size_t mcols;
    size_t mrows;
    
    double *kernal;
    size_t kcols;
    size_t krows;

    double *outMatrix;              /* output matrix */

    /* get the value of the scalar input  */
    //multiplier = mxGetScalar(prhs[0]);

    /* create a pointer to the real data in the input matrix  */
    #if MX_HAS_INTERLEAVED_COMPLEX
    mask = mxGetDoubles(prhs[0]);
    kernal = mxGetDoubles(prhs[1]);
    #else
    mask = mxGetPr(prhs[0]);
    kernal = mxGetPr(prhs[1]);
    #endif
    

    /* get dimensions of the input matrix */

    mrows = mxGetM(prhs[0]);
    mcols = mxGetN(prhs[0]);

    krows = mxGetM(prhs[1]);
    kcols = mxGetN(prhs[1]);

    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix((mwSize)mrows,(mwSize)mcols,mxREAL);

    /* get a pointer to the real data in the output matrix */
    #if MX_HAS_INTERLEAVED_COMPLEX
    outMatrix = mxGetDoubles(plhs[0]);
    #else
    outMatrix = mxGetPr(plhs[0]);
    #endif
    convolution2D(mask,kernal,outMatrix,(mwSize)mrows,(mwSize)mcols,(mwSize)krows,(mwSize)kcols);
}
