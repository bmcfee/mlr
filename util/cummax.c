/* CREATED:2010-01-08 17:25:37 by Brian McFee <bmcfee@cs.ucsd.edu> */
/* cummax.c
 *
 *  cumulative maximum (analogous to cumsum)
 *
 * Compile: 
 *  mex cummax.c
 */

#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Declare variables */
    int n;      /* number of elements */
    double *pi; /* input array */
    double *po1, *po2; /* output array(s) */
    double currentMax;
    double currentMaxPointer;
    int i;

    if (nrhs != 1) {
        mexErrMsgTxt("Only one input argument is required.");
    }
    if (nlhs > 2) {
        mexErrMsgTxt("Too many output arguments.");
    }

    if (!(mxIsDouble(prhs[0]))) {
        mexErrMsgTxt("Input array must be of type double.");
    }

    n       = mxGetNumberOfElements(prhs[0]);
    pi      = (double *)mxGetPr(prhs[0]);
    
    plhs[0] = mxCreateDoubleMatrix(1, n, mxREAL);
    po1     = mxGetPr(plhs[0]);

    plhs[1] = mxCreateDoubleMatrix(1, n, mxREAL);
    po2     = mxGetPr(plhs[1]);

    /* Now for the meat */

    currentMax = pi[0];
    currentMaxPointer = 1;
    for (i = 0; i < n; i++) {
        if (pi[i] > currentMax) {
            currentMax          = pi[i];
            currentMaxPointer   = i + 1;
        }
        po1[i] = currentMax;
        po2[i] = currentMaxPointer;
    }

    mxSetN(plhs[0], n);
    mxSetN(plhs[1], n);

}
