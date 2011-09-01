/* CREATED:2010-01-17 08:12:57 by Brian McFee <bmcfee@cs.ucsd.edu> */
/* binarysearch.c
 *
 *  binary search
 *
 * Compile: 
 *  mex -DNAN_EQUALS_ZERO binarysearch.c
 */

#include "mex.h"


#if NAN_EQUALS_ZERO
#define IsNonZero(d) ((d) != 0.0 || mxIsNan(d))
#else
#define IsNonZero(d) ((d) != 0.0)
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Declare variables */
    int nQuery, nData;          /* number of elements */
    double *pQuery, *pData;     /* input arrays */
    double *positions;          /* output array(s) */

    int i;

    if (nrhs != 2) {
        mexErrMsgTxt("Two input arguments are required.");
    }
    if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }

    if (!(mxIsDouble(prhs[0])) || !(mxIsDouble(prhs[1]))) {
        mexErrMsgTxt("Input arrays must be of type double.");
    }

    nQuery      = mxGetNumberOfElements(prhs[0]);
    nData       = mxGetNumberOfElements(prhs[1]);
    pQuery      = (double *)mxGetPr(prhs[0]);
    pData       = (double *)mxGetPr(prhs[1]);
    
    plhs[0]     = mxCreateDoubleMatrix(1, nQuery, mxREAL);
    positions   = mxGetPr(plhs[0]);

    /* Now for the meat */

    for (i = 0; i < nQuery; i++) {
        positions[i] = 0;
        positions[i] = binarysearch(pQuery[i], pData, nData);
    }

    mxSetN(plhs[0], nQuery);
}

int binarysearch(double q, double *data, int n) {
    int l = 0;
    int u = n-1;
    int pivot;

    while (l < u) {
        pivot = l + (u - l) / 2;

        if (q > data[pivot]) {
            u = pivot - 1;
        } else {
            l = pivot + 1;
        }
    }

    /* Break ties to the right */
    if (l == n || q > data[l]) {
        return l;
    } else {
        return l + 1;
    }

}
