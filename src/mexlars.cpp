//-*-c++-*-

/** Matlab MEX wrapper code
 *
 * LARS++, Copyright (C) 2007 Varun Ganapathi, David Vickery, James
 * Diebel, Stanford University
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301 USA.
*/

#include "lars_interface.h"
#include "mex.h"

using namespace std;

template <class real>
inline void denseLarsWrapper(int            nlhs,
			     mxArray       *plhs[],
			     int            nrhs,
			     const mxArray *prhs[]) {
  
  // Check for proper number of arguments:
  if (nrhs != 10 && nrhs != 14) {
    mexErrMsgTxt("MEXLARS: wrong number of input values.");
  }
  if (nlhs != 1 && nlhs != 4) {
    mexErrMsgTxt("MEXLARS: wrong number of output values.");
  }
    
  // Set default values of optional args
  LARS::METHOD method = LARS::LAR;
  LARS::STOP_TYPE stop_type = LARS::NONE;
  real stop_val = 1.0;
  bool return_whole_path = true;
  bool least_squares_beta = false;
  bool verbose = false;
  KERNEL kernel = AUTO;
  bool precomputed = false;
  bool return_sparse;
  real tmp;

  // get LARS method
  tmp = mxGetScalar(prhs[2]);
  if (tmp == 0) method = LARS::LASSO;
  if (tmp == 1) method = LARS::POSITIVE_LASSO;
  
  // get LARS stop condition
  tmp = mxGetScalar(prhs[3]);
  if (tmp == 0) stop_type = LARS::NORM;
  if (tmp == 1) stop_type = LARS::LAMBDA;
  if (tmp == 2) stop_type = LARS::NUM_ITER;
  if (tmp == 3) stop_type = LARS::NUM_BETA;

  // get LARS stop value
  tmp = mxGetScalar(prhs[4]);
  stop_val = (real)tmp;

  // get return_whole_path
  tmp = mxGetScalar(prhs[5]);
  return_whole_path = (bool)tmp;
  
  // get least_squares_beta
  tmp = mxGetScalar(prhs[6]);
  least_squares_beta = (bool)tmp;

  // get verbose
  tmp = mxGetScalar(prhs[7]);
  verbose = (bool)tmp;

  // get kernelized
  tmp = mxGetScalar(prhs[8]);
  if (tmp == 0) kernel = NO_KERN;
  if (tmp == 1) kernel = KERN;

  // get precomputed
  tmp = mxGetScalar(prhs[9]);
  precomputed = (bool)tmp;

  // Extract data pointers, key dimensions
  real* X_val;
  int* indx;
  int* jndx;
  int nz;
  real* X;
  real* Y = (real*)mxGetPr(prhs[1]);
  int N = mxGetM(prhs[0]);
  int p = mxGetN(prhs[0]);
  int r = mxGetN(prhs[1]);
  bool sparse = mxIsSparse(prhs[0]);

  // Run LARS
  vector< vector< pair<int,real> > > beta;
  int M;
  if (sparse) {
    X_val = (real*)mxGetPr(prhs[10]);
    indx = (int*)mxGetPr(prhs[11]);
    jndx = (int*)mxGetPr(prhs[12]);
    nz = (int)mxGetScalar(prhs[13]);

    if (r==1) {
      M = LARS::lars(&beta, X_val, indx, jndx, nz, Y, N, p, 
		     method, stop_type, stop_val,
		     return_whole_path, least_squares_beta, verbose);
    } else {
      M = LARS::lars(&beta, X, Y, N, p, r, method, stop_type, stop_val,
		     least_squares_beta, verbose);
    }  
  } else { // not sparse
    X = (real*)mxGetPr(prhs[0]);

    if (r==1) {
      M = LARS::lars(&beta, X, Y, N, p, method, stop_type, stop_val,
		     return_whole_path, least_squares_beta, verbose,
		     kernel, precomputed);
    } else {
      M = LARS::lars(&beta, X, Y, N, p, r, method, stop_type, stop_val,
		     least_squares_beta, verbose, kernel, precomputed);
    }
  }

  // Copy result into output
  if (nlhs == 1) {
    if (sizeof(real) == sizeof(float)) {
      plhs[0] = mxCreateNumericMatrix(p,M,mxSINGLE_CLASS,mxREAL);
    } else {
      plhs[0] = mxCreateNumericMatrix(p,M,mxDOUBLE_CLASS,mxREAL);
    }
    real* b = (real*) mxGetPr(plhs[0]);
    LARS::flattenSparseMatrix<real>(beta,b,p,M);
  } else if (nlhs == 4) { 
    // count the number of non-zero elements
    int ns = 0;
    for (int j=0;j<M;j++) ns += (int)beta[j].size();

    // allocate memory
    plhs[0] = mxCreateNumericMatrix(ns,1,mxDOUBLE_CLASS,mxREAL);
    plhs[1] = mxCreateNumericMatrix(ns,1,mxDOUBLE_CLASS,mxREAL);
    plhs[2] = mxCreateNumericMatrix(ns,1,mxDOUBLE_CLASS,mxREAL);
    plhs[3] = mxCreateNumericMatrix(1, 1,mxDOUBLE_CLASS,mxREAL);
    double* ia = (double*) mxGetPr(plhs[0]);
    double* ja = (double*) mxGetPr(plhs[1]);
    double* va = (double*) mxGetPr(plhs[2]);
    double* ra = (double*) mxGetPr(plhs[3]);

    // copy data from beta into output matrices
    *ra = M; for (int j=0,is=0;j<M;j++) {
      for (int i=0,n=(int)beta[j].size();i<n;i++,is++) {
	ia[is] = (double)(beta[j][i].first+1);
	ja[is] = (double)(j+1);
	va[is] = (double)beta[j][i].second;
      }
    } 
  } else {
    mexErrMsgTxt("MEXLARS: returns only one or four matrices.");
  }
  
  return;  
}

void mexFunction(int            nlhs,
                 mxArray       *plhs[],
                 int            nrhs,
                 const mxArray *prhs[]) {
  
  // Check for valid input

  // if wrong nlhs or nrhs, exit immediately
  if (nrhs < 2) {
    mexErrMsgTxt("MEXLARS: requires inputs X and Y.");
  } else if (nlhs > 1 && nlhs != 4) {
    mexErrMsgTxt("MEXLARS: returns only one or four matrices.");
  }

  // X must be numeric with 2 dimensions
  if (!mxIsNumeric(prhs[0]) ||
      mxGetNumberOfDimensions(prhs[0]) != 2) {
    mexErrMsgTxt("MEXLARS: X must be numeric 2d array.");
  }

  // Y must be numeric and full
  int y_dim = mxGetNumberOfDimensions(prhs[1]);
  if (!mxIsNumeric(prhs[1]) || 
      mxIsSparse(prhs[1]) || 
      (y_dim != 1 && y_dim != 2)) {
    mexErrMsgTxt("MEXLARS: Y must be full 1d/2d numeric array.");
  }

  // Y must have same number of rows as X
  if (mxGetM(prhs[0]) != mxGetM(prhs[1])) {
    mexErrMsgTxt("MEXLARS: incompatible matrix dimensions.");
  }

  // If X is sparse, we require 14 input parameters
  if (mxIsSparse(prhs[0])) {
    if (nrhs != 14) {
      mexErrMsgTxt("MEXLARS: Wrong number of input params for sparse X.");
    }
  } else { // otherwise we require 10 input parameters
    if (nrhs != 10) {
      mexErrMsgTxt("MEXLARS: Wrong number of input params for full X.");
    }
  }

  // check numeric types of X and Y and run LARS
  bool X_is_double = mxIsDouble(prhs[0]);
  bool X_is_single = mxIsSingle(prhs[0]);
  bool Y_is_double = mxIsDouble(prhs[1]);
  bool Y_is_single = mxIsSingle(prhs[1]);
  if (X_is_double && Y_is_double) {
    denseLarsWrapper<double>(nlhs,plhs,nrhs,prhs);
  } else if (X_is_single && Y_is_single) {
    denseLarsWrapper<float>(nlhs,plhs,nrhs,prhs);
  } else {
    mexErrMsgTxt("MEXLARS: X and Y must both be double or both be float.");
  }
}
