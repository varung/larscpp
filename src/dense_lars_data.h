//-*-c++-*-
#ifndef DENSE_LARS_DATA_H
#define DENSE_LARS_DATA_H

/** class DenseLarsData<float/double>
 * 
 * Main data class for Lars library.  Contains the matrix (X) and vector (y),
 * where we are finding regularized solutions to the equation X*beta = y.
 *
 * This class uses only flat C-style arrays internally, as this is
 * what is used by BLAS, which we use to do all the heavy lifting.
 *
 * This class is hidden behind the wrappers, which provide convenient
 * user interfaces to the library routines, so it is not likely that
 * any user will need to use this class.
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

extern "C" { 
#include "cblas.h"
}

typedef enum {AUTO=-1, NO_KERN, KERN} KERNEL;

template <class T>
class DenseLarsData {
public:
  typedef T real;
  typedef vector< pair<int,real> > SparseVector;

  // Constructor that accepts flat C column-major arrays
  DenseLarsData(const T* X, const T* y, const int N, const int p,
		const KERNEL kernel_, const bool precomputed_);
  
  // Destructor
  ~DenseLarsData();

  // dots two column vectors together
  T col_dot_product( int c1, int c2 ) const;

  // So that we can handle both kernelized and not
  void getXtY(vector<T>* xty ) const;
  void computeXtX();

  // Computes director correlation
  void compute_direction_correlation(const vector<pair<int,T> >& beta, 
				     const vector<T>& wval, 
				     T* a);

  // Computes current lambda given a sparse vector beta
  T compute_lambda(const vector< pair<int,T> >& beta) const;

  // Computes least squares solution based on given sparsity pattern
  T compute_lls_beta(SparseVector& beta) const;

  int nrows() const { return N; }
  int ncols() const { return p; }
  
private:
  // Main problem data
  const int N; // number of rows/samples
  const int p; // number of cols/features
  const T* X;  // data matrix is Nxp
  const T* y;  // response vector is Nx1

  // Internal work data
  T* Xw;    // X*w is Nx1
  T* Xty;   // X'*y is px1
  T* tmp_p; // temporary storage px1

  // Kernelized LARS support
  KERNEL kernel;    // use the kernelized form by pre-computing X'*X
  bool precomputed; // the given data matrix is actually X'*X, not X
  T** XtX_col;      // pointers to the columns of X'*X
  T* XtX_buf;       // temp space only used when not precomputed
};

// Constructor that accepts flat C column-major arrays
template <class T>
inline DenseLarsData<T>::DenseLarsData(const T* X_in, const T* y_in, 
				       const int N_in, const int p_in,
				       const KERNEL kernel_,
				       const bool precomputed_) :
  X(X_in), y(y_in), N(N_in), p(p_in), 
  kernel(kernel_), precomputed(precomputed_)
{

  // auto-select whether to use kernel or not
  if (kernel == AUTO) {
    if (N>=int(0.5*p)) kernel = KERN;
    else kernel = NO_KERN;
  }

  // allocate memory for internal work space
  tmp_p = new T[p];
  
  // - if X'*X and X'*y are precomputed, this implies we are using a kernel
  // - otherwise, we'll need some temp space to store X'*y
  if (precomputed == true) kernel = KERN;
  else Xty = new T[p];

  // - If we are using a kernelized form, we need XtX_col and possibly
  //   XtX_buf (if X'*X was not precomputed)
  // - otherwise we need space for X*w in computing X'*(X*w)
  if (kernel == KERN) {
    XtX_col = new T*[p];
    if (precomputed) {
      for (int i=0;i<p;i++) XtX_col[i] = (T*)&X[i*p];
      XtX_buf = (T*)X;
    } else {
      XtX_buf = new T[p*p];
      computeXtX();
      for (int i=0;i<p;i++) XtX_col[i] = &XtX_buf[i*p];;
    }
  } else {
    Xw = new T[N];
  }
}

// Destructor
template <class T>
inline DenseLarsData<T>::~DenseLarsData() {
  delete [] tmp_p;
  if (!precomputed) delete [] Xty;
  if (kernel == KERN) {
    delete [] XtX_col;
    if (!precomputed) delete [] XtX_buf;
  } else {
    delete [] Xw;
  }
}

///////////////////////////
// Double Precision Case //
///////////////////////////

// dots two column vectors together
template <>
inline double DenseLarsData<double>::col_dot_product(const int c1, 
						     const int c2) const {
  if (kernel == KERN) {
    return XtX_buf[c1+p*c2];
  } else {
    return cblas_ddot(N, &X[N*c1], 1, &X[N*c2], 1);
  }
}

// So that we can handle both kernelized and not
template <>
inline void DenseLarsData<double>::getXtY(vector<double>* xty) const {  
  xty->resize(p);
  if (precomputed) {
    for(int i=0;i<p;++i) (*xty)[i] = y[i];
  } else {
    cblas_dgemv(CblasColMajor, CblasTrans, N, p, 1.0, X, N, y, 1, 0.0, Xty, 1);
    for(int i=0;i<p;++i) (*xty)[i] = Xty[i];
  }
}

// Computes internal copy of X'*X if required
template <>
inline void DenseLarsData<double>::computeXtX() {
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
	      p, p, N, 1.0, X, N, X, N, 0.0, XtX_buf, p);
}

// Computes director correlation a = X'*X*w
template <>
inline void DenseLarsData<double>::
compute_direction_correlation(const vector< pair<int,double> >& beta, 
			      const vector<double>& wval, 
			      double* a ) { 
  // clear old data
  memset(a,0,p*sizeof(double));
  
  // compute X'*X*beta, in one of two ways:
  if (kernel == KERN) {
    for(int i=0,n=beta.size();i<n;++i) {
      // add (X'*X)_i * w_i
      cblas_daxpy(p, wval[i], XtX_col[beta[i].first], 1, a, 1);
    }
  } else {
    // add columns into X*w
    memset(Xw,0,N*sizeof(double));
    for(int i=0,n=beta.size();i<n;++i)
      cblas_daxpy(N, wval[i], &X[beta[i].first*N], 1, Xw, 1);
    // now do X'*(X*w)
    cblas_dgemv(CblasColMajor,CblasTrans,N,p,1.0,X,N,Xw,1,0.0,a,1);
  }
}

// Computes current lambda given a sparse vector beta
template <>
inline double DenseLarsData<double>::
compute_lambda(const vector< pair<int,double> >& beta) const {
  // clear old data
  memset(tmp_p,0,p*sizeof(double));
  
  // compute max(abs(2*X'*(X*beta - y))) in one of two ways:
  if (kernel == KERN) {
    // X'*y - (X'*X)*beta
    memcpy(tmp_p,Xty,p*sizeof(double));
    for(int i=0,n=beta.size();i<n;++i) {
      // subtract (X'*X)_i * beta_i
      cblas_daxpy(p, -beta[i].second, XtX_col[beta[i].first], 1, tmp_p, 1);
    }
    return 2.0*fabs(tmp_p[cblas_idamax(p, tmp_p, 1)]);
  } else {
    // compute Xw = y - X*beta
    memcpy(Xw,y,N*sizeof(double));
    for (int i=0,n=beta.size();i<n;++i)
      cblas_daxpy(N, -beta[i].second, &X[N*beta[i].first], 1, Xw, 1);
    // now compute 2*X'*Xw = 2*X'*(y - X*beta)
    cblas_dgemv(CblasColMajor,CblasTrans,N,p,2.0,X,N,Xw,1,0.0,tmp_p,1);
    return fabs(tmp_p[cblas_idamax(p, tmp_p, 1)]);
  }
}

///////////////////////////
// Single Precision Case //
///////////////////////////

// dots two column vectors together
template <>
inline float DenseLarsData<float>::col_dot_product(const int c1, 
						     const int c2) const {
  if (kernel == KERN) {
    return XtX_buf[c1+p*c2];
  } else {
    return cblas_sdot(N, &X[N*c1], 1, &X[N*c2], 1);
  }
}

// So that we can handle both kernelized and not
template <>
inline void DenseLarsData<float>::getXtY(vector<float>* xty) const {  
  xty->resize(p);
  if (precomputed) {
    for(int i=0;i<p;++i) (*xty)[i] = y[i];
  } else {
    cblas_sgemv(CblasColMajor, CblasTrans, N, p, 1.0, X, N, y, 1, 0.0, Xty, 1);
    for(int i=0;i<p;++i) (*xty)[i] = Xty[i];
  }
}

// Computes internal copy of X'*X if required
template <>
inline void DenseLarsData<float>::computeXtX() {
  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
	      p, p, N, 1.0, X, N, X, N, 0.0, XtX_buf, p);
}

// Computes director correlation a = X'*Xw
template <>
inline void DenseLarsData<float>::
compute_direction_correlation(const vector< pair<int,float> >& beta, 
			      const vector<float>& wval, 
			      float* a ) { 
  // clear old data
  memset(a,0,p*sizeof(float));
  
  // compute X'*X*beta, in one of two ways:
  if (kernel == KERN) {
    for(int i=0,n=beta.size();i<n;++i) {
      // add (X'*X)_i * w_i
      cblas_saxpy(p, wval[i], XtX_col[beta[i].first], 1, a, 1);
    }
  } else {
    // add columns into X*w
    memset(Xw,0,N*sizeof(float));
    for(int i=0,n=beta.size();i<n;++i)
      cblas_saxpy(N, wval[i], &X[beta[i].first*N], 1, Xw, 1);
    // now do X'*(X*w)
    cblas_sgemv(CblasColMajor,CblasTrans,N,p,1.0,X,N,Xw,1,0.0,a,1);
  }
}

// Computes current lambda given a sparse vector beta
template <>
inline float DenseLarsData<float>::
compute_lambda(const vector< pair<int,float> >& beta) const {
  // clear old data
  memset(tmp_p,0,p*sizeof(float));
  
  // compute max(abs(2*X'*(X*beta - y))) in one of two ways:
  if (kernel == KERN) {
    // X'*y - (X'*X)*beta
    memcpy(tmp_p,Xty,p*sizeof(float));
    for(int i=0,n=beta.size();i<n;++i) {
      // subtract (X'*X)_i * beta_i
      cblas_saxpy(p, -beta[i].second, XtX_col[beta[i].first], 1, tmp_p, 1);
    }
    return 2.0*fabs(tmp_p[cblas_isamax(p, tmp_p, 1)]);
  } else {
    // compute Xw = y - X*beta
    memcpy(Xw,y,N*sizeof(float));
    for (int i=0,n=beta.size();i<n;++i)
      cblas_saxpy(N, -beta[i].second, &X[N*beta[i].first], 1, Xw, 1);
    // now compute 2*X'*Xw = 2*X'*(y - X*beta)
    cblas_sgemv(CblasColMajor,CblasTrans,N,p,2.0,X,N,Xw,1,0.0,tmp_p,1);
    return fabs(tmp_p[cblas_isamax(p, tmp_p, 1)]);
  }
}

#endif
