//-*-c++-*-
#ifndef SPARSE_LARS_DATA_H
#define SPARSE_LARS_DATA_H

/** class SparseLarsData<float/double>
 * 
 * Main data class for Lars library.  Contains the matrix (X) and vector (y),
 * where we are finding regularized solutions to the equation X*beta = y.
 *
 * This class uses the Sparse BLAS standard interface, and is
 * currently linking to the Sparse BLAS reference implementation
 * provided by NIST.
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

#include "blas_sparse.h"

template <class T>
class SparseLarsData {
public:
  typedef T real;
  typedef vector< pair<int,real> > SparseVector;

  // Constructor that accepts sparse 
  SparseLarsData(T* X_val, int* indx, int* jndx,
		 const int nz, const T* y_in, 
		 const int N_in, const int p_in);

  // Destructor
  ~SparseLarsData();

  // dots two column vectors together
  T col_dot_product( int c1, int c2 ) const;

  // So that we can handle both kernelized and not
  void getXtY(vector<T>* xty ) const;

  // Computes director correlation
  void compute_direction_correlation(const vector<pair<int,T> >& beta, 
				     const vector<T>& wval, 
				     T* a) const;

  // Computes current lambda given a sparse vector beta
  T compute_lambda(const vector< pair<int,T> >& beta) const;

  // Computes least squares solution based on given sparsity pattern
  T compute_lls_beta(SparseVector& beta) const;

  int nrows() const { return N; }
  int ncols() const { return p; }
  
private:
  // Main problem data
  const int N;                // number of rows/samples
  const int p;                // number of cols/features
  const T* y;                 // response vector is Nx1

  blas_sparse_matrix X; // data matrix is Nxp
  T** xj_val;           // array of cols of data matrix
  int** xj_idx;         // array of cols of indices
  int* xj_nz;           // array of numbers of elements

  // Internal work arrays
  T* Xw;    // X*w is Nx1
  T* Xty;   // X'*y is px1
  T* tmp_N; // temporary storage Nx1
  T* tmp_p; // temporary storage px1
};

// Destructor
template <class T>
inline SparseLarsData<T>::~SparseLarsData() {
  BLAS_usds(X);
  delete [] xj_val;
  delete [] xj_idx;
  delete [] xj_nz;
  delete [] Xty;
  delete [] Xw;
  delete [] tmp_N;
  delete [] tmp_p;
}

///////////////////////////
// Double Precision Case //
///////////////////////////

// Constructor that accepts flat C-array for y and a list of index
// pairs and values for X
template <>
inline SparseLarsData<double>::
SparseLarsData(double* X_val, int* indx, int* jndx,
	       const int nz, const double* y_in, 
	       const int N_in, const int p_in) :
  y(y_in), N(N_in), p(p_in)
{
  // Allocate internal memory
  xj_val = new double*[p];
  xj_idx = new int*[p];
  xj_nz  = new int[p];
  Xty   = new double[p];
  Xw    = new double[N];
  tmp_N = new double[N];
  tmp_p = new double[p];

  // Build sparse matrix X from passed index-pairs and values
  X = BLAS_duscr_begin(N,p);
  BLAS_ussp(X,blas_zero_base);
  if (BLAS_duscr_insert_entries(X,nz,X_val,indx,jndx) != 0)
    cout << "Warning: can't add to sparse matrix X." << endl << flush;
  if (BLAS_duscr_end(X) != 0)
    cout << "Warning: can't allocate sparse matrix X." << endl << flush;

  // Build individual columns
  for (int j=0,ks=0,ke=0;j<p;j++) {
    // find ending point of data in this column
    ks=ke; while (ke<nz && jndx[ke]==j) ke++;

    // add pointers to column data
    xj_val[j] = (double*)&X_val[ks];
    xj_idx[j] = (int*)&indx[ks];
    xj_nz[j]  = ke-ks;
  }
}

// dots two column vectors together
template <>
inline double SparseLarsData<double>::col_dot_product(const int c1, 
						      const int c2) const {
  
  // clear elements in tmp_N corresponding to non-zero elements in c1
  for (int k=0,n=xj_nz[c1];k<n;k++) tmp_N[xj_idx[c1][k]] = 0.0;
  
  // scatter elements from c2 into tmp_N
  BLAS_dussc(xj_nz[c2], xj_val[c2], tmp_N, 1, xj_idx[c2], blas_zero_base);
  
  // compute the dot product between 
  double tmp = 0.0;
  if (xj_nz[c1]) 
    BLAS_dusdot(blas_no_conj, xj_nz[c1], xj_val[c1], xj_idx[c1], 
		tmp_N, 1, &tmp, blas_zero_base);
  return tmp;
}

// So that we can handle both kernelized and not
template <>
inline void SparseLarsData<double>::getXtY(vector<double>* xty) const {  
  xty->resize(p);
  memset(Xty,0,p*sizeof(double));
  BLAS_dusmv(blas_trans, 1.0, X, y, 1, Xty, 1);

  // put it in xty
  for(int i=0; i<p; ++i) (*xty)[i] = Xty[i];
}

// Computes director correlation a = X'*Xw
template <>
inline void SparseLarsData<double>::
compute_direction_correlation(const vector< pair<int,double> >& beta, 
			      const vector<double>& wval, 
			      double* a ) const { 
  // clear old data
  memset(Xw,0,N*sizeof(double));
  memset(a,0,p*sizeof(double));
  
  // add columns into Xw
  for(int i=0,n=beta.size();i<n;++i) {
    int j = beta[i].first;
    for (int k=0,nk=xj_nz[j];k<nk;k++) 
      Xw[xj_idx[j][k]] += xj_val[j][k]*wval[i];
  }
  
  // now do X'*(X*w)
  BLAS_dusmv(blas_trans, 1.0, X, Xw, 1, a, 1);
}

// Computes current lambda given a sparse vector beta
template <>
inline double SparseLarsData<double>::
compute_lambda(const vector< pair<int,double> >& beta) const {
  // clear old data
  memset(tmp_p,0,p*sizeof(double));
  memcpy(Xw,y,N*sizeof(double));
  
  // compute Xw = y - X*beta
  for (int i=0,n=beta.size();i<n;++i) {
    int j = beta[i].first;
    for (int k=0,nk=xj_nz[j];k<nk;k++) 
      Xw[xj_idx[j][k]] += -beta[i].second*xj_val[j][k];
  }
  
  // now compute -2*X'*Xw = 2*X'*(X*beta - y)
  BLAS_dusmv(blas_trans, -2.0, X, Xw, 1, tmp_p, 1);
  
  return fabs(tmp_p[cblas_idamax(p, tmp_p, 1)]);
}

///////////////////////////
// Single Precision Case //
///////////////////////////

// Constructor that accepts flat C-array for y and a list of index
// pairs and values for X
template <>
inline SparseLarsData<float>::
SparseLarsData(float* X_val, int* indx, int* jndx,
	       const int nz, const float* y_in, 
	       const int N_in, const int p_in) :
  y(y_in), N(N_in), p(p_in)
{
  // Build sparse matrix X from passed index-pairs and values
  X = BLAS_suscr_begin(N,p);
  BLAS_ussp(X,blas_zero_base);
  if (BLAS_suscr_insert_entries(X,nz,X_val,indx,jndx) != 0)
    cerr << "Warning: can't add to sparse matrix X." << endl << flush;
  if (BLAS_suscr_end(X) != 0)
    cerr << "Warning: can't allocate sparse matrix X." << endl << flush;

  // Build individual columns
  xj_val = new float*[p];
  xj_idx = new int*[p];
  xj_nz  = new int[p];
  for (int j=0,ks=0,ke=0;j<p;j++) {
    // find ending point of data in this column
    ks=ke; while (ke<p && jndx[ke]==j) ke++;

    // add pointers to column data
    xj_val[j] = &X_val[ks];
    xj_idx[j] = &indx[ks];
    xj_nz[j]  = ke-ks;
  }
  
  // allocate memory for internal work arrays
  Xty   = new float[p];
  Xw    = new float[N];
  tmp_N = new float[N];
  tmp_p = new float[p];
}

// dots two column vectors together
template <>
inline float SparseLarsData<float>::col_dot_product(const int c1, 
						      const int c2) const {
  // clear elements in tmp_N corresponding to non-zero elements in c1
  for (int k=0,n=xj_nz[c1];k<n;k++) tmp_N[xj_idx[c1][k]] = 0.0;
  
  // scatter elements from c2 into tmp_N
  BLAS_sussc(xj_nz[c2], xj_val[c2], tmp_N, 1, xj_idx[c2], blas_zero_base);
  
  // compute the dot product between 
  float tmp;
  BLAS_susdot(blas_no_conj, xj_nz[c1], xj_val[c1], xj_idx[c1], 
	      tmp_N, 1, &tmp, blas_zero_base);
  
  return tmp;
}

// So that we can handle both kernelized and not
template <>
inline void SparseLarsData<float>::getXtY(vector<float>* xty) const {  
  xty->resize(p);
  BLAS_susmv(blas_trans, 1.0, X, y, 1, Xty, 1);
  
  // put it in xty
  for(int i=0; i<p; ++i) (*xty)[i] = Xty[i];
}

// Computes director correlation a = X'*Xw
template <>
inline void SparseLarsData<float>::
compute_direction_correlation(const vector< pair<int,float> >& beta, 
			      const vector<float>& wval, 
			      float* a ) const { 
  // clear old data
  memset(Xw,0,N*sizeof(float));
  memset(a,0,p*sizeof(float));
  
  // add columns into Xw
  for(int i=0,n=beta.size();i<n;++i) {
    int j = beta[i].first;
    BLAS_susaxpy(xj_nz[j],wval[i],xj_val[j],xj_idx[j],Xw,1,blas_zero_base);
  }
  
  // now do X'*(X*w)
  BLAS_susmv(blas_trans, 1.0, X, Xw, 1, a, 1);
}

// Computes current lambda given a sparse vector beta
template <>
inline float SparseLarsData<float>::
compute_lambda(const vector< pair<int,float> >& beta) const {
  // clear old data
  memset(tmp_p,0,p*sizeof(float));
  memcpy(Xw,y,N*sizeof(float));
  
  // compute Xw = y - X*beta
  for (int i=0,n=beta.size();i<n;++i) {
    int j = beta[i].first;
    BLAS_susaxpy(xj_nz[j],-beta[i].second,xj_val[j],xj_idx[j],
		 Xw,1,blas_zero_base);
  }
  
  // now compute -2*X'*Xw = 2*X'*(X*beta - y)
  BLAS_susmv(blas_trans, -2.0, X, Xw, 1, tmp_p, 1);
  
  return fabs(tmp_p[cblas_isamax(p, tmp_p, 1)]);
}

/*template <>
inline double SparseLarsData<double>::col_dot_product(const int c1, 
						      const int c2) const {
  // clear elements in tmp_N corresponding to non-zero elements in c1
  for (int k=0,n=xj_nz[c1],k<n;k++) tmp_N[xj_idx[c1][k]] = 0.0;
  //memset(tmp_N,0,N*sizeof(double));

  // scatter elements from c2 into tmp_N
  //for (int k=0,n=xj_nz[c2],k<n;k++) tmp_N[xj_idx[c2][k]] = xj_val[c2][k];

  // compute the dot product between 
  //tmp = 0.0;
  //for (int k=0,n=xj_nz[c1],k<n;k++) tmp+=tmp_N[xj_idx[c1][k]]*xj_val[c1][k];
  
  return tmp;
  }*/

  /*
  // clear elements in tmp_N corresponding to non-zero elements in c1
  for (int k=0,n=xj_nz[c1];k<n;k++) tmp_N[xj_idx[c1][k]] = 0.0;
  //memset(tmp_N,0,N*sizeof(double));

  // scatter elements from c2 into tmp_N
  for (int k=0,n=xj_nz[c2];k<n;k++) tmp_N[xj_idx[c2][k]] = xj_val[c2][k];

  // compute the dot product between 
  double tmp = 0.0;
  for (int k=0,n=xj_nz[c1];k<n;k++) tmp+=tmp_N[xj_idx[c1][k]]*xj_val[c1][k];
  
  return tmp;
  */

  /*for (int j=0;j<p;j++) {
    double tmp;
    BLAS_dusdot(blas_no_conj, xj_nz[j], xj_val[j], xj_idx[j], 
		y, 1, &tmp, blas_zero_base);
    //cout << tmp << endl;
    Xty[j] = tmp;
    }*/

  /*cout << endl << endl << "Xt*y = " << endl;
  for(int i=0; i<p; ++i) cout << Xty[i] << endl;
  cout << endl;*/


#endif
