LARS++
======

Lars++ is a C++ and Matlab software library for solving L1-regularized
least-squares, both exactly and approximately.  It is based on the
paper:

Efron, B., Johnstone, I., Hastie, T. and Tibshirani, R. (2002). Least
angle regression ps file; pdf file To appear, Annals of Statistics
2003

All code contained in this package, excepting the reference BLAS and
CBLAS implementations is copyrighted.  See the LICENSE.txt file for
details on use.  Generally speaking, it is freely available, without
restriction, for corporate and non-corporate use.  It has absolutely
no warranty.

Copyright (c) 2006 Varun Ganapathi, David Vickery, James Diebel,
Stanford University


## LICENSE

See the file LICENSE.txt for the full license.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301 USA.



## STATEMENT OF PROBLEM

Variables:
  - int N is the number of samples
  - int p is the number of variables
  - int r is the number of right-hand-sides
  - T* X is (N x p) data matrix (N samples, p variables)
  - T* Y is (N x r) response matrix in case of multiple right-hand-sides

  - T lambda is the scalar L1-norm penalty weight
  - T t is the L1-norm constraint
  
There are two equivalent ways to formulate the problem:
  
  (1) L1-regularized formulation:
  
minimize norm(X*beta-y,2) + lambda*norm(beta,1)
  
  (2) L1-constrained formulation:
  
  minimize norm(X*beta-y,2)
  subject to norm(beta,1) < t
  
In both cases, we use norm(~,n) to be the Ln-norm of ~.
  
There is a one-to-one mapping between lambda and t, though we never
actually solve for this mapping.
  
LARS and LASSO both provide many solutions to this problem, with
various values of lambda/t, correponding to various numbers of
non-zero elements in beta.
  
We provide several interfaces to this problem, described below.

## INSTALLATION

The user may either use the included BLAS/CBLAS or link to
user-supplied versions (for details, please see the next section of
this README).

If the user wishes to use the included reference implementations, they
must compile them.  To do this (starting from the unpacked source
directory):

cd blas
make
cd ../cblas
make
cd ..

This will create two libraries (liblarsblas.a and liblarscblas.a) in
the main source directory.  These are the default choice in the
Makefile, so no further changes are required to make the library.  The
user may make the library by typing:

make

in the main source directory.  If different BLAS/CBLAS is desired,
please edit the Makefile to reflect that.  The variable BLASLIBS at
the top of the file must be changed to include the path to the
alternative libraries.

This will make a testing routine called testlars, and a Matlab MEX
routine called mexlars.mexglx.  The latter may be copied to wherever
it is needed.  The accompanying file, larspp.m, provides a convenient
public interface to the low-level routine.

If you do not have Matlab, the second part will not work.  This will
not effect the compilation of the C++ part.  If you don't have Matlab
and would like to avoid an error message during the make process, you
may type

make testlars

This is only a superficial change from typing 'make' since the latter
will just exit once it can't find the Matlab compiler script.



## USER-PROVIDED BLAS

We have designed this to be as fast as possible. Computationally
intensive operations are performed with calls to BLAS through the
CBLAS interface.  We include the reference implementation of the
required BLAS routines in this code, but STRONGLY encourage the user
to link to a faster BLAS.  The fastest BLAS that we are aware of is
called GotoBLAS, and may be found at:

http://www.tacc.utexas.edu/resources/software/

Another fast BLAS is called ATLAS.  This may be found at:

http://math-atlas.sourceforge.net/



## MATLAB INTERFACE

The public interface in Matlab is provided in larspp.m.  In Matlab,
type 

help larspp 

for details on how to use this.



## C++ INTERFACE

The C++ interface is contained in lars_interface.h.  There are two C++
functions to run LARS.  The first accepts a single right-hand-side and
the second accepts multiple right-hand-sides.  Details on how to use
these are contained in the comments in lars_interface.h.

All functions are templated to accept either float or double.

An important data type that we use is a Sparse Vector.  This is an STL
vector of pairs of integers and floating point values.  Each pair
represents a non-zero element of the sparse vector.  This is:

vector< pair<int,T> > beta

where T is either float or double.  The ith index is beta[i].first,
and the corresponding value is beta[i].second.  We also use arrays of
these sparse vectors (i.e., a sparse matrix), which has the signature:

vector< vector< pair<int,T> > > beta

These may be flattened into C-style arrays/matrices (always
column-major ordering) using some utility functions that we include in
the lars_interface.h header.

Also in lars_interface.h are several other functions that may be of
use for C++ users.  In all, the functions in this file are:

FOR SINGLE RIGHT-HAND-SIDES:
  template<typename T>
  inline int lars(vector< vector< pair<int,T> > >* beta,
		  const T* X,
		  const T* y,
		  const int N,
		  const int p,
		  const METHOD method = LAR,
		  const STOP_TYPE stop_type = -1,
		  const T stop_val = T(0),
		  const bool return_whole_path = true,
		  const bool least_squares_beta = false,
		  const bool verbose = false);

FOR MULTIPLE RIGHT-HAND-SIDES:
  template<typename T>
  inline int lars(vector< vector< pair<int,T> > >* beta,
		  const T* X,
		  const T* Y,
		  const int N,
		  const int p,
		  const int r,
		  const METHOD method = LAR,
		  const STOP_TYPE stop_type = -1,
		  const T stop_val = T(0),
		  const bool least_squares_beta = false,
		  const bool verbose = false);

COMPUTE THE L1 NORM OF BETA:
  template<typename T>
  inline T l1NormSparseVector(const vector< pair<int,T> >& beta);

FLATTEN A SINGLE SPARSE VECTOR INTO A C-ARRAY:
  template<typename T>
  inline void flattenSparseVector(const vector< pair<int,T> >& beta, 
				  T* beta_dense, const int p);

FLATTEN A VECTOR OF SPARSE VECTORS INTO A C-ARRAY:
  template<typename T>
  inline void flattenSparseMatrix(const vector< vector< pair<int,T> > >& beta, 
				  T* beta_dense, const int p, const int M);

PRINT A VECTOR OF SPARSE VECTORS AS A MATRIX:
  template<typename T>
  inline void printMatrix(T* a, const int N, const int p, 
			  string label, std::ostream& out);

INTERPOLATE BETWEEN TWO SPARSE VECTORS:
  template<typename T>
  inline void interpolateSparseVector(vector< pair<int,T> >* beta_interp,
				      const int p,
				      const vector< pair<int,T> >& beta_old,
				      const vector< pair<int,T> >& beta_new,
				      const T f);
