//-*-c++-*-
#ifndef __DENSE_CHOLESKY_H
#define __DENSE_CHOLESKY_H

/** class DenseCholesky<float/double>
 * 
 * Represents state of cholesky factorization of symmetric positive definite X
 * - Allows you to add row/col to X and incrementally update cholesky
 * - Allows you to remove row/col from X and incrementally update cholesky
 * - Allows you to use the cholesky factorization to
 * - find beta for X beta = y 
 *
 * Uses general utility routines for cholesky decompositions found in
 * cholesky.h
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

#include <iostream>
#include <cstdio>
#include "array2d.h"
#include "cholesky.h"

using namespace std;

template<typename T>
class DenseCholesky {
public:
  /// Constructor accepts the maximum possible size of the cholesky
  DenseCholesky( int max_size ) : A(max_size, max_size), used(0) {}
  
  /// Add a new row/col to the internal matrix (the number of values expected
  /// is equal to the number of existing rows/cols + 1)
  void addRowCol( const T* vals ) {
    // grow the array2d
    int j=used;
    resize(used+1);
    for(int i=0; i<=j; ++i)
      A(j,i) = vals[i];
    update_cholesky(A, j);
  }

  /// Remove a row/column from X updates cholesky automatically
  void removeRowCol( int r ) {
    downdate_cholesky(A,used,r);
    used--;
  }

  /// Solves for beta given y 
  void solve( const vector<T> y, vector<T>* beta ) {
    backsolve( A, *beta, y, used );   
  }

  /// Print the cholesky
  void print( ostream& out ) {
    char buff[255];
    out << "Used:" << used << endl;
    for(int i=0; i< used; ++i){
      for(int j=0; j<used; ++j){
        sprintf(buff, "%16.7le", A(i,j));
        out << buff;
      }
      out << endl;
    }
  }

private:
  void resize(int howManyRowCol ) {
    if(A.nrows() < howManyRowCol || A.ncols() < howManyRowCol ) {
      assert(0);
    }
    used=howManyRowCol;
  }
  
  static double dot( const int N, const double* a, const double* b ) {
    return cblas_ddot( N, a, 1, b, 1);
  }
  static float dot( const int N, const float* a, const float* b ) {
    return cblas_sdot( N, a, 1, b, 1);
  }

public:
  array2d<T> A;
  int used;
};

#endif
