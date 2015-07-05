//-*-c++-*-

/** Test code to evaluate a random LARS problem.
 *  
 *  - User supplies N, p as command-line args (in none given, defaults used).
 *  - User supplies optional number of right-hand-sides.
 *  - Generates random problem data and runs Lars on it, reporting results.
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
#include "miscutil.h"
#include "lars_interface.h"

using namespace std;

///////////////////////////////////////////////////////////////////////////
////////////////////////// User-Editable Portion //////////////////////////
///////////////////////////////////////////////////////////////////////////

// We support both double and float.  Change this typedef to select which.
typedef double real;

// Define default problem size.
const int p_default = 64;
const int N_default = 64;

// For the case of solving for multiple right-hand-sides
const int r = 1;

// To have accurate times, we allow several trials for each size.
const int num_rhs_default = 1;

// Define which type of problem to test (0 for LARS, 1 for LASSO).
const LARS::METHOD method = LARS::LASSO;
const LARS::STOP_TYPE stop_type = LARS::NORM;
const real stop_val = real(1);
const bool return_whole_path = true;
const bool least_squares_beta = false;
const bool verbose = true;
const bool use_multiple_right_hand_sides = false;
const KERNEL kernel = AUTO;
const bool precomputed = false;

///////////////////////////////////////////////////////////////////////////
//////////////////////// End User-Editable Portion /////////////////////////
///////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]){
  // Give message about calling conventions
  if (argc <= 1) {
    cout << endl 
	 << "Usage: " << endl
	 << "testlars [N] [p] [# RHS]" << endl
	 << " - Data matrix is (N x p) " << endl
	 << " - # RHS is the number of right-hand side vectors to consider." 
	 << endl << endl;
  }

  // Data for dense lars problem
  real* X;
  real* Y;
  real* norm_beta;
  vector< vector< pair<int,real> > > beta;
  
  // Grab command line args
  int p = p_default;
  int N = N_default;
  int num_rhs = num_rhs_default;

  if (argc > 1) N = atoi(argv[1]);
  if (argc > 2) p = atoi(argv[2]);
  if (argc > 3) num_rhs = atoi(argv[3]);

  // Generate random data (function prepareData(...) is in miscutil.h
  //seedRand();
  srand(1);
  prepareData(N, p, r, false, X, Y);

  cout << "Testing LARS Library on random data: "
       << "X is " << N << " x " << p << ", " 
       << "Y is " << N << " x " << r << "..." << endl << flush;

  // Run LARS
  int M;
  Timer t;
  t.start();

  for (int i=0;i<num_rhs;i++) {
    beta.clear();
    if (use_multiple_right_hand_sides) {
      M = LARS::lars(&beta, X, Y, N, p, r, method, stop_type, stop_val,
		     least_squares_beta, verbose, kernel, precomputed);
    } else {
      M = LARS::lars(&beta, X, Y, N, p, method, stop_type, stop_val,
		     return_whole_path, least_squares_beta, verbose,
		     kernel, precomputed);
    }
  }

  if (M <= 0) {
    cerr << "Bad return value. Exiting early. " << endl << flush;
    exit(1);
  }
  cout << "Done." << endl << endl;
  cout << "M = " << M << endl;
  t.stopAndPrint("Total time: " );

  // Print if small enough
  if (max(p,N) < 20) {
    // Make array of beta norms
    norm_beta = new real[beta.size()];
    for (int i=0;i<beta.size();i++) {
      norm_beta[i] = LARS::l1NormSparseVector(beta[i]);
    }
    
    // Make beta dense for printing
    real* beta_dense = new real[M*p];
    LARS::flattenSparseMatrix(beta,beta_dense,p,M);
    
    // print results
    cout << endl;
    cout.precision(8); 
    LARS::printMatrix(X,N,p,"X",cout);
    LARS::printMatrix(Y,N,r,"Y",cout); 
    LARS::printMatrix(beta_dense,p,M,"b",cout);
    LARS::printMatrix(norm_beta,1,beta.size(),"norm_beta",cout);

    // clean up
    delete [] beta_dense;
    delete [] norm_beta;
  }

  // clean up
  delete [] X;
  delete [] Y;
}
