//-*-c++-*-
#ifndef LARS__H
#define LARS__H

/** class Lars<DenseLarsData/GeneralLarsData>
 * 
 * Represent the current state of the lars algorithm and can iterate
 * the algorithm.  This does not contain any high-level controls, so
 * the user is encouraged to use the interface functions contained in
 * lars_interface.h.
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
 *
 *
 * TODO: Use linked list for active/inactive sets instead of a flat
 *       vector
 */

#include <cstdio>
#include <iostream>
#include <fstream>
#include <iterator>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <valarray>

//#include "vec.h"
#include "dense_cholesky.h"

using namespace std;

namespace LARS {
  typedef enum {LAR=-1, LASSO, POSITIVE_LASSO} METHOD;

  template< typename T >
  class Lars {
   public:
    typedef typename T::real real;

    /** get the current parameters */
    const vector<pair<int,real> >& getParameters();


    template<typename U>
    void print( const U& v ) {
      for(int i=0; i<v.size(); ++i)
        fprintf(fid, "%12.5f", v[i]);
      fprintf(fid, "\n");
    }

    /** get least squares parameters for active set */
    const void getParameters(vector<pair<int,real> >* p,
			     const vector<pair<int,real> >& b);

    /** Constructor accepts a LarsDenseData object and a method. */
    Lars( T& data, METHOD m ):
      data_(data), 
      m_(m), 
      small_(numeric_limits<real>::epsilon()),
      chol_( min(data.nrows(),data.ncols())),
      recentDeactivation_(false)
    { 
      initialize(); 
    }

    ~Lars() {
      fprintf(fid, "DONE\n");
      fclose(fid);
    }
    /** Perform a single interation of the LARS loop. */
    bool iterate(){
      if(vars >= nvars ) return false;
      // if( beta_.size() >= data_.ncols() ) return false;
      k++;
      fprintf(fid, "K: %12d\n", k );
      fprintf(fid, "%12d %12d\n", vars, nvars);

      // [C j] = max(abs(c(I)));
      // j = I(j);
      real C; int j;
      GetMaxAbsCorrelation(j, C);
      fprintf(fid, "[C,j] = [%12.5f, %12d]\n", C, j+1 );
      if(!lassocond) {
        fprintf(fid, "activating %d\n", j+1 );
        activate(j);
      }
      // computes w, AA, and X'w
      real AA = findSearchDirection();
      fprintf(fid, "W:");
      print(w_);

      fprintf(fid, "AA: %12.5f\n", AA);
      real gamma;
      if( vars == nvars ) {
        gamma = C/AA;
        fprintf(fid, "gamma: %12.5f\n", gamma);
      } else {
        //fprintf(fid, "a:");
        //print(a_);
        gamma = C/AA;
        int min_index = -1;
        // temp = [(C - c(I))./(AA - a(I)); (C + c(I))./(AA + a(I))];
        // gamma = min([temp(temp > 0); C/AA]);
        for(int j=0; j<a_.size(); ++j) {
          // only consider inactive features
          if(isActive(j)) continue; 
          real t1 = (C - c_[j])/(AA - a_[j]);
          real t2 = (C + c_[j])/(AA + a_[j]);
          // consider only positive items
          if( t1 > 0 && t1 < gamma ) { 
            gamma = t1; min_index = j;
          }
          if( t2 > 0 && t2 < gamma ) { 
            gamma = t2; min_index = j;
          }
        }
        fprintf(fid, "min_index: %12d\n", min_index+1);
        fprintf(fid, "gamma: %12.5f\n", gamma);
      }

      lassocond = false;
      // temp = -beta(k,A)./w';
      // [gamma_tilde] = min([temp(temp > 0) gamma]);
      // j = find(temp == gamma_tilde);
      // if gamma_tilde < gamma,
      //     gamma = gamma_tilde;
      //     lassocond = 1;
      // end
      if( m_ == LASSO ) {
        // fprintf(fid, "LASSO\n");
        // find out minimum amount so parameter hits 0
        real gamma_tilde = gamma;
        // fprintf(fid, "beta_size: %12d\n", beta_.size());
        for(int i=0; i<beta_.size(); ++i){
          // fprintf(fid, "i: %12d\n", i);
          real temp = -beta_[i].second / w_[i];
          if( temp > 0 && temp < gamma_tilde ) {
            gamma_tilde = temp;
            j = beta_[i].first; // record the parameter
          }
        }
        fprintf(fid, "gamma_tilde: %12.5f\n", gamma_tilde);
        if( gamma_tilde < gamma ) {
          gamma = gamma_tilde;
          lassocond = true;
          fprintf(fid, "j: %12d\n", j+1 );
        }
      }
      fprintf(fid, "gamma: %12.5f\n", gamma);
      // add lambda * w to beta
      for(int i=0; i<beta_.size(); ++i)
        beta_[i].second += gamma * w_[i];

      // update correlation with a
      for(int i=0; i<c_.size(); ++i)
        c_[i] -= gamma * a_[i];
      
      // print the beta
      fprintf(fid, "beta: ");
      for(int i=0; i<beta_.size(); ++i) {
        fprintf(fid, "%12.5f", beta_[i].second );
      }
      fprintf(fid, "\n");

      if(lassocond) {
        deactivate(j);
      }


      return true;
    }

   private:
    /** initialize the state of LARS */
    void initialize();

    /** add to active set, returns if anything changed */
    bool updateActiveSet();

    /** returns true if parameter i is active */
    bool isActive( int i );

    /** activate parameter i and updates cholesky */
    bool activate( int i );

    /** deactivates parameter i and downdates cholesky */
    void deactivate( int i );


    /**
     * Function: sign
     * --------------
     *  returns sign of input (1,0,-1)
     */
    inline real sign( real temp ) {
      if( temp > 0 ) return 1.0;
      if( temp < 0 ) return  -1.0;
      return 0;
    }

    /** 
     * Function: findSearchDirection
     * -----------------------------
     *
     * Solves for the step in parameter space given the current
     *  active parameters.
     *
     *  GA1 = R\(R'\s);
     *  AA = 1/sqrt(dot(GA1,s));
     *  w = AA*GA1;
     *  returns AA
     **/

    real findSearchDirection() {
      w_.resize(beta_.size());
      assert(w_.size()==beta_.size());
      //fprintf(fid, "w_.size() = %d\n", w_.size() );
      // set w_ = sign(c_[A])
      for(int i=0; i<w_.size(); ++i){
        w_[i] = sign(c_[beta_[i].first]);
      }
      //fprintf(fid, "sign(c_[A]):");
      //print(w_);
      // w_ = R\(R'\s)
      chol_.solve(w_, &w_ );
      //fprintf(fid, "w_:");
      //print(w_);

      // AA = 1/sqrt(dot(GA1,s));
      real AA(0.0);
      for(int i=0; i<w_.size(); ++i) AA += w_[i]*sign(c_[beta_[i].first]);
      AA = real(1.0)/sqrt(AA);
      //fprintf(fid, "AA: %12.5f\n", AA);
      for(int i=0; i<w_.size(); ++i) w_[i] *= AA;


      // calculate the a (uses beta to get active indices )
      // a_ = X'Xw
      data_.compute_direction_correlation( beta_, w_, &(a_[0]) );

      return AA;
    }
  

    // member variables
    T& data_; // data(contains X and y)
    METHOD m_;            // chooses type of LARS algorithm
    vector<pair<int,real> > beta_;   // current parameters(solution) [Not Sorted]

    // incrementally updated quantities
    valarray<int> active_; // active[i] = position in beta of active param or -1
    vector<real> c_; // correlation of columns of X with current residual
    vector<real> w_;          // step direction ( w_.size() == # active )
    valarray<real> a_;   // correlation of columns of X with current step dir
    real small_;                 // what is small?
    DenseCholesky<real> chol_;   // keeps track of cholesky
    // temporaries
    valarray<real> temp_;      // temporary storage for active correlations
    bool recentDeactivation_;  // don't add next step after a deactivation


    /** New Variable to exactly replicate matlab lars */
    vector<real> Xty; // correlation of columns of X with current residual
    bool lassocond;
    bool stopcond;
    int vars;
    int nvars;
    int k;
    FILE* fid;
  
    /** 
     * Function: GetMaxCorrelation
     * ---------------------------
     * [C j] = max(abs(c(I)));
     * j = I(j);
     */
    void GetMaxAbsCorrelation( int& index, real& val ) {
      val = real(0);
      for(int i=0; i<c_.size(); ++i){
        if( isActive(i) ) continue;
        if( fabs(c_[i]) > val ) {
          index = i;
          val = fabs(c_[i]);
        }
      }
    }
  };

  /** Initialization routine. */
  template<typename T>
  void Lars<T>::initialize(){
    // initially all parameters are 0
    // so current residual is y
    
    data_.getXtY( &Xty );
    nvars = min<int>(data_.nrows()-1,data_.ncols());
    lassocond = 0;
    stopcond = 0;
    k = 0;
    vars = 0;
    fid = fopen("vlarspp_debug.txt","w");
   // fid = stderr;

    data_.getXtY( &c_ );
    // step dir = 0 so a_ = 0
    a_.resize(c_.size());
    active_.resize(data_.ncols(),-1);
    temp_.resize(data_.ncols());
  }

  /** Activate all equally corelated x. */
  template<typename T>
  bool Lars<T>::updateActiveSet() {
    if(recentDeactivation_) {
      recentDeactivation_ = false;
      return true;
    }
    bool changed = false;
    // find highest correlation
    real C = 0;
    for(int i=0; i<c_.size(); ++i){
      if( isActive(i) ) continue;
      C = max(fabs(c_[i]), C);
    }

    // activate them if not already active
    for(int i=0; i<c_.size(); ++i) {
      if( !isActive(i) && fabs(fabs(c_[i])-C) < small_ )  {
        if (!activate( i )) return false;
        changed = true;
      }
    }
    return changed;
  }

  /** Returns bool of whether that row is active. */
  template<typename T>
  inline bool Lars<T>::isActive(int i) {
    return active_[i] != -1;
  }

  /** 
   * Function: activate
   * ------------------
   * Update state so that feature i is active with weight 0
   *  if it is not already active.
   * 
   * R = cholinsert(R,X(:,j),X(:,A));
   * A = [A j];
   * I(I == j) = [];
   * vars = vars + 1;
   *
   **/
  template<typename T>
  bool Lars<T>::activate(int i) { 
    //fprintf(fid, "activate(%d)\n", i );
    if(isActive(i) || beta_.size() >= data_.nrows()) {
      fprintf(fid, "blash\n");
      return false;
    }
    active_[i] = beta_.size();
    beta_.push_back(make_pair(i,0.0));
    w_.resize(beta_.size());
    //fprintf(fid, "beta.size(): %d\n", beta_.size());
    // dot i with all the other active columns f => xtx(i,j)
    for(int f=0; f<beta_.size(); ++f){
      temp_[f] = data_.col_dot_product(i, beta_[f].first );
    }
    chol_.addRowCol( &temp_[0] );
    vars++;
    // fprintf(fid, "vars %d\n", vars );
    return true;
  }


  /** 
   * Function: deactivate
   * --------------------
   * Update state so that feature i is no longer active if it is
   * already active.
   *
   *  R = choldelete(R,j);
   *  I = [I A(j)];
   *  A(j) = [];
   *  vars = vars - 1;
   **/
  template<typename T>
  void Lars<T>::deactivate(int i) { 
    //cout << "deactivate(" << i <<")"<< endl;
    assert(!isActive(i));
    // if(!isActive(i)) return;
    int beta_index = active_[i];
    beta_.erase( beta_.begin() + beta_index ); // check this!!
    active_[i] = -1;
    chol_.removeRowCol( beta_index );
    // fix the active set!
    for(int r=0; r<active_.size(); ++r)
      active_[r] = -1;
    for(int r=0; r<beta_.size(); ++r)
      active_[beta_[r].first]=r;
    vars--;
  }


  /** Take the step! */
  /*
  template<typename T>
  void Lars<T>::takeStep() {
    // based on the type, figure out how far we need to go
    // then take the step
    real lambda = 1;
    if(m_ == LAR || m_ == LASSO ) {
      real A = a_[beta_[0].first];
      real C = c_[beta_[0].first];
      for(int j=0; j<a_.size(); ++j) {
        // only consider inactive features
        if(isActive(j)) continue; 

        real t1 = (C - c_[j])/(A - a_[j]);
        real t2 = (C + c_[j])/(A + a_[j]);

        // consider only positive items
        if( t1 > 0 ) lambda = min( lambda, t1 );
        if( t2 > 0 ) lambda = min( lambda, t2 );
      }
    }
    
    int beta_0_index = -1;
    if( m_ == LASSO ) {
      // find out minimum amount so parameter hits 0
      real lambda_0 = numeric_limits<real>::infinity();
      for(int i=0; i<beta_.size(); ++i){
        real temp = -beta_[i].second / w_[i];
        if( temp > 0 && temp < lambda_0 ) {
          lambda_0 = temp;
          beta_0_index = i;
        }
      }

      if( lambda_0 < lambda ) lambda = lambda_0;
      else                    beta_0_index = -1;
    }

    // add lambda * w to beta
    for(int i=0; i<beta_.size(); ++i)
      beta_[i].second += lambda * w_[i];

    // update correlation with a
    for(int i=0; i<c_.size(); ++i)
      c_[i] -= lambda * a_[i];

    if( m_ == LASSO && beta_0_index != -1 ) {
      deactivate( beta_[beta_0_index].first );
    }
  }*/



  /** Return a reference to the current active set of beta parameters. */
  template<typename T>
  const vector<pair<int,typename T::real> >& Lars<T>::getParameters() {
    return beta_;
  }

  /** Return the Least-squares solution to X*beta = y for the subset
   * of currently active beta parameters */
  template<typename T>
  const void Lars<T>::
  getParameters(vector<pair<int,typename T::real> >* p, 
		const vector<pair<int,typename T::real> >& b) {

    vector<real> temp(c_.size());
    vector<real> temp2(w_.size());
    
    p->resize(b.size());
    data_.getXtY( &temp );
    for(int i=0; i<b.size(); ++i){
      temp2[i] = temp[b[i].first];
    }
    chol_.solve(temp2, &temp2 );
    for(int i=0; i<b.size(); ++i){
      (*p)[i].first=b[i].first;
      (*p)[i].second=temp2[i];
    }
  }
};
#endif
