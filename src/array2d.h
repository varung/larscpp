//-*-c++-*-
#ifndef ARRAY2D_H
#define ARRAY2D_H

/* class array2d<float/double/etc.> 
 *
 * Templated two-dimensional array container class, based on the stl
 * valarray<> type.  Uses row-major ordering, which means the linear
 * storage of the data contains the entire first row, followed by the 
 * entire second row, etc.  This is notably in contrast to the BLAS
 * and LAPACK standards, which use column-wise ordering.  As such, this
 * class is not used in portions of the code in which BLAS or LAPACK
 * are used.
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

#include <algorithm>
#include <numeric>
#include <valarray>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cassert>

#define BOUNDS_CHECKING
const int IOSTREAM_PRECISION = 12; 

using namespace std;

template<typename T>
static void valarray_write(ostream& os, const valarray<T>&v){
  os << setprecision(IOSTREAM_PRECISION);
  os << v.size() << endl;	
  for(size_t i=0; i<v.size(); i++)
    os << v[i] << endl;
}

template<typename T>
static void valarray_read(istream& is, valarray<T>* v){
  size_t numelms = 0;
  is >> numelms;
  v->resize(numelms);
  for(size_t i=0; i<numelms; i++ )
    is >> (*v)[i];
  assert(!is.fail());
}

template<typename T>
static void print(ostream& out, valarray<T>& b){
  if(!b.size())
    return;
  out << "[";
  copy(&b[0],&b[b.size()-1],ostream_iterator<T>(out," "));
  out << b[b.size()-1] << "]" << endl;
}

template<typename T>
static void apply(valarray<T> *elms, T(*fun)(T)){
  for(size_t i=0;i<elms->size();i++){
    (*elms)[i] = fun((*elms)[i]);
  }
}

// Two dimensional array class using valarray as backing store
template<typename T>
class array2d {
public:
  typedef T value_type; 
  array2d(): r_(0), c_(0) {}
  array2d(const array2d& other) : 
    r_(other.r_), c_(other.c_), elms_(other.elms_) {}
      
  // forms array2d that is [x;y] (vertically joined x and y)
  array2d(const array2d& x, const array2d &y) {
    assert(x.ncols() == y.ncols() );
    r_ = (x.nrows()+y.nrows());
    c_ = (x.ncols());
    elms_.resize(r_*c_);
    elms_[slice(0,x.elms_.size(),1)] = x.elms_; // copy x in
    elms_[slice(x.elms_.size(),y.elms_.size(),1)] = y.elms_; // copy y in
  }
  // construct with all values set to constant
  array2d(size_t r,size_t c, T def=0):r_(r),c_(c),elms_(def,r*c){ elms_ = def;}
  
  // reshape an array of values(copies vals in) (row-major)
  array2d(size_t r,size_t c, T* vals):r_(r),c_(c), elms_(vals,r*c){}

  // creates 2d array from filename
  array2d(string path){ load(path);}

  array2d(vector<vector<T> > bob ) {
    resize( bob.size(), bob[0].size() );
    for(int i=0; i< bob.size(); i++) {
      for(int j=0; j<bob[i].size(); j++) {
	operator()(i,j)=bob[i][j];
      }
    }
  }

  T* operator[]( int r ) {
    return &(elms_[r*ncols()]);
  }

  const T* operator[]( int r ) const {
    return &(elms_[r*ncols()]);
  }

  // fortran_major = row major
  void toVecVec( vector<vector<T> >& vecvec, bool fortran_major=false ){
    if( ! fortran_major ){
      vecvec.resize( nrows() );
      for(int i=0; i<nrows(); ++i) {
	vecvec[i].resize(ncols());
	for(int j=0; j<ncols(); ++j)
	  vecvec[i][j]=operator()(i,j);
      }
    }
    assert(0);
  }

  inline T& operator()(size_t r, size_t c){ 
#ifdef BOUNDS_CHECKING
    assert(r<nrows()&&c<ncols());
    assert(r*ncols()+c < elms_.size());
#endif
    return elms_[r*ncols()+c];
  }

  inline T operator()(size_t r, size_t c) const { 
#ifdef BOUNDS_CHECKING
    assert(r<nrows()&&c<ncols());
    assert((unsigned)(r*ncols()+c) < elms_.size());
#endif
    return elms_[r*ncols()+c];
  }

  /** prints out the matrix to any output stream
   * this is not the best format for serialization.
   * use << and >> for serialization.
   *
   * @param out  output stream
   */
  void print(std::ostream& out) const{
    for(size_t r=0;r<nrows();r++) {
      for(size_t c=0;c<ncols();c++) {
	out.width(14);
	out.precision(5);
	out << operator()(r,c) << " ";
      }
      out << std::endl;
    }
  }

  // Note: Slice(start,length,stride)
  std::slice_array<T> row(size_t r){
#ifdef BOUNDS_CHECKING
    assert(r<nrows());
#endif
    return elms_[slice(r*c_,c_,1)];
  }

  std::valarray<T> row(size_t r) const {
#ifdef BOUNDS_CHECKING
    assert(r<nrows());
#endif
    return elms_[slice(r*c_,c_,1)];
  }

  std::slice_array<T> col(size_t c){
#ifdef BOUNDS_CHECKING
    assert(c<ncols());
#endif
    return elms_[slice(c,r_,c_)];
  }

  std::valarray<T> col(size_t c) const{
#ifdef BOUNDS_CHECKING
    assert(c<ncols());
#endif
    return elms_[slice(c,r_,c_)];
  }

  void resize(size_t r, size_t c) {
    elms_.resize(r*c,(T)(0.0)); elms_ = (T)(0); r_=r; c_=c;
  }

  void resize(size_t r, size_t c, const T& val) {
    elms_.resize(r*c,val); elms_ = val; r_=r; c_=c;
  }

  void reset(const T& val) {
    elms_ = val;
  }

  void copy(const array2d<T>& other) {
    resize(other.nrows(), other.ncols());
    assert(elms_.size() == other.elms_.size());
    elms_ = other.elms_;
  }

  // load commands
  // these assume a whitespace delimited file for the matrix
  // where the number of lines = number of rows e.g.,
  // 1 2 3 
  // 4 5 6
  // This is different from the serialization format of the << and >> operators
  // which require the first two items to be the number of rows and cols
  void load(ifstream& in){
    vector<T> vals; // TODO: change to T?
    bool expectCol = true;
    char c;
    T val;
    unsigned int m = 0,n =0;
    // Determine number of columns
    while (in.get(c)) {
      if (c == '\r' || c == '\n') break;
      if (c == '\t' || c == ' ' || c == ',') { expectCol = true;
      } else if (expectCol) { n++; expectCol = false; }
    }
    in.seekg(0, ios_base::beg);

    // Read values from file
    while (in >> val)
      vals.push_back(val);
		
    m = vals.size() / n;
    // Integrity check
    if (m * n == 0 || m * n != vals.size()) {
      cerr << "Matrix.loadFromPath: data format invalid " 
	   << m << "," << n << endl;
      exit(-1);
    }
    resize(m,n);
    for(size_t i=0; i<vals.size();i++)
      elms_[i]=(T)(vals[i]);
  }

  void load(const string path){
    ifstream in(path.c_str(), ios::in);
    if(in.fail()){
      cerr << "cannot open:" << path << endl;
      exit(-1);
    }
    load(in);
    in.close();
  }
	
  // Serialization
  // Format: (whitespace delimited)
  // numrows numcols
  // item1 item2 . . . . itemN where N = numrows * numcols
  friend ostream& operator<<(ostream& os, const array2d<T>& ia) {
    os << setprecision(IOSTREAM_PRECISION);
    os << ia.nrows() << " " << ia.ncols() << endl;
    ia.print(os);
    return os;
  }

  friend istream& operator>>(istream& is, array2d<T>& ia) {
    size_t r = -1, c = -1;
    is >> r >> c; // read in dimensions
    ia.resize(r,c); // init ia
    vector<T> items;
    for( size_t i = 0; i < r; i++)
      for( size_t j =0; j < c; j++)
	is >> ia(i,j);
    return is;
  }
	
  inline size_t nrows() const {return r_;}
  inline size_t ncols() const {return c_;}

  T rowsum(size_t r){
    T temp = 0;
    for(size_t i=0; i< c_; i++)
      temp+= operator()(r,i);
    return temp;
  }

  void rowscale(const size_t r, const T& val){
    for(size_t i =0; i < ncols(); i++)
      operator()(r,i) *= val;
  }

  void rowapply(const size_t r, T (*fun)(T)){
    for(size_t i =0; i < ncols(); i++){
      T& item = operator()(r,i);
      item = fun(item);
    }
  }

  void apply(T (*fun)(T)){
    ::apply(&elms_, fun);
  }

  // makes rows sum to 1
  void normalizerow(size_t r){
    T sum = rowsum(r);
    T scale = 1.0/sum;
    rowscale(r,scale);
  }

  void normalizerows(){
    for(size_t i=0;i<r_;i++) normalizerow(i);
  }

  void transpose(){
    array2d<T> dup( c_, r_ );
    for(int i=0; i<dup.nrows(); ++i)
      for(int j=0; j<dup.ncols(); ++j)
	dup(i,j) = operator()(j,i);
    swap(r_, c_ );
    elms_ = dup.elms_;
  }
    
  size_t r_;
  size_t c_;
public:
  std::valarray<T> elms_;
};

typedef array2d<double> dblarray2d;
typedef array2d<int> intarray2d;

#endif
