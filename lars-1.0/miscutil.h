//-*-c++-*-
#ifndef MISCUTIL_H
#define MISCUTIL_H

/** Set of basic templated utility functions used in the test code. 
 * These functions, which accept at least <float/double> Included:
 *   - Seed the random-number generator with part of the current time.
 *   - Generate a uniform random number over range [a,b]
 *   - Generate a normally-distributed random number with given mean/var.
 *   - Prepare a random data matrix and response vector, possibly normalized.
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

#include <cstdlib>
#include <time.h>
#include <math.h>
#include <iostream>
#ifdef WIN32
  #include <sys/timeb.h>
  #include <Winsock2.h>
#else 
  #include <sys/time.h>
#endif

#define M_PI       3.14159265358979323846

using namespace std;

#ifdef WIN32  
  /// Seed psudo-random number generator
  inline void seedRand() {
    struct _timeb timebuffer;
    _ftime( &timebuffer );
    unsigned int n = int(timebuffer.time*1000 + timebuffer.dstflag); 
    std::srand(n);
  }
#else
  /// Seed psudo-random number generator
  inline void seedRand() {
    timeval tv; gettimeofday(&tv,NULL);
    unsigned int n = int(tv.tv_sec*1000000 + tv.tv_usec); std::srand(n);
  }
#endif
  
/// Returns a sample from a normal distribution
template <class T> 
inline T normalRand(T mean = T(0), T stdev = T(1)) {
  const double norm = 1.0/(RAND_MAX + 1.0);
  double u = 1.0 - std::rand()*norm;
  double v = rand()*norm;
  double z = sqrt(-2.0*log(u))*cos(2.0*M_PI*v);
  return T(mean + stdev*z);
}

/// Generate random problem data (X and y) of size (Nxp), optionally 
/// normalized so that each column is zero mean and unit variance.
template <class T> 
inline void prepareData(const int N, const int p, const int r, 
			const bool norm,
			T*& X, T*& y) {
  X = new T[N*p];
  y = new T[N*r];
  for (int j=0,k=0;j<p;j++) {
    T s = T(0);
    T s2 = T(0);
    for (int i=0;i<N;i++,k++) {
      T v = normalRand<T>();
      X[k] = v;
      s += v;
      s2 += v*v;
    }
    if (norm) {
      T std = sqrt(s2 - s*s/T(N));
      k -= N;
      for (int i=0;i<N;i++,k++) {
	X[k] = (X[k] - s/T(N))/std;
      }
    }
  }
  
  for (int i=0;i<N*r;i++) {
    y[i] = normalRand<T>();
  }
}


class Timer {
public:
  // Constructor/Destructor
  Timer();

  // Methods
  void start(); // starts watch
  double stop(); // stops watch, returning time
  void reset(); // reset watch to zero
  double print(char* label = NULL); // prints time on watch without stopping
  double stopAndPrint(char* label = NULL); // stops and prints time

private:
  // Data
  bool timing;
  long double start_time;
  long double stop_time;
  long double total_time;
};

inline Timer::Timer() {
  reset();
}

inline void Timer::start() {
  if (!timing) {
    start_time = (long double)(clock())/(long double)(CLOCKS_PER_SEC);
    timing = true;
  }
}

inline double Timer::stop() {
  if (timing) {
    stop_time = (long double)(clock())/(long double)(CLOCKS_PER_SEC);
    timing = false;
    total_time += (stop_time - start_time);
  }
  return (double)total_time;
}

inline void Timer::reset() {
  total_time = start_time = stop_time = 0.0;
  timing = false;
}

inline double Timer::print(char* label) {
  bool was_timing = timing;
  double current_time = stop();
  if (label) cout << label;
  cout.precision(4);
  cout << current_time << " seconds" << endl << flush;
  if (was_timing) start();
  return current_time;
}

inline double Timer::stopAndPrint(char* label) {
  double current_time = stop();
  if (label) cout << label;
  cout.precision(4);
  cout << current_time << " seconds" << endl << flush;
  return current_time;
}

#endif
