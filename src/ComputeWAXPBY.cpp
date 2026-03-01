
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeWAXPBY.cpp

 HPCG routine
 */

#ifndef HPCG_NO_LAIK
#include "laik/hpcg_laik.hpp"
#endif
#include "ComputeWAXPBY.hpp"
#include "ComputeWAXPBY_ref.hpp"

#ifndef HPCG_NO_LAIK
int ComputeWAXPBY_laik(const local_int_t n, const double alpha, const Laik_Blob * x,
                  const double beta, const Laik_Blob * y, Laik_Blob * w, bool &isOptimized)
{

  // This line and the next two lines should be removed and your version of ComputeWAXPBY should be used.
  isOptimized = false;
  return ComputeWAXPBY_laik_ref(n, alpha, x, beta, y, w);
}
#endif

int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized) {

  // This line and the next two lines should be removed and your version of ComputeWAXPBY should be used.
  isOptimized = false;
  return ComputeWAXPBY_ref(n, alpha, x, beta, y, w);
}
