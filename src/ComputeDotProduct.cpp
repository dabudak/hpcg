
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
 @file ComputeDotProduct.cpp

 HPCG routine
 */
#include "laik/hpcg_laik.hpp"
#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"

int ComputeDotProduct_laik(const local_int_t n, const Laik_Blob *x, const Laik_Blob *y,
                      double &result, double &time_allreduce, bool &isOptimized)
{

  // This line and the next two lines should be removed and your version of ComputeDotProduct should be used.
  isOptimized = false;
  return ComputeDotProduct_laik_ref(n, x, y, result, time_allreduce);
}

int ComputeDotProduct(const local_int_t n, const Vector &x, const Vector &y,
                      double &result, double &time_allreduce, bool &isOptimized)
{

  // This line and the next two lines should be removed and your version of ComputeDotProduct should be used.
  isOptimized = false;
  return ComputeDotProduct_ref(n, x, y, result, time_allreduce);
}
