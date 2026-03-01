
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

#ifndef COMPUTEDOTPRODUCT_HPP
#define COMPUTEDOTPRODUCT_HPP

#ifndef HPCG_NO_LAIK
#include "laik/hpcg_laik.hpp"
#endif
#include "Vector.hpp"

#ifndef HPCG_NO_LAIK
int ComputeDotProduct_laik(const local_int_t n, const Laik_Blob *x, const Laik_Blob *y,
                      double &result, double &time_allreduce, bool &isOptimized);
#endif

int ComputeDotProduct(const local_int_t n, const Vector &x, const Vector &y,
                      double &result, double &time_allreduce, bool &isOptimized);

#endif // COMPUTEDOTPRODUCT_HPP
