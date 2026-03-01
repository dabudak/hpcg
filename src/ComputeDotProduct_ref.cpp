
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
 @file ComputeDotProduct_ref.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include "mpi.h"
#include "laik/hpcg_laik.hpp"
#include "mytimer.hpp"
#endif
#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>

#include "ComputeDotProduct_ref.hpp"

int ComputeDotProduct_laik_ref(const local_int_t n, const Laik_Blob *x, const Laik_Blob *y,
                          double &result, double &time_allreduce)
{
  assert(x->localLength == n); // Test vector lengths
  assert(y->localLength == n);

  double local_result = 0.0;

  double *xv;
  double *yv;
  laik_get_map_1d(x->values, 0, (void **)&xv, 0);
  laik_get_map_1d(y->values, 0, (void **)&yv, 0);

  if (yv == xv)
  {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+ : local_result)
#endif
    for (local_int_t i = 0; i < n; i++)
      local_result += xv[i] * xv[i];
  }
  else
  {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+ : local_result)
#endif
    for (local_int_t i = 0; i < n; i++)
      local_result += xv[i] * yv[i];
  }

  // Collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  laik_allreduce(&local_result, &global_result, 1, laik_Double, LAIK_RO_Sum);
  result = global_result;
  time_allreduce += mytimer() - t0;

  return 0;
}

int ComputeDotProduct_ref(const local_int_t n, const Vector &x, const Vector &y,
                          double &result, double &time_allreduce)
{
  assert(x.localLength >= n); // Test vector lengths
  assert(y.localLength >= n);

  double local_result = 0.0;
  double *xv = x.values;
  double *yv = y.values;
  if (yv == xv)
  {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+ : local_result)
#endif
    for (local_int_t i = 0; i < n; i++)
      local_result += xv[i] * xv[i];
  }
  else
  {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+ : local_result)
#endif
    for (local_int_t i = 0; i < n; i++)
      local_result += xv[i] * yv[i];
  }

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  result = global_result;
  time_allreduce += mytimer() - t0;
#else
  time_allreduce += 0.0;
  result = local_result;
#endif

  return 0;
}
