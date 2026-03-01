
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
#include <cstdio>
#include <cmath>

#include "ComputeDotProduct_ref.hpp"

int ComputeDotProduct_laik_ref(const local_int_t n, const Laik_Blob *x, const Laik_Blob *y,
                          double &result, double &time_allreduce)
{
  assert(x->localLength == n); // Test vector lengths
  assert(y->localLength == n);

  const char* dp_dbg = std::getenv("HPCG_LAIK_DP_DEBUG");
  static int dp_reported = 0;
  if (dp_dbg && dp_dbg[0] != '\0' && !dp_reported) {
    double *xv_dbg = 0;
    double *yv_dbg = 0;
    uint64_t xlen = 0;
    uint64_t ylen = 0;
    laik_get_map_1d(x->values, 0, (void **)&xv_dbg, &xlen);
    laik_get_map_1d(y->values, 0, (void **)&yv_dbg, &ylen);
    std::fprintf(stderr, "[rank %d] DP map0 lens x=%llu y=%llu n=%d\n",
                 laik_myid(world), (unsigned long long)xlen, (unsigned long long)ylen, (int)n);
  }

  double local_result = 0.0;

  double *xv;
  double *yv;
  laik_get_map_1d(x->values, 0, (void **)&xv, 0);
  laik_get_map_1d(y->values, 0, (void **)&yv, 0);

  double sumabs_x = 0.0;
  double sumabs_y = 0.0;
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

  if (dp_dbg && dp_dbg[0] != '\0' && !dp_reported) {
    for (local_int_t i = 0; i < n; i++) {
      sumabs_x += std::fabs(xv[i]);
      sumabs_y += std::fabs(yv[i]);
    }
    std::fprintf(stderr,
                 "[rank %d] DP local_result=%g sumabs_x=%g sumabs_y=%g\n",
                 laik_myid(world), local_result, sumabs_x, sumabs_y);
    dp_reported = 1;
  }

  // Collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  laik_allreduce(&local_result, &global_result, 1, laik_Double, LAIK_RO_Sum);
  result = global_result;
  if (dp_dbg && dp_dbg[0] != '\0') {
    if (!std::isfinite(result)) {
      std::fprintf(stderr, "[rank %d] DP result non-finite: %g\n", laik_myid(world), result);
    }
  }
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
