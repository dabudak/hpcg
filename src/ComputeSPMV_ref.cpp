
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
 @file ComputeSPMV_ref.cpp

 HPCG routine
 */

 
#include "ComputeSPMV_ref.hpp"

#ifndef HPCG_NO_MPI
#include <iostream>
#include <cstdlib>

#include "ExchangeHalo.hpp"
#include "laik/hpcg_laik.hpp"
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>

int ComputeSPMV_laik_ref(const SparseMatrix &A, Laik_Blob *x, Laik_Blob *y)
{
  assert(x->localLength == A.localNumberOfRows); // Test vector lengths
  assert(y->localLength == A.localNumberOfRows);

  laik_switchto_partitioning(x->values, A.ext, LAIK_DF_Preserve, LAIK_RO_None);

  double * xv;
  double * yv;
  laik_get_map_1d(x->values, 0, (void **)&xv, 0);
  laik_get_map_1d(y->values, 0, (void **)&yv, 0);
  const local_int_t nrow = A.localNumberOfRows;

#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif

  for (local_int_t i=0; i< nrow; i++)  {
    double sum = 0.0;
    const double * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];
    for (int j=0; j< cur_nnz; j++)
    {
      sum += cur_vals[j] * xv[cur_inds[j]];
    }
    yv[i] = sum;
  }


  laik_switchto_partitioning(x->values, A.local, LAIK_DF_None, LAIK_RO_None);

  return 0;
}

int ComputeSPMV_ref(const SparseMatrix &A, Vector &x, Vector &y)
{

  assert(x.localLength >= A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength >= A.localNumberOfRows);

#ifndef HPCG_NO_MPI
  ExchangeHalo(A, x);
#endif
  const double *const xv = x.values;
  double *const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
  for (local_int_t i = 0; i < nrow; i++)
  {
    double sum = 0.0;
    const double *const cur_vals = A.matrixValues[i];
    const local_int_t *const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    for (int j = 0; j < cur_nnz; j++)
      sum += cur_vals[j] * xv[cur_inds[j]];
    yv[i] = sum;
  }
  return 0;
}
