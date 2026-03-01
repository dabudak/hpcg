
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
 @file ComputeRestriction_ref.cpp

 HPCG routine
 */


#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#ifndef HPCG_NO_LAIK
#include "laik/hpcg_laik.hpp"
#endif
#include "ComputeRestriction_ref.hpp"

#if !defined(HPCG_NO_MPI) && !defined(HPCG_NO_LAIK)
int ComputeRestriction_laik_ref(const SparseMatrix &A, const Laik_Blob *rf)
{
  double *rfv;
  double *rcv;
  double *Axfv;

  laik_get_map_1d(rf->values, 0, (void **)&rfv, 0);
  laik_get_map_1d(A.mgData->rc_blob->values, 0, (void **)&rcv, 0);
  laik_get_map_1d(A.mgData->Axf_blob->values, 0, (void **)&Axfv, 0);

  local_int_t *f2c = A.mgData->f2cOperator;
  local_int_t nc = A.mgData->rc_blob->localLength;

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
  for (local_int_t i = 0; i < nc; ++i)
  {
    local_int_t j = f2c[i];
    rcv[i] = rfv[j] - Axfv[j];
  }

  return 0;
}
#else
int ComputeRestriction_ref(const SparseMatrix & A, const Vector & rf) {

  double * Axfv = A.mgData->Axf->values;
  double * rfv = rf.values;
  double * rcv = A.mgData->rc->values;
  local_int_t * f2c = A.mgData->f2cOperator;
  local_int_t nc = A.mgData->rc->localLength;

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
  for (local_int_t i=0; i<nc; ++i) rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];

  return 0;
}
#endif
