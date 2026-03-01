
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
 @file ComputeProlongation_ref.cpp

 HPCG routine
 */

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "laik/hpcg_laik.hpp"
#include "ComputeProlongation_ref.hpp"

#ifndef HPCG_NO_MPI
int ComputeProlongation_laik_ref(const SparseMatrix & Af, Laik_Blob * xf) {

  double * xfv;
  double *xcv;

  laik_get_map_1d(xf->values, 0, (void **)&xfv, 0);
  laik_get_map_1d(Af.mgData->xc_blob->values, 0, (void **)&xcv, 0);

  local_int_t * f2c = Af.mgData->f2cOperator;
  local_int_t nc = Af.mgData->rc_blob->localLength;

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
  for (local_int_t i=0; i<nc; ++i)
    xfv[f2c[i]] += xcv[i]; // This loop is safe to vectorize

  return 0;
}

#else
int ComputeProlongation_ref(const SparseMatrix &Af, Vector &xf)
{

  double *xfv = xf.values;
  double *xcv = Af.mgData->xc->values;
  local_int_t *f2c = Af.mgData->f2cOperator;
  local_int_t nc = Af.mgData->rc->localLength;

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
  for (local_int_t i = 0; i < nc; ++i)
    xfv[f2c[i]] += xcv[i]; // This loop is safe to vectorize

  return 0;
}
#endif
