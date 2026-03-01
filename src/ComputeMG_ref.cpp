
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
 @file ComputeSYMGS_ref.cpp

 HPCG routine
 */

#include <cassert>
#include <iostream>
#include <cmath>

#include "laik/hpcg_laik.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeSYMGS_ref.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeRestriction_ref.hpp"
#include "ComputeProlongation_ref.hpp"

#ifndef HPCG_NO_LAIK
int ComputeMG_laik_ref(const SparseMatrix &A, const Laik_Blob * r, Laik_Blob * x)
{
  assert(x->localLength == A.localNumberOfRows);
  assert(x->localLength == r->localLength);

  ZeroLaikVector(x); // initialize x to zero

  const char* nan_check = std::getenv("HPCG_LAIK_NAN_CHECK");

  int ierr = 0;
  if (A.mgData != 0)
  { // Go to next coarse level if defined
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i = 0; i < numberOfPresmootherSteps; ++i) ierr += ComputeSYMGS_laik_ref(A, r, x);

    if (nan_check && nan_check[0] != '\0') {
      double* xv = 0;
      laik_get_map_1d(x->values, 0, (void**)&xv, 0);
      for (local_int_t i = 0; i < A.localNumberOfRows; ++i) {
        if (!std::isfinite(xv[i])) {
          std::fprintf(stderr, "[rank %d] MG presmooth non-finite at %d: %g\n",
                       A.geom ? A.geom->rank : -1, (int)i, xv[i]);
          break;
        }
      }
    }

    if (ierr != 0)
      return ierr;
    ierr = ComputeSPMV_laik_ref(A, x, A.mgData->Axf_blob);
    if (ierr != 0)
      return ierr;

    // Perform restriction operation using simple injection
    ierr = ComputeRestriction_laik_ref(A, r);
    if (ierr != 0)
      return ierr;

    ierr = ComputeMG_laik_ref(*A.Ac, A.mgData->rc_blob, A.mgData->xc_blob);
    if (ierr != 0)
      return ierr;
    ierr = ComputeProlongation_laik_ref(A, x);
    if (ierr != 0)
      return ierr;
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i = 0; i < numberOfPostsmootherSteps; ++i)
      ierr += ComputeSYMGS_laik_ref(A, r, x);
    if (ierr != 0)
      return ierr;

    if (nan_check && nan_check[0] != '\0') {
      double* xv = 0;
      laik_get_map_1d(x->values, 0, (void**)&xv, 0);
      for (local_int_t i = 0; i < A.localNumberOfRows; ++i) {
        if (!std::isfinite(xv[i])) {
          std::fprintf(stderr, "[rank %d] MG postsmooth non-finite at %d: %g\n",
                       A.geom ? A.geom->rank : -1, (int)i, xv[i]);
          break;
        }
      }
    }
  }
  else
  {
    ierr = ComputeSYMGS_laik_ref(A, r, x);
    if (ierr != 0)
      return ierr;

    if (nan_check && nan_check[0] != '\0') {
      double* xv = 0;
      laik_get_map_1d(x->values, 0, (void**)&xv, 0);
      for (local_int_t i = 0; i < A.localNumberOfRows; ++i) {
        if (!std::isfinite(xv[i])) {
          std::fprintf(stderr, "[rank %d] MG leaf non-finite at %d: %g\n",
                       A.geom ? A.geom->rank : -1, (int)i, xv[i]);
          break;
        }
      }
    }
  }
  return 0;
}
#else
int ComputeMG_ref(const SparseMatrix &A, const Vector &r, Vector &x)
{
  assert(x.localLength == A.localNumberOfColumns); // Make sure x contain space for halo values

  ZeroVector(x); // initialize x to zero

  int ierr = 0;
  if (A.mgData != 0)
  { // Go to next coarse level if defined
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i = 0; i < numberOfPresmootherSteps; ++i)
      ierr += ComputeSYMGS_ref(A, r, x);
    if (ierr != 0)
      return ierr;
    ierr = ComputeSPMV_ref(A, x, *A.mgData->Axf);
    if (ierr != 0)
      return ierr;
    // Perform restriction operation using simple injection
    ierr = ComputeRestriction_ref(A, r);
    if (ierr != 0)
      return ierr;
    ierr = ComputeMG_ref(*A.Ac, *A.mgData->rc, *A.mgData->xc);
    if (ierr != 0)
      return ierr;
    ierr = ComputeProlongation_ref(A, x);
    if (ierr != 0)
      return ierr;
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i = 0; i < numberOfPostsmootherSteps; ++i)
      ierr += ComputeSYMGS_ref(A, r, x);
    if (ierr != 0)
      return ierr;
  }
  else
  {
    ierr = ComputeSYMGS_ref(A, r, x);
    if (ierr != 0)
      return ierr;
  }
  return 0;
}
#endif
