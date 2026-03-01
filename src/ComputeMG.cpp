
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
 @file ComputeMG.cpp

 HPCG routine
 */

#ifndef HPCG_NO_LAIK
#include "laik/hpcg_laik.hpp"
#endif
#include "ComputeMG.hpp"
#include "ComputeMG_ref.hpp"

#if !defined(HPCG_NO_MPI) && !defined(HPCG_NO_LAIK)
int ComputeMG_laik(const SparseMatrix &A, const Laik_Blob *r, Laik_Blob *x)
{

  // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
  A.isMgOptimized = false;
  return ComputeMG_laik_ref(A, r, x);
}
#else
int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x) {

  // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
  A.isMgOptimized = false;
  return ComputeMG_ref(A, r, x);
}
#endif
