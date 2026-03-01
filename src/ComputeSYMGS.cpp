
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
 @file ComputeSYMGS.cpp

 HPCG routine
 */

#ifndef HPCG_NO_LAIK
#include "laik/hpcg_laik.hpp"
#endif
#include "ComputeSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"

#ifndef HPCG_NO_LAIK
int ComputeSYMGS_laik(const SparseMatrix &A, const Laik_Blob *r, Laik_Blob *x)
{

  // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
  return ComputeSYMGS_laik_ref(A, r, x);
}
#endif

int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x) {

  // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
  return ComputeSYMGS_ref(A, r, x);
}
