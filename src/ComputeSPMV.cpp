
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
 @file ComputeSPMV.cpp

 HPCG routine
 */

#ifndef HPCG_NO_LAIK
#include "laik/hpcg_laik.hpp"
#endif
#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"

#ifndef HPCG_NO_LAIK
int ComputeSPMV_laik( const SparseMatrix & A, Laik_Blob *x_blob, Laik_Blob *y_blob) {

  // This line and the next two lines should be removed and your version of ComputeSPMV should be used.
  A.isSpmvOptimized = false;
  return ComputeSPMV_laik_ref(A, x_blob, y_blob);
}
#endif

int ComputeSPMV(const SparseMatrix &A, Vector &x, Vector &y)
{

  // This line and the next two lines should be removed and your version of ComputeSPMV should be used.
  A.isSpmvOptimized = false;
  return ComputeSPMV_ref(A, x, y);
}
