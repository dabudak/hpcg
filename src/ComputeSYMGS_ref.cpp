
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

#ifndef HPCG_NO_MPI
#include <cmath>
#include "ExchangeHalo.hpp"
#include "laik/hpcg_laik.hpp"
#endif
#include <cassert>
#include <iostream>
#include <cstdlib>
#include "ComputeSYMGS_ref.hpp"

int ComputeSYMGS_laik_ref(const SparseMatrix &A, const Laik_Blob *r, Laik_Blob *x)
{

  assert(x->localLength == A.localNumberOfRows);

  laik_switchto_partitioning(x->values, A.ext, LAIK_DF_Preserve, LAIK_RO_None);

  const local_int_t nrow = A.localNumberOfRows;
  double **matrixDiagonal = A.matrixDiagonal; // An array of pointers to the diagonal entries A.matrixValues

  const double * rv;
  double * xv;

  laik_get_map_1d(x->values, 0, (void **)&xv, 0);
  laik_get_map_1d(r->values, 0, (void **)&rv, 0);

  for (local_int_t i = 0; i < nrow; i++)
  {
    const double *const currentValues = A.matrixValues[i];
    const local_int_t *const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i];        // RHS value

    for (int j = 0; j < currentNumberOfNonzeros; j++)
    {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }

    sum += xv[i] * currentDiagonal; // Remove diagonal contribution from previous loop

    xv[i] = sum / currentDiagonal;
  }

  // Now the back sweep.

  for (local_int_t i = nrow - 1; i >= 0; i--)
  {
    const double *const currentValues = A.matrixValues[i];
    const local_int_t *const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i];       // RHS value

    for (int j = 0; j < currentNumberOfNonzeros; j++)
    {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }

    sum += xv[i] * currentDiagonal; // Remove diagonal contribution from previous loop
    xv[i] = sum / currentDiagonal;
  }

  laik_switchto_partitioning(x->values, A.local, LAIK_DF_None, LAIK_RO_None);

  return 0;
}

int ComputeSYMGS_ref(const SparseMatrix &A, const Vector &r, Vector &x)
{

  assert(x.localLength == A.localNumberOfColumns); // Make sure x contain space for halo values

#ifndef HPCG_NO_MPI
  ExchangeHalo(A, x);
#endif

  const local_int_t nrow = A.localNumberOfRows;
  double **matrixDiagonal = A.matrixDiagonal; // An array of pointers to the diagonal entries A.matrixValues
  const double *const rv = r.values;
  double *const xv = x.values;

  for (local_int_t i = 0; i < nrow; i++)
  {
    const double *const currentValues = A.matrixValues[i];
    const local_int_t *const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i];                                  // RHS value

    for (int j = 0; j < currentNumberOfNonzeros; j++)
    {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }

    sum += xv[i] * currentDiagonal; // Remove diagonal contribution from previous loop

    xv[i] = sum / currentDiagonal;
  }

  // Now the back sweep.

  for (local_int_t i = nrow - 1; i >= 0; i--)
  {
    const double *const currentValues = A.matrixValues[i];
    const local_int_t *const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i];                                  // RHS value

    for (int j = 0; j < currentNumberOfNonzeros; j++)
    {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }

    sum += xv[i] * currentDiagonal; // Remove diagonal contribution from previous loop
    xv[i] = sum / currentDiagonal;
  }

  return 0;
}
