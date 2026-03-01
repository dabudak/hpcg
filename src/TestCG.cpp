
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
 @file TestCG.cpp

 HPCG routine
 */

// Changelog
//
// Version 0.4
// - Added timing of setup time for sparse MV
// - Corrected percentages reported for sparse MV with overhead
//
/////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>
using std::endl;
#include <vector>
#include <cmath>

#ifndef HPCG_NO_LAIK
#include "laik/hpcg_laik.hpp"
#endif
#include "hpcg.hpp"
#include "TestCG.hpp"
#include "CG.hpp"
#include "CG_ref.hpp"
#include "Vector.hpp"

#ifndef HPCG_NO_LAIK
int TestCG_laik(SparseMatrix &A, CGData &data, Laik_Blob *b, Laik_Blob *x, TestCGData &testcg_data)
{
  // Use this array for collecting timing information
  std::vector<double> times(8, 0.0);
  // Temporary storage for holding original diagonal and RHS
  Vector origDiagA;
  Vector exaggeratedDiagA;
  Vector origB;
  InitializeVector(origDiagA, A.localNumberOfRows);
  InitializeVector(exaggeratedDiagA, A.localNumberOfRows);
  InitializeVector(origB, A.localNumberOfRows);
  CopyMatrixDiagonal(A, origDiagA);
  CopyVector(origDiagA, exaggeratedDiagA);
  CopyLaikVectorToVector(b, origB);

  // Modify the matrix diagonal to greatly exaggerate diagonal values.
  // CG should converge in about 10 iterations for this problem, regardless of problem size
  for (local_int_t i = 0; i < A.localNumberOfRows; ++i)
  {
    global_int_t globalRowID = A.localToGlobalMap[i];
    if (globalRowID < 9)
    {
      double scale = (globalRowID + 2) * 1.0e6;
      ScaleVectorValue(exaggeratedDiagA, i, scale);
      ScaleLaikVectorValue(b, i, scale);
    }
    else
    {
      ScaleVectorValue(exaggeratedDiagA, i, 1.0e6);
      ScaleLaikVectorValue(b, i, 1.0e6);
    }
  }

  ReplaceMatrixDiagonal(A, exaggeratedDiagA);

  const char* diag_check = std::getenv("HPCG_LAIK_DIAG_CHECK");
  if (diag_check && diag_check[0] != '\0' && A.rowD && A.valD && A.colD && A.matrixDiagonal_d && A.world)
  {
    if (laik_size(A.world) == 1) {
      int64_t* rp = 0; uint64_t rp_len = 0;
      int64_t* col = 0; uint64_t col_len = 0;
      double* val = 0; uint64_t val_len = 0;
      double* diag_d = 0; uint64_t diag_len = 0;
      laik_get_map_1d(A.rowD, 0, (void**)&rp, &rp_len);
      laik_get_map_1d(A.colD, 0, (void**)&col, &col_len);
      laik_get_map_1d(A.valD, 0, (void**)&val, &val_len);
      laik_get_map_1d(A.matrixDiagonal_d, 0, (void**)&diag_d, &diag_len);

      local_int_t rows = A.localNumberOfRows;
      local_int_t sample = rows < 10 ? rows : 10;
      int missing = 0;
      double max_diff = 0.0;
      for (local_int_t i = 0; i < sample; ++i) {
        global_int_t g = A.localToGlobalMap[i];
        int64_t beg = rp[g];
        int64_t end = rp[g + 1];
        double csr_diag = 0.0;
        bool found = false;
        for (int64_t o = beg; o < end; ++o) {
          if (col[o] == g) {
            csr_diag = val[o];
            found = true;
            break;
          }
        }
        if (!found) {
          missing++;
          continue;
        }
        double diff = std::abs(csr_diag - diag_d[i]);
        if (diff > max_diff) max_diff = diff;
      }
      if (A.geom->rank == 0) {
        std::cout << "Diag check: sample=" << sample << " missing=" << missing
                  << " max diff=" << max_diff << std::endl;
      }
    }
  }

  int niters = 0;
  double normr = 0.0;
  double normr0 = 0.0;
  int maxIters = 50;
  int numberOfCgCalls = 2;
  double tolerance = 1.0e-12;               // Set tolerance to reasonable value for grossly scaled diagonal terms
  testcg_data.expected_niters_no_prec = 12; // For the unpreconditioned CG call, we should take about 10 iterations, permit 12
  testcg_data.expected_niters_prec = 2;     // For the preconditioned case, we should take about 1 iteration, permit 2
  testcg_data.niters_max_no_prec = 0;
  testcg_data.niters_max_prec = 0;
  for (int k = 0; k < 2; ++k)
  { // This loop tests both unpreconditioned and preconditioned runs
    int expected_niters = testcg_data.expected_niters_no_prec;
    if (k == 1)
      expected_niters = testcg_data.expected_niters_prec;
    for (int i = 0; i < numberOfCgCalls; ++i)
    {
      ZeroLaikVector(x); // Zero out x
      int ierr = CG_laik(A, data, b, x, maxIters, tolerance, niters, normr, normr0, &times[0], k == 1); // true);
      if (ierr)
        HPCG_fout << "Error in call to CG: " << ierr << ".\n"
                  << endl;
      const char* cg_dbg = std::getenv("HPCG_LAIK_CG_DEBUG");
      if (cg_dbg && cg_dbg[0] != '\0' && A.geom->rank == 0) {
        double scaled = (normr0 != 0.0) ? (normr / normr0) : -1.0;
        std::cout << "TestCG " << (k == 1 ? "prec" : "noprec")
                  << " call " << i
                  << " niters=" << niters
                  << " scaled=" << scaled
                  << (std::isfinite(scaled) ? "" : " (non-finite)")
                  << std::endl;
      }
      if (niters <= expected_niters)
      {
        ++testcg_data.count_pass;
      }
      else
      {
        ++testcg_data.count_fail;
      }
      if (k == 0 && niters > testcg_data.niters_max_no_prec)
        testcg_data.niters_max_no_prec = niters; // Keep track of largest iter count
      if (k == 1 && niters > testcg_data.niters_max_prec)
        testcg_data.niters_max_prec = niters; // Same for preconditioned run
      if (A.geom->rank == 0)
      {
        HPCG_fout << "Call [" << i << "] Number of Iterations [" << niters << "] Scaled Residual [" << normr / normr0 << "]" << endl;
        if (niters > expected_niters)
          HPCG_fout << " Expected " << expected_niters << " iterations.  Performed " << niters << "." << endl;
      }
    }
  }

  // Restore matrix diagonal and RHS
  ReplaceMatrixDiagonal(A, origDiagA);
  CopyVectorToLaikVector(origB, b);
  // Delete vectors
  DeleteVector(origDiagA);
  DeleteVector(exaggeratedDiagA);
  DeleteVector(origB);

  testcg_data.normr = normr;

  return 0;
}
#else
int TestCG(SparseMatrix & A, CGData & data, Vector & b, Vector & x, TestCGData & testcg_data) {


  // Use this array for collecting timing information
  std::vector< double > times(8,0.0);
  // Temporary storage for holding original diagonal and RHS
  Vector origDiagA, exaggeratedDiagA, origB;
  InitializeVector(origDiagA, A.localNumberOfRows);
  InitializeVector(exaggeratedDiagA, A.localNumberOfRows);
  InitializeVector(origB, A.localNumberOfRows);
  CopyMatrixDiagonal(A, origDiagA);
  CopyVector(origDiagA, exaggeratedDiagA);
  CopyVector(b, origB);

  // Modify the matrix diagonal to greatly exaggerate diagonal values.
  // CG should converge in about 10 iterations for this problem, regardless of problem size
  for (local_int_t i=0; i< A.localNumberOfRows; ++i) {
    global_int_t globalRowID = A.localToGlobalMap[i];
    if (globalRowID<9) {
      double scale = (globalRowID+2)*1.0e6;
      ScaleVectorValue(exaggeratedDiagA, i, scale);
      ScaleVectorValue(b, i, scale);
    } else {
      ScaleVectorValue(exaggeratedDiagA, i, 1.0e6);
      ScaleVectorValue(b, i, 1.0e6);
    }
  }
  ReplaceMatrixDiagonal(A, exaggeratedDiagA);

  int niters = 0;
  double normr = 0.0;
  double normr0 = 0.0;
  int maxIters = 50;
  int numberOfCgCalls = 2;
  double tolerance = 1.0e-12; // Set tolerance to reasonable value for grossly scaled diagonal terms
  testcg_data.expected_niters_no_prec = 12; // For the unpreconditioned CG call, we should take about 10 iterations, permit 12
  testcg_data.expected_niters_prec = 2;   // For the preconditioned case, we should take about 1 iteration, permit 2
  testcg_data.niters_max_no_prec = 0;
  testcg_data.niters_max_prec = 0;
  for (int k=0; k<2; ++k) { // This loop tests both unpreconditioned and preconditioned runs
    int expected_niters = testcg_data.expected_niters_no_prec;
    if (k==1) expected_niters = testcg_data.expected_niters_prec;
    for (int i=0; i< numberOfCgCalls; ++i) {
      ZeroVector(x); // Zero out x
      int ierr = CG(A, data, b, x, maxIters, tolerance, niters, normr, normr0, &times[0], k==1);
      if (ierr) HPCG_fout << "Error in call to CG: " << ierr << ".\n" << endl;
      if (niters <= expected_niters) {
        ++testcg_data.count_pass;
      } else {
        ++testcg_data.count_fail;
      }
      if (k==0 && niters>testcg_data.niters_max_no_prec) testcg_data.niters_max_no_prec = niters; // Keep track of largest iter count
      if (k==1 && niters>testcg_data.niters_max_prec) testcg_data.niters_max_prec = niters; // Same for preconditioned run
      if (A.geom->rank==0) {
        HPCG_fout << "Call [" << i << "] Number of Iterations [" << niters <<"] Scaled Residual [" << normr/normr0 << "]" << endl;
        if (niters > expected_niters)
          HPCG_fout << " Expected " << expected_niters << " iterations.  Performed " << niters << "." << endl;
      }
    }
  }

  // Restore matrix diagonal and RHS
  ReplaceMatrixDiagonal(A, origDiagA);
  CopyVector(origB, b);
  // Delete vectors
  DeleteVector(origDiagA);
  DeleteVector(exaggeratedDiagA);
  DeleteVector(origB);
  testcg_data.normr = normr;

  return 0;
}
#endif
