
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
#include "laik/laik_x_vector.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#include <cmath>

/*!
  Routine to compute matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This is the reference SPMV implementation.  It CANNOT be modified for the
  purposes of this benchmark.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV
*/
int ComputeSPMV_ref( const SparseMatrix & A, Vector & x, Vector & y) {

  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NO_MPI
    ExchangeHalo(A,x);
#endif
  const double * const xv = x.values;
  double * const yv = y.values;
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
      sum += cur_vals[j]*xv[cur_inds[j]];
    yv[i] = sum;
  }
  return 0;
}

// LAIK-based SpMV using A.rowD/valD/colD and A.x_blob
int ComputeSPMV_ref_laik(const SparseMatrix& A, std::vector<double>& y) {
  const char* trace = std::getenv("HPCG_LAIK_TRACE");
  if (trace) {
    std::fprintf(stderr, "[rank %d] LAIK SpMV: switch x -> ext\n", A.geom ? A.geom->rank : -1);
    std::fflush(stderr);
  }
  // Switch x to ext to trigger halo exchange (kelekcibo-style)
  laik_switchto_partitioning(A.x_blob->values, A.ext, LAIK_DF_Preserve, LAIK_RO_None);
  if (trace) {
    std::fprintf(stderr, "[rank %d] LAIK SpMV: switched, get x map\n", A.geom ? A.geom->rank : -1);
    std::fflush(stderr);
  }
  double* xv = 0; uint64_t xcount = 0;
  laik_get_map_1d(A.x_blob->values, 0, (void**)&xv, &xcount);
  if (trace) {
    std::fprintf(stderr, "[rank %d] LAIK SpMV: xcount=%llu\n", A.geom ? A.geom->rank : -1,
                 (unsigned long long)xcount);
    std::fflush(stderr);
  }

  // Build external global -> local index map for fast lookup
  std::unordered_map<global_int_t, local_int_t> extMap;
  extMap.reserve((size_t)A.numberOfExternalValues * 2 + 1);
  for (local_int_t i = 0; i < A.numberOfExternalValues; ++i) {
    extMap[A.externalLocalToGlobal[i]] = A.localNumberOfRows + i;
  }

  // Check halo values: expected to be 1.0 (x initialized to ones)
  if (xv && A.numberOfExternalValues > 0) {
    int mismatches = 0;
    double minv = xv[A.localNumberOfRows];
    double maxv = xv[A.localNumberOfRows];
    for (local_int_t i = 0; i < A.numberOfExternalValues; ++i) {
      double v = xv[A.localNumberOfRows + i];
      if (v < minv) minv = v;
      if (v > maxv) maxv = v;
      if (std::fabs(v - 1.0) > 1e-12) mismatches++;
    }
    if (mismatches > 0 || trace) {
      std::fprintf(stderr,
                   "[rank %d] LAIK SpMV: halo check mismatches=%d/%lld min=%.6g max=%.6g\n",
                   A.geom ? A.geom->rank : -1,
                   mismatches,
                   (long long)A.numberOfExternalValues,
                   minv,
                   maxv);
      std::fflush(stderr);
    }
  }

  y.assign(A.localNumberOfRows, 0.0);

  int mapCount = laik_my_mapcount(A.rowsP);
  for (int mapNo = 0; mapNo < mapCount; ++mapNo) {
    int mrCount = laik_my_maprangecount(A.rowsP, mapNo);
    if (mrCount != 1) continue;
    Laik_TaskRange* tr = laik_my_maprange(A.rowsP, mapNo, 0);
    const Laik_Range* s = laik_taskrange_get_range(tr);
    int64_t rf = s->from.i[0];
    int64_t rt = s->to.i[0];

    int64_t* rp_base = 0; uint64_t rp_len = 0;
    laik_get_map_1d(A.rowD, mapNo, (void**)&rp_base, &rp_len);
    double* val = 0; uint64_t val_len = 0;
    int64_t* col = 0; uint64_t col_len = 0;
    laik_get_map_1d(A.valD, mapNo, (void**)&val, &val_len);
    laik_get_map_1d(A.colD, mapNo, (void**)&col, &col_len);

    int64_t rowsHere = rt - rf;
    int64_t base = rp_base[0];
    for (int64_t i = 0; i < rowsHere; ++i) {
      int64_t beg = rp_base[i] - base;
      int64_t end = rp_base[i + 1] - base;
      double sum = 0.0;
      for (int64_t o = beg; o < end; ++o) {
        global_int_t gcol = (global_int_t)col[o];
        auto it = A.globalToLocalMap.find(gcol);
        local_int_t lcol = 0;
        if (it != A.globalToLocalMap.end()) {
          lcol = it->second;
        } else {
          lcol = extMap[gcol];
        }
        sum += val[o] * xv[lcol];
      }
      local_int_t li = A.globalToLocalMap.find(rf + i)->second;
      y[li] = sum;
    }
  }

  return 0;
}
