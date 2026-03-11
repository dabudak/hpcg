
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
#ifndef HPCG_NO_LAIK
#include "laik/laik_x_vector.hpp"
#endif

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
#include <cstring>

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
#ifndef HPCG_NO_LAIK
int ComputeSPMV_laik_ref(const SparseMatrix &A, Laik_Blob *x, Laik_Blob *y) {
  assert(x->localLength == A.localNumberOfRows); // Test vector lengths
  assert(y->localLength == A.localNumberOfRows);

  // Switch x to ext to make halo values available
  {
    assert(x->toExtActions);
    Laik_Partitioning* x_active = laik_data_get_partitioning(x->values);
    if (x_active == x->localP && x->extP) {
      if (x->base && x->base_ext && x->base_ext != x->base) {
        std::memcpy(x->base_ext, x->base, x->localCount * sizeof(double));
      }
      EnsureLaikActionsToExt(x);
      laik_exec_actions(x->toExtActions);
    } else {
      assert(x_active == x->extP || x->extP == 0);
    }
  }

  double* xv = x->base;
  if (x->extP && laik_data_get_partitioning(x->values) == x->extP && x->base_ext)
    xv = x->base_ext;
  assert(xv);

  // Ensure y is in local partitioning
  {
    assert(y->toLocalActions);
    Laik_Partitioning* y_active = laik_data_get_partitioning(y->values);
    if (y_active == y->extP && y->localP) {
      EnsureLaikActionsToLocal(y);
      laik_exec_actions(y->toLocalActions);
    } else {
      assert(y_active == y->localP || y->extP == 0);
    }
  }
  double* yv = y->base;
  if (y->extP && laik_data_get_partitioning(y->values) == y->extP && y->base_ext)
    yv = y->base_ext;
  assert(yv);

  if (yv) {
    for (local_int_t i = 0; i < A.localNumberOfRows; ++i)
      yv[i] = 0.0;
  }

  assert(A.extMapBuilt);
  const GlobalToLocalMap &extMap = A.extLocalMap;

  const int mapNo = 0;
  double* val = 0; uint64_t val_len = 0;
  int64_t* col = 0; uint64_t col_len = 0;
  laik_get_map_1d(A.valD, mapNo, (void**)&val, &val_len);
  laik_get_map_1d(A.colD, mapNo, (void**)&col, &col_len);
  int mrCount = laik_my_maprangecount(A.rowsP, mapNo);
  int64_t range_val_offset = 0;
  for (int mr = 0; mr < mrCount; ++mr) {
      Laik_TaskRange* tr = laik_my_maprange(A.rowsP, mapNo, mr);
      const Laik_Range* s = laik_taskrange_get_range(tr);
      int64_t rf = s->from.i[0];
      int64_t rt = s->to.i[0];
      int64_t rowsHere = rt - rf;

      int row_map_no = -1;
      uint64_t row_lfrom_u = 0;
      Laik_Mapping* row_map = laik_global2maplocal_1d(A.rowD, rf, &row_map_no, &row_lfrom_u);
      assert(row_map);
      int64_t* rp_base = 0; uint64_t rp_len = 0;
      laik_get_map_1d(A.rowD, row_map_no, (void**)&rp_base, &rp_len);
      assert(row_lfrom_u + (uint64_t)(rowsHere + 1) <= rp_len);
      int64_t* row_ptr = rp_base + row_lfrom_u;

      int64_t base = row_ptr[0] - range_val_offset;
      for (int64_t i = 0; i < rowsHere; ++i) {
        int64_t beg = row_ptr[i] - base;
        int64_t end = row_ptr[i + 1] - base;
        double sum = 0.0;
        for (int64_t o = beg; o < end; ++o) {
          global_int_t gcol = (global_int_t)col[o];
          auto it = A.globalToLocalMap.find(gcol);
          local_int_t lcol = 0;
          if (it != A.globalToLocalMap.end()) {
            lcol = it->second;
          } else {
            auto eit = extMap.find(gcol);
            if (eit == extMap.end()) {
              std::fprintf(stderr,
                           "[rank %d] SpMV missing gcol=%lld (rf=%lld i=%lld)\n",
                           A.geom ? A.geom->rank : -1,
                           (long long)gcol,
                           (long long)rf,
                           (long long)i);
              std::abort();
            }
            lcol = eit->second;
          }
          sum += val[o] * xv[lcol];
        }
        auto lit = A.globalToLocalMap.find(rf + i);
        if (lit == A.globalToLocalMap.end()) {
          std::fprintf(stderr,
                       "[rank %d] SpMV missing row g=%lld\n",
                       A.geom ? A.geom->rank : -1,
                       (long long)(rf + i));
          std::abort();
        }
        local_int_t li = lit->second;
        yv[li] = sum;
      }
      range_val_offset += (row_ptr[rowsHere] - row_ptr[0]);
  }

  // Return x to local partitioning while preserving updated data
  {
    assert(x->toLocalActions);
    Laik_Partitioning* x_active = laik_data_get_partitioning(x->values);
    if (x_active == x->extP && x->localP) {
      EnsureLaikActionsToLocal(x);
      laik_exec_actions(x->toLocalActions);
    } else {
      assert(x_active == x->localP || x->extP == 0);
    }
  }

  return 0;
}
#endif

int ComputeSPMV_ref(const SparseMatrix & A, Vector & x, Vector & y) {

  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NO_MPI
  ExchangeHalo(A, x);
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
#ifndef HPCG_NO_LAIK
int ComputeSPMV_ref_laik(const SparseMatrix& A, std::vector<double>& y) {
  // Switch x to ext to trigger halo exchange (kelekcibo-style)
  {
    assert(A.x_blob->toExtActions);
    Laik_Partitioning* x_active = laik_data_get_partitioning(A.x_blob->values);
    if (x_active == A.x_blob->localP && A.x_blob->extP) {
      laik_exec_actions(A.x_blob->toExtActions);
    } else {
      assert(x_active == A.x_blob->extP || A.x_blob->extP == 0);
    }
  }
  double* xv = A.x_blob->base;
  if (A.x_blob->extP && laik_data_get_partitioning(A.x_blob->values) == A.x_blob->extP && A.x_blob->base_ext)
    xv = A.x_blob->base_ext;
  assert(xv);

  // Build external global -> local index map for fast lookup
  std::unordered_map<global_int_t, local_int_t> extMap;
  extMap.reserve((size_t)A.numberOfExternalValues * 2 + 1);
  for (local_int_t i = 0; i < A.numberOfExternalValues; ++i) {
    extMap[A.externalLocalToGlobal[i]] = A.localNumberOfRows + i;
  }

  y.assign(A.localNumberOfRows, 0.0);

  const int mapNo = 0;
  double* val = 0; uint64_t val_len = 0;
  int64_t* col = 0; uint64_t col_len = 0;
  laik_get_map_1d(A.valD, mapNo, (void**)&val, &val_len);
  laik_get_map_1d(A.colD, mapNo, (void**)&col, &col_len);
  int mrCount = laik_my_maprangecount(A.rowsP, mapNo);
  int64_t range_val_offset = 0;
  for (int mr = 0; mr < mrCount; ++mr) {
      Laik_TaskRange* tr = laik_my_maprange(A.rowsP, mapNo, mr);
      const Laik_Range* s = laik_taskrange_get_range(tr);
      int64_t rf = s->from.i[0];
      int64_t rt = s->to.i[0];
      int64_t rowsHere = rt - rf;

      int row_map_no = -1;
      uint64_t row_lfrom_u = 0;
      Laik_Mapping* row_map = laik_global2maplocal_1d(A.rowD, rf, &row_map_no, &row_lfrom_u);
      assert(row_map);
      int64_t* rp_base = 0; uint64_t rp_len = 0;
      laik_get_map_1d(A.rowD, row_map_no, (void**)&rp_base, &rp_len);
      assert(row_lfrom_u + (uint64_t)(rowsHere + 1) <= rp_len);
      int64_t* row_ptr = rp_base + row_lfrom_u;
      int64_t base = row_ptr[0] - range_val_offset;
      for (int64_t i = 0; i < rowsHere; ++i) {
        int64_t beg = row_ptr[i] - base;
        int64_t end = row_ptr[i + 1] - base;
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
      range_val_offset += (row_ptr[rowsHere] - row_ptr[0]);
  }

  return 0;
}
#endif
