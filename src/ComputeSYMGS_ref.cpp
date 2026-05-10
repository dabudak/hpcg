
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
#endif
#ifndef HPCG_NO_LAIK
#include "laik/hpcg_laik.hpp"
#endif
#include <cassert>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <unordered_map>
#include <vector>
#include "ComputeSYMGS_ref.hpp"

#ifndef HPCG_NO_LAIK
int ComputeSYMGS_laik_ref(const SparseMatrix &A, const Laik_Blob *r, Laik_Blob *x)
{

  assert(x->localLength == A.localNumberOfRows);

  laik_switchto_partitioning(x->values, A.ext, LAIK_DF_Preserve, LAIK_RO_None);

  const local_int_t nrow = A.localNumberOfRows;
  double **matrixDiagonal = A.matrixDiagonal; // An array of pointers to the diagonal entries A.matrixValues
  double *matrixDiagonal_d = 0;
  if (A.matrixDiagonal_d)
    laik_get_map_1d(A.matrixDiagonal_d, 0, (void **)&matrixDiagonal_d, 0);

  const double * rv;
  double * xv;

  laik_get_map_1d(x->values, 0, (void **)&xv, 0);
  laik_get_map_1d(r->values, 0, (void **)&rv, 0);

  if (A.rowD && A.valD && A.colD && A.rowsP) {
    assert(A.extMapBuilt);
    const GlobalToLocalMap &extMap = A.extLocalMap;

    int mapCount = laik_my_mapcount(A.rowsP);
    for (int mapNo = 0; mapNo < mapCount; ++mapNo) {
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
        if (!row_map)
          continue;
        int64_t* rp_base = 0; uint64_t rp_len = 0;
        laik_get_map_1d(A.rowD, row_map_no, (void**)&rp_base, &rp_len);
        if ((uint64_t)(rowsHere + 1) > rp_len)
          continue;
        int64_t* row_ptr = rp_base;

        double* val = 0; uint64_t val_len = 0;
        int64_t* col = 0; uint64_t col_len = 0;
        laik_get_map_1d(A.valD, mapNo, (void**)&val, &val_len);
        laik_get_map_1d(A.colD, mapNo, (void**)&col, &col_len);

        int64_t base = row_ptr[0] - range_val_offset;
        for (int64_t i = 0; i < rowsHere; ++i) {
          global_int_t g = (global_int_t)(rf + i);
          auto lit = A.globalToLocalMap.find(g);
          if (lit == A.globalToLocalMap.end())
            continue;
          local_int_t li = lit->second;
          const double currentDiagonal = matrixDiagonal_d ? matrixDiagonal_d[li] : matrixDiagonal[li][0];
          int64_t beg = row_ptr[i] - base;
          int64_t end = row_ptr[i + 1] - base;
          double sum = rv[li];
          for (int64_t o = beg; o < end; ++o) {
            global_int_t gcol = (global_int_t)col[o];
            auto it = A.globalToLocalMap.find(gcol);
            local_int_t lcol = 0;
            if (it != A.globalToLocalMap.end())
              lcol = it->second;
            else
            {
              GlobalToLocalMap::const_iterator eit = extMap.find(gcol);
              assert(eit != extMap.end());
              lcol = eit->second;
            }
            sum -= val[o] * xv[lcol];
          }
          sum += xv[li] * currentDiagonal;
          xv[li] = sum / currentDiagonal;
        }
        range_val_offset += (row_ptr[rowsHere] - row_ptr[0]);
      }
    }
  } else {
    if (A.rowD && A.valD && A.colD && A.rowsP) {
      assert(A.extMapBuilt);
      const GlobalToLocalMap &extMap = A.extLocalMap;

      int mapCount = laik_my_mapcount(A.rowsP);
      for (int mapNo = 0; mapNo < mapCount; ++mapNo) {
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
          if (!row_map)
            continue;
          int64_t* rp_base = 0; uint64_t rp_len = 0;
          laik_get_map_1d(A.rowD, row_map_no, (void**)&rp_base, &rp_len);
          if ((uint64_t)(rowsHere + 1) > rp_len)
            continue;
          int64_t* row_ptr = rp_base;

          double* val = 0; uint64_t val_len = 0;
          int64_t* col = 0; uint64_t col_len = 0;
          laik_get_map_1d(A.valD, mapNo, (void**)&val, &val_len);
          laik_get_map_1d(A.colD, mapNo, (void**)&col, &col_len);

          int64_t base = row_ptr[0] - range_val_offset;
          for (int64_t i = 0; i < rowsHere; ++i) {
            global_int_t g = (global_int_t)(rf + i);
            auto lit = A.globalToLocalMap.find(g);
            if (lit == A.globalToLocalMap.end())
              continue;
            local_int_t li = lit->second;
            const double currentDiagonal = matrixDiagonal_d ? matrixDiagonal_d[li] : matrixDiagonal[li][0];
            int64_t beg = row_ptr[i] - base;
            int64_t end = row_ptr[i + 1] - base;
            double sum = rv[li];
            for (int64_t o = beg; o < end; ++o) {
              global_int_t gcol = (global_int_t)col[o];
              auto it = A.globalToLocalMap.find(gcol);
              local_int_t lcol = 0;
              if (it != A.globalToLocalMap.end())
                lcol = it->second;
              else
              {
                GlobalToLocalMap::const_iterator eit = extMap.find(gcol);
                assert(eit != extMap.end());
                lcol = eit->second;
              }
              sum -= val[o] * xv[lcol];
            }
            sum += xv[li] * currentDiagonal;
            xv[li] = sum / currentDiagonal;
          }
          range_val_offset += (row_ptr[rowsHere] - row_ptr[0]);
        }
      }
    } else {
      for (local_int_t i = 0; i < nrow; i++)
      {
        const double *const currentValues = A.matrixValues[i];
        const local_int_t *const currentColIndices = A.mtxIndL[i];
        const int currentNumberOfNonzeros = A.nonzerosInRow[i];
        const double currentDiagonal = matrixDiagonal_d ? matrixDiagonal_d[i] : matrixDiagonal[i][0]; // Current diagonal value
        double sum = rv[i];        // RHS value

        for (int j = 0; j < currentNumberOfNonzeros; j++)
        {
          local_int_t curCol = currentColIndices[j];
          sum -= currentValues[j] * xv[curCol];
        }

        sum += xv[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        xv[i] = sum / currentDiagonal;
      }
    }
  }

  // Now the back sweep.

  if (A.rowD && A.valD && A.colD && A.rowsP) {
    assert(A.extMapBuilt);
    const GlobalToLocalMap &extMap = A.extLocalMap;

    int mapCount = laik_my_mapcount(A.rowsP);
    for (int mapNo = mapCount - 1; mapNo >= 0; --mapNo) {
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
        if (!row_map)
          continue;
        int64_t* rp_base = 0; uint64_t rp_len = 0;
        laik_get_map_1d(A.rowD, row_map_no, (void**)&rp_base, &rp_len);
        if ((uint64_t)(rowsHere + 1) > rp_len)
          continue;
        int64_t* row_ptr = rp_base;

        double* val = 0; uint64_t val_len = 0;
        int64_t* col = 0; uint64_t col_len = 0;
        laik_get_map_1d(A.valD, mapNo, (void**)&val, &val_len);
        laik_get_map_1d(A.colD, mapNo, (void**)&col, &col_len);

        int64_t base = row_ptr[0] - range_val_offset;
        for (int64_t i = rowsHere - 1; i >= 0; --i) {
          global_int_t g = (global_int_t)(rf + i);
          auto lit = A.globalToLocalMap.find(g);
          if (lit == A.globalToLocalMap.end())
            continue;
          local_int_t li = lit->second;
          const double currentDiagonal = matrixDiagonal_d ? matrixDiagonal_d[li] : matrixDiagonal[li][0];
          int64_t beg = row_ptr[i] - base;
          int64_t end = row_ptr[i + 1] - base;
          double sum = rv[li];
          for (int64_t o = beg; o < end; ++o) {
            global_int_t gcol = (global_int_t)col[o];
            auto it = A.globalToLocalMap.find(gcol);
            local_int_t lcol = 0;
            if (it != A.globalToLocalMap.end())
              lcol = it->second;
            else
            {
              GlobalToLocalMap::const_iterator eit = extMap.find(gcol);
              assert(eit != extMap.end());
              lcol = eit->second;
            }
            sum -= val[o] * xv[lcol];
          }
          sum += xv[li] * currentDiagonal;
          xv[li] = sum / currentDiagonal;
        }
        range_val_offset += (row_ptr[rowsHere] - row_ptr[0]);
      }
    }
  } else {
    if (A.rowD && A.valD && A.colD && A.rowsP) {
      assert(A.extMapBuilt);
      const GlobalToLocalMap &extMap = A.extLocalMap;

      int mapCount = laik_my_mapcount(A.rowsP);
      for (int mapNo = mapCount - 1; mapNo >= 0; --mapNo) {
        int mrCount = laik_my_maprangecount(A.rowsP, mapNo);
        std::vector<int64_t> offsets(mrCount, 0);
        int64_t range_val_offset = 0;
        for (int mr = 0; mr < mrCount; ++mr) {
          Laik_TaskRange* tr = laik_my_maprange(A.rowsP, mapNo, mr);
          const Laik_Range* s = laik_taskrange_get_range(tr);
          int64_t rf = s->from.i[0];
          int64_t rt = s->to.i[0];
          int64_t rowsHere = rt - rf;

          offsets[mr] = range_val_offset;
          int row_map_no = -1;
          uint64_t row_lfrom_u = 0;
          Laik_Mapping* row_map = laik_global2maplocal_1d(A.rowD, rf, &row_map_no, &row_lfrom_u);
          if (!row_map)
            continue;
          int64_t* rp_base = 0; uint64_t rp_len = 0;
          laik_get_map_1d(A.rowD, row_map_no, (void**)&rp_base, &rp_len);
          if ((uint64_t)(rowsHere + 1) > rp_len)
            continue;
          int64_t* row_ptr = rp_base;
          range_val_offset += (row_ptr[rowsHere] - row_ptr[0]);
        }

        for (int mr = mrCount - 1; mr >= 0; --mr) {
          Laik_TaskRange* tr = laik_my_maprange(A.rowsP, mapNo, mr);
          const Laik_Range* s = laik_taskrange_get_range(tr);
          int64_t rf = s->from.i[0];
          int64_t rt = s->to.i[0];
          int64_t rowsHere = rt - rf;

          int row_map_no = -1;
          uint64_t row_lfrom_u = 0;
          Laik_Mapping* row_map = laik_global2maplocal_1d(A.rowD, rf, &row_map_no, &row_lfrom_u);
          if (!row_map)
            continue;
          int64_t* rp_base = 0; uint64_t rp_len = 0;
          laik_get_map_1d(A.rowD, row_map_no, (void**)&rp_base, &rp_len);
          if ((uint64_t)(rowsHere + 1) > rp_len)
            continue;
          int64_t* row_ptr = rp_base;

          double* val = 0; uint64_t val_len = 0;
          int64_t* col = 0; uint64_t col_len = 0;
          laik_get_map_1d(A.valD, mapNo, (void**)&val, &val_len);
          laik_get_map_1d(A.colD, mapNo, (void**)&col, &col_len);

          int64_t base = row_ptr[0] - offsets[mr];
          for (int64_t i = rowsHere - 1; i >= 0; --i) {
            global_int_t g = (global_int_t)(rf + i);
            auto lit = A.globalToLocalMap.find(g);
            if (lit == A.globalToLocalMap.end())
              continue;
            local_int_t li = lit->second;
            const double currentDiagonal = matrixDiagonal_d ? matrixDiagonal_d[li] : matrixDiagonal[li][0];
            int64_t beg = row_ptr[i] - base;
            int64_t end = row_ptr[i + 1] - base;
            double sum = rv[li];
            for (int64_t o = beg; o < end; ++o) {
              global_int_t gcol = (global_int_t)col[o];
              auto it = A.globalToLocalMap.find(gcol);
              local_int_t lcol = 0;
              if (it != A.globalToLocalMap.end())
                lcol = it->second;
              else
              {
                GlobalToLocalMap::const_iterator eit = extMap.find(gcol);
                assert(eit != extMap.end());
                lcol = eit->second;
              }
              sum -= val[o] * xv[lcol];
            }
            sum += xv[li] * currentDiagonal;
            xv[li] = sum / currentDiagonal;
          }
        }
      }
    } else {
      for (local_int_t i = nrow - 1; i >= 0; i--)
      {
        const double *const currentValues = A.matrixValues[i];
        const local_int_t *const currentColIndices = A.mtxIndL[i];
        const int currentNumberOfNonzeros = A.nonzerosInRow[i];
        const double currentDiagonal = matrixDiagonal_d ? matrixDiagonal_d[i] : matrixDiagonal[i][0]; // Current diagonal value
        double sum = rv[i];       // RHS value

        for (int j = 0; j < currentNumberOfNonzeros; j++)
        {
          local_int_t curCol = currentColIndices[j];
          sum -= currentValues[j] * xv[curCol];
        }

        sum += xv[i] * currentDiagonal; // Remove diagonal contribution from previous loop
        xv[i] = sum / currentDiagonal;
      }
    }
  }

  // Preserve updated x when switching back to local partitioning.
  laik_switchto_partitioning(x->values, A.local, LAIK_DF_Preserve, LAIK_RO_None);

  return 0;
}
#endif

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
