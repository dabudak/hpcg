
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
 @file GenerateProblem_ref.cpp

 HPCG routine
 */

#if !defined(HPCG_NO_MPI) && defined(HPCG_NO_LAIK)
#include <mpi.h>
#endif
#ifndef HPCG_NO_LAIK
#include "laik/laik_reductions.hpp"
#include <laik.h>
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#if defined(HPCG_DEBUG) || defined(HPCG_DETAILED_DEBUG)
#include <fstream>
using std::endl;
#include "hpcg.hpp"
#endif
#include <cassert>
#include <cstdlib>

#include "GenerateProblem_ref.hpp"
#include "SetupHalo.hpp"
#ifndef HPCG_NO_LAIK
#include "laik/laik_x_vector.hpp"
#include <laik/data.h>
#endif


/*!
  Reference version of GenerateProblem to generate the sparse matrix, right hand side, initial guess, and exact solution.

  @param[in]  A      The known system matrix
  @param[inout] b      The newly allocated and generated right hand side vector (if b!=0 on entry)
  @param[inout] x      The newly allocated solution vector with entries set to 0.0 (if x!=0 on entry)
  @param[inout] xexact The newly allocated solution vector with entries set to the exact solution (if the xexact!=0 non-zero on entry)

  @see GenerateGeometry
*/

void GenerateProblem_ref(SparseMatrix & A, Vector * b, Vector * x, Vector * xexact) {

  // Make local copies of geometry information.  Use global_int_t since the RHS products in the calculations
  // below may result in global range values.
  global_int_t nx = A.geom->nx;
  global_int_t ny = A.geom->ny;
  global_int_t nz = A.geom->nz;
  global_int_t gnx = A.geom->gnx;
  global_int_t gny = A.geom->gny;
  global_int_t gnz = A.geom->gnz;
  global_int_t gix0 = A.geom->gix0;
  global_int_t giy0 = A.geom->giy0;
  global_int_t giz0 = A.geom->giz0;

  local_int_t localNumberOfRows = nx*ny*nz; // This is the size of our subblock
  // If this assert fails, it most likely means that the local_int_t is set to int and should be set to long long
  assert(localNumberOfRows>0); // Throw an exception of the number of rows is less than zero (can happen if int overflow)
  local_int_t numberOfNonzerosPerRow = 27; // We are approximating a 27-point finite element/volume/difference 3D stencil

  global_int_t totalNumberOfRows = gnx*gny*gnz; // Total number of grid points in mesh
  // If this assert fails, it most likely means that the global_int_t is set to int and should be set to long long
  assert(totalNumberOfRows>0); // Throw an exception of the number of rows is less than zero (can happen if int overflow)


  // Allocate arrays that are of length localNumberOfRows
  char * nonzerosInRow = new char[localNumberOfRows];
  global_int_t ** mtxIndG = new global_int_t*[localNumberOfRows];
  local_int_t  ** mtxIndL = new local_int_t*[localNumberOfRows];
  double ** matrixValues = new double *[localNumberOfRows];
  double ** matrixDiagonal = new double *[localNumberOfRows];

  if (b!=0) InitializeVector(*b, localNumberOfRows);
  if (x!=0) InitializeVector(*x, localNumberOfRows);
  if (xexact!=0) InitializeVector(*xexact, localNumberOfRows);
  double * bv = 0;
  double * xv = 0;
  double * xexactv = 0;
  if (b!=0) bv = b->values; // Only compute exact solution if requested
  if (x!=0) xv = x->values; // Only compute exact solution if requested
  if (xexact!=0) xexactv = xexact->values; // Only compute exact solution if requested
  A.localToGlobalMap.resize(localNumberOfRows);

  // Use a parallel loop to do initial assignment:
  // distributes the physical placement of arrays of pointers across the memory system
#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
  for (local_int_t i=0; i< localNumberOfRows; ++i) {
    matrixValues[i] = 0;
    matrixDiagonal[i] = 0;
    mtxIndG[i] = 0;
    mtxIndL[i] = 0;
  }

#ifndef HPCG_CONTIGUOUS_ARRAYS
  // Now allocate the arrays pointed to
  for (local_int_t i=0; i< localNumberOfRows; ++i)
    mtxIndL[i] = new local_int_t[numberOfNonzerosPerRow];
  for (local_int_t i=0; i< localNumberOfRows; ++i)
   matrixValues[i] = new double[numberOfNonzerosPerRow];
  for (local_int_t i=0; i< localNumberOfRows; ++i)
   mtxIndG[i] = new global_int_t[numberOfNonzerosPerRow];

#else
  // Now allocate the arrays pointed to
  mtxIndL[0] = new local_int_t[localNumberOfRows * numberOfNonzerosPerRow];
  matrixValues[0] = new double[localNumberOfRows * numberOfNonzerosPerRow];
  mtxIndG[0] = new global_int_t[localNumberOfRows * numberOfNonzerosPerRow];

  for (local_int_t i=1; i< localNumberOfRows; ++i) {
  mtxIndL[i] = mtxIndL[0] + i * numberOfNonzerosPerRow;
  matrixValues[i] = matrixValues[0] + i * numberOfNonzerosPerRow;
  mtxIndG[i] = mtxIndG[0] + i * numberOfNonzerosPerRow;
  }
#endif

  local_int_t localNumberOfNonzeros = 0;
  // TODO:  This triply nested loop could be flattened or use nested parallelism
#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
  for (local_int_t iz=0; iz<nz; iz++) {
    global_int_t giz = giz0+iz;
    for (local_int_t iy=0; iy<ny; iy++) {
      global_int_t giy = giy0+iy;
      for (local_int_t ix=0; ix<nx; ix++) {
        global_int_t gix = gix0+ix;
        local_int_t currentLocalRow = iz*nx*ny+iy*nx+ix;
        global_int_t currentGlobalRow = giz*gnx*gny+giy*gnx+gix;
#ifndef HPCG_NO_OPENMP
// C++ std::map is not threadsafe for writing
        #pragma omp critical
#endif
        A.globalToLocalMap[currentGlobalRow] = currentLocalRow;

        A.localToGlobalMap[currentLocalRow] = currentGlobalRow;
#ifdef HPCG_DETAILED_DEBUG
        HPCG_fout << " rank, globalRow, localRow = " << A.geom->rank << " " << currentGlobalRow << " " << A.globalToLocalMap[currentGlobalRow] << endl;
#endif
        char numberOfNonzerosInRow = 0;
        double * currentValuePointer = matrixValues[currentLocalRow];   // Pointer to current value in current row
        global_int_t * currentIndexPointerG = mtxIndG[currentLocalRow]; // Pointer to current index in current row
        for (int sz=-1; sz<=1; sz++) {
          if (giz+sz>-1 && giz+sz<gnz) {
            for (int sy=-1; sy<=1; sy++) {
              if (giy+sy>-1 && giy+sy<gny) {
                for (int sx=-1; sx<=1; sx++) {
                  if (gix+sx>-1 && gix+sx<gnx) {
                    global_int_t curcol = currentGlobalRow+sz*gnx*gny+sy*gnx+sx;
                    if (curcol == currentGlobalRow) {
                      matrixDiagonal[currentLocalRow] = currentValuePointer;
                      *currentValuePointer++ = 26.0;
                    } else {
                      *currentValuePointer++ = -1.0;
                    }
                    *currentIndexPointerG++ = curcol;
                    numberOfNonzerosInRow++;
                  } // end x bounds test
                } // end sx loop
              } // end y bounds test
            } // end sy loop
          } // end z bounds test
        } // end sz loop
        nonzerosInRow[currentLocalRow] = numberOfNonzerosInRow;
#ifndef HPCG_NO_OPENMP
        #pragma omp critical
#endif
        localNumberOfNonzeros += numberOfNonzerosInRow; // Protect this with an atomic
        if (b!=0)      bv[currentLocalRow] = 26.0 - ((double) (numberOfNonzerosInRow-1));
        if (x!=0)      xv[currentLocalRow] = 0.0;
        if (xexact!=0) xexactv[currentLocalRow] = 1.0;
      } // end ix loop
    } // end iy loop
  } // end iz loop
#ifdef HPCG_DETAILED_DEBUG
  HPCG_fout     << "Process " << A.geom->rank << " of " << A.geom->size <<" has " << localNumberOfRows    << " rows."     << endl
      << "Process " << A.geom->rank << " of " << A.geom->size <<" has " << localNumberOfNonzeros<< " nonzeros." <<endl;
#endif

  global_int_t totalNumberOfNonzeros = 0;
#ifndef HPCG_NO_MPI
  // Use a reduce function to sum all nonzeros
#ifdef HPCG_NO_LONG_LONG
#ifdef HPCG_NO_LAIK
  MPI_Allreduce(&localNumberOfNonzeros, &totalNumberOfNonzeros, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
  laik_allreduce(&localNumberOfNonzeros, &totalNumberOfNonzeros, 1, laik_Int32, LAIK_RO_Sum);
#endif
#else
  long long lnnz = localNumberOfNonzeros, gnnz = 0; // convert to 64 bit
#ifdef HPCG_NO_LAIK
  MPI_Allreduce(&lnnz, &gnnz, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
#else
  laik_allreduce(&lnnz, &gnnz, 1, laik_Int64, LAIK_RO_Sum);
#endif
  totalNumberOfNonzeros = gnnz; // Copy back
#endif
#else
  totalNumberOfNonzeros = localNumberOfNonzeros;
#endif
  const char* gen_dbg = std::getenv("HPCG_LAIK_GEN_DEBUG");
  if (gen_dbg && gen_dbg[0] != '\0') {
    fprintf(stderr,
            "[rank %d] GenerateProblem_ref: nx=%lld ny=%lld nz=%lld gnx=%lld gny=%lld gnz=%lld "
            "gix0=%lld giy0=%lld giz0=%lld localRows=%d localNNZ=%d totalNNZ=%lld\n",
            A.geom ? A.geom->rank : -1,
            (long long)nx, (long long)ny, (long long)nz,
            (long long)gnx, (long long)gny, (long long)gnz,
            (long long)gix0, (long long)giy0, (long long)giz0,
            (int)localNumberOfRows, (int)localNumberOfNonzeros,
            (long long)totalNumberOfNonzeros);
  }

  // If this assert fails, it most likely means that the global_int_t is set to int and should be set to long long
  // This assert is usually the first to fail as problem size increases beyond the 32-bit integer range.
  assert(totalNumberOfNonzeros>0); // Throw an exception of the number of nonzeros is less than zero (can happen if int overflow)

  A.title = 0;
  A.totalNumberOfRows = totalNumberOfRows;
  A.totalNumberOfNonzeros = totalNumberOfNonzeros;
  A.localNumberOfRows = localNumberOfRows;
  A.localNumberOfColumns = localNumberOfRows;
  A.localNumberOfNonzeros = localNumberOfNonzeros;
  A.nonzerosInRow = nonzerosInRow;
  A.mtxIndG = mtxIndG;
  A.mtxIndL = mtxIndL;
  A.matrixValues = matrixValues;
  A.matrixDiagonal = matrixDiagonal;

  // Build halo info before LAIK initialization
  SetupHalo(A);

#ifndef HPCG_NO_LAIK
  // LAIK CSR initialization for SpMV (based on old tcp2 fix)
  const char* csr_dbg = std::getenv("HPCG_LAIK_CSR_DEBUG");
  if (csr_dbg && csr_dbg[0] != '\0') {
    fprintf(stderr,
            "[rank %d] CSR debug: hpcg_instance=%p world=%p\n",
            A.geom ? A.geom->rank : -1,
            (void*)hpcg_instance,
            (void*)world);
    fflush(stderr);
  }

  if (hpcg_instance && world) {
    if (csr_dbg && csr_dbg[0] != '\0') {
      fprintf(stderr,
              "[rank %d] CSR setup start: totalRows=%lld\n",
              laik_myid(world), (long long)A.totalNumberOfRows);
      fflush(stderr);
    }
    A.inst = hpcg_instance;
    A.world = world;
    if (!A.space)
      A.space = laik_new_space_1d(A.inst, A.totalNumberOfRows);

    if (!A.local || !A.ext) {
      partition_d* local_pd = new partition_d();
      partition_d* ext_pd = new partition_d();
      init_partition_data(A, local_pd, ext_pd);
      init_partitionings(A, local_pd, ext_pd, A.world);
    }

    if (!A.x_blob) A.x_blob = init_blob(A, true, "x", A.local, A.ext);

    if (!A.matrixDiagonal_d) {
      A.matrixDiagonal_d = laik_new_data(A.space, laik_Double);
      Laik_Data_Parameters* dparams = (Laik_Data_Parameters*)malloc(sizeof(*dparams));
      dparams->prefix_row_data = 0;
      dparams->vector_local_indices = reinterpret_cast<const int64_t*>(A.localToGlobalMap.data());
      dparams->vector_local_count = (uint64_t)A.localNumberOfRows;
      dparams->vector_external_indices = 0;
      dparams->vector_external_count = 0;
      laik_data_attach_params(A.matrixDiagonal_d, dparams);
      laik_data_set_layout_factory(A.matrixDiagonal_d, laik_new_layout_vector);
      laik_switchto_partitioning(A.matrixDiagonal_d, A.local, LAIK_DF_None, LAIK_RO_None);
    }

    // Mirror diagonal into LAIK data for SYMGS/validation paths.
    double* diag_d = 0; uint64_t diag_len = 0;
    laik_get_map_1d(A.matrixDiagonal_d, 0, (void**)&diag_d, &diag_len);
    if (diag_d) {
      for (local_int_t i = 0; i < A.localNumberOfRows; ++i) {
        diag_d[i] = A.matrixDiagonal[i][0];
      }
    }

    if (!A.rowSpacePrefix)
      A.rowSpacePrefix = laik_new_space_1d(A.inst, A.totalNumberOfRows + 1);

    if (!A.rowD) A.rowD = laik_new_data(A.rowSpacePrefix, laik_Int64);
    if (!A.valD) A.valD = laik_new_data(A.space, laik_Double);
    if (!A.colD) A.colD = laik_new_data(A.space, laik_Int64);

    Laik_Data_Parameters* vparams_val = (Laik_Data_Parameters*)malloc(sizeof(*vparams_val));
    vparams_val->prefix_row_data = A.rowD;
    vparams_val->var_ranges = 0;
    vparams_val->var_range_count = 0;

    Laik_Data_Parameters* vparams_col = (Laik_Data_Parameters*)malloc(sizeof(*vparams_col));
    vparams_col->prefix_row_data = A.rowD;
    vparams_col->var_ranges = 0;
    vparams_col->var_range_count = 0;

    laik_data_attach_params(A.valD, vparams_val);
    laik_data_attach_params(A.colD, vparams_col);
    laik_data_set_layout_factory(A.valD, laik_new_layout_variable);
    laik_data_set_layout_factory(A.colD, laik_new_layout_variable);

    Laik_Partitioner* master_pr = laik_new_master_partitioner();
    Laik_Partitioning* row_master = laik_new_partitioning(master_pr, A.world, A.rowSpacePrefix, 0);
    Laik_Partitioning* rows_master = laik_new_partitioning(master_pr, A.world, A.space, 0);

    if (!A.rowP || !A.rowsP) {
      if (laik_size(A.world) == 1) {
        A.rowP = row_master;
        A.rowsP = rows_master;
      } else {
        partition_d* local_pd = new partition_d();
        partition_d* ext_pd = new partition_d();
        init_partition_data(A, local_pd, ext_pd);
        Laik_Partitioner* rows_pr = laik_new_partitioner("rows_pr", partitioner_alg_for_rows,
                                                        (void*)local_pd, LAIK_PF_None);
        A.rowP = laik_new_partitioning(rows_pr, A.world, A.rowSpacePrefix, NULL);
        A.rowsP = laik_new_partitioning(rows_pr, A.world, A.space, NULL);
      }
    }

    laik_switchto_partitioning(A.rowD, row_master, LAIK_DF_None, LAIK_RO_None);

    if (laik_myid(A.world) == 0) {
      int64_t* rp = 0; uint64_t rp_len = 0;
      laik_get_map_1d(A.rowD, 0, (void**)&rp, &rp_len);
      assert(rp_len == (uint64_t)(A.totalNumberOfRows + 1));
      int64_t off = 0;
      for (int64_t g = 0; g < A.totalNumberOfRows; ++g) {
        int64_t gix = g % gnx;
        int64_t giy = (g / gnx) % gny;
        int64_t giz = g / (gnx * gny);
        int nnz = 0;
        for (int sz = -1; sz <= 1; ++sz) {
          int64_t z = giz + sz;
          if (z < 0 || z >= gnz) continue;
          for (int sy = -1; sy <= 1; ++sy) {
            int64_t y = giy + sy;
            if (y < 0 || y >= gny) continue;
            for (int sx = -1; sx <= 1; ++sx) {
              int64_t xg = gix + sx;
              if (xg < 0 || xg >= gnx) continue;
              nnz++;
            }
          }
        }
        rp[g] = off;
        off += nnz;
      }
      rp[A.totalNumberOfRows] = off;

      if (csr_dbg && csr_dbg[0] != '\0') {
        fprintf(stderr,
                "[rank %d] CSR prefix: totalRows=%lld nnz_total=%lld rp_last=%lld\n",
                laik_myid(A.world),
                (long long)A.totalNumberOfRows,
                (long long)off,
                (long long)rp[A.totalNumberOfRows]);
        fflush(stderr);
      }
    }

    if (csr_dbg && csr_dbg[0] != '\0') {
      int64_t* rp = 0; uint64_t rp_len = 0;
      laik_get_map_1d(A.rowD, 0, (void**)&rp, &rp_len);
      if (rp && rp_len > 0) {
        fprintf(stderr,
                "[rank %d] CSR rowD map0: rp_len=%llu rp0=%lld rplast=%lld\n",
                laik_myid(A.world),
                (unsigned long long)rp_len,
                (long long)rp[0],
                (long long)rp[rp_len - 1]);
        fflush(stderr);
      }
    }

    laik_switchto_partitioning(A.valD, rows_master, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(A.colD, rows_master, LAIK_DF_None, LAIK_RO_None);

    if (laik_myid(A.world) == 0) {
      double* val = 0; uint64_t val_len = 0;
      int64_t* col = 0; uint64_t col_len = 0;
      laik_get_map_1d(A.valD, 0, (void**)&val, &val_len);
      laik_get_map_1d(A.colD, 0, (void**)&col, &col_len);

      int64_t offv = 0;
      for (int64_t g = 0; g < A.totalNumberOfRows; ++g) {
        int64_t gix = g % gnx;
        int64_t giy = (g / gnx) % gny;
        int64_t giz = g / (gnx * gny);
        for (int sz = -1; sz <= 1; ++sz) {
          int64_t z = giz + sz;
          if (z < 0 || z >= gnz) continue;
          for (int sy = -1; sy <= 1; ++sy) {
            int64_t y = giy + sy;
            if (y < 0 || y >= gny) continue;
            for (int sx = -1; sx <= 1; ++sx) {
              int64_t xg = gix + sx;
              if (xg < 0 || xg >= gnx) continue;
              int64_t curcol = g + sz * gnx * gny + sy * gnx + sx;
              col[offv] = curcol;
              val[offv] = (curcol == g) ? 26.0 : -1.0;
              offv++;
            }
          }
        }
      }
    }

    if (laik_size(A.world) > 1) {
      laik_switchto_partitioning(A.rowD, A.rowP, LAIK_DF_Preserve, LAIK_RO_Single);
      laik_switchto_partitioning(A.valD, A.rowsP, LAIK_DF_Preserve, LAIK_RO_Single);
      laik_switchto_partitioning(A.colD, A.rowsP, LAIK_DF_Preserve, LAIK_RO_Single);
    }

    A.colDIsLocal = false;
  }
#endif // HPCG_NO_LAIK

  return;
}
