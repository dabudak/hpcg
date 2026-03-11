
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
 @file SparseMatrix.hpp

 HPCG data structures for the sparse matrix
 */

#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#include <vector>
#include <cassert>
#include <map>

#ifndef HPCG_NO_LAIK
#include "laik/hpcg_laik.hpp"

// forw. decl.
#ifdef REPARTITION
allocation_int_t map_l2a_A(const SparseMatrix &A, local_int_t localIndex);
void replaceMatrixValues(SparseMatrix &A);
#endif
void exit_hpcg_run(const char *msg, bool wait);
// forw. decl.
#endif

#include "Geometry.hpp"
#include "Vector.hpp"
#include "MGData.hpp"
#if __cplusplus < 201103L
// for C++03
#include <map>
    typedef std::map<global_int_t, local_int_t> GlobalToLocalMap;
#else
// for C++11 or greater
#include <unordered_map>
#include <cstdio>
using GlobalToLocalMap = std::unordered_map< global_int_t, local_int_t >;
#endif

    struct SparseMatrix_STRUCT
    {
      char *title;                                //!< name of the sparse matrix
      Geometry *geom;                             //!< geometry associated with this matrix
      global_int_t totalNumberOfRows;             //!< total number of matrix rows across all processes
      global_int_t totalNumberOfNonzeros;         //!< total number of matrix nonzeros across all processes
      local_int_t localNumberOfRows;              //!< number of rows local to this process
      local_int_t localNumberOfColumns;           //!< number of columns local to this process
      local_int_t localNumberOfNonzeros;          //!< number of nonzeros local to this process
      char *nonzerosInRow;                        //!< The number of nonzeros in a row will always be 27 or fewer
      global_int_t **mtxIndG;                     //!< matrix indices as global values
      local_int_t **mtxIndL;                      //!< matrix indices as local values
      double **matrixValues;                      //!< values of matrix entries
      double **matrixDiagonal;                    //!< values of matrix diagonal entries
      GlobalToLocalMap globalToLocalMap;          //!< global-to-local mapping
      std::vector<global_int_t> localToGlobalMap; //!< local-to-global mapping
      mutable bool isDotProductOptimized;
      mutable bool isSpmvOptimized;
      mutable bool isMgOptimized;
      mutable bool isWaxpbyOptimized;
      /*
       This is for storing optimized data structres created in OptimizeProblem and
       used inside optimized ComputeSPMV().
       */
      mutable struct SparseMatrix_STRUCT *Ac; // Coarse grid matrix
      mutable MGData *mgData;                 // Pointer to the coarse level data for this fine matrix
      void *optimizationData;                 // pointer that can be used to store implementation-specific data

#ifndef HPCG_NO_MPI
      local_int_t numberOfExternalValues; //!< number of entries that are external to this process
      int numberOfSendNeighbors;          //!< number of neighboring processes that will be send local data
      local_int_t totalToBeSent;          //!< total number of entries to be sent
      local_int_t *elementsToSend;        //!< elements to send to neighboring processes
      int *neighbors;                     //!< neighboring processes
      local_int_t *receiveLength;         //!< lenghts of messages received from neighboring processes
      local_int_t *sendLength;            //!< lenghts of messages sent to neighboring processes
      double *sendBuffer;                 //!< send buffer for non-blocking sends
  std::vector<global_int_t> externalLocalToGlobal; //!< external local index -> global index

#ifndef HPCG_NO_LAIK
      // ############### Data needed to create partitionings and Laik_Data container
  Laik_Instance *inst;
  Laik_Group *world;
      Laik_Space *space;
      Laik_Partitioning *ext;
      Laik_Partitioning *local;

  // LAIK CSR data for SpMV
  Laik_Space *rowSpacePrefix;
  Laik_Partitioning *rowP;
  Laik_Partitioning *rowsP;
  Laik_Partitioning *rowPrefixP;
  Laik_Data *rowD;
  Laik_Data *valD;
  Laik_Data *colD;
  mutable bool colDIsLocal;
  GlobalToLocalMap extLocalMap;
  mutable bool extMapBuilt;
  Laik_Data *matrixDiagonal_d;
  Laik_Blob *x_blob;
  Laik_Blob *b_blob;

#ifdef REPARTITION

      bool repartition_me; /* Tell the app, that a reseize should happen. We want to test it during the call to CG_REFin CG Reference Timing Phase */
      bool repartitioned;

      uint64_t *mapping_; // Need mapping due to the lex_layout. @see map_l2a_A()
      int offset_;

      // Special space for 2D arrays implemented as 1D array
      Laik_Space *space2d;

      // Partitionings for ressources below
      Laik_Partitioning *partitioning_1d;
      Laik_Partitioning *partitioning_2d;

      // Ressources of this matrix, which will be partitioned
      Laik_Data *nonzerosInRow_d;  //!< The number of nonzeros in a row will always be 27 or fewer
      Laik_Data *mtxIndG_d;        //!< matrix indices as global values
      Laik_Data *matrixValues_d;   //!< values of matrix entries

      /*
        This variable is only for x_l
        He will store the pointer to xexact_l
        We need to re-switch this vector as well
        But when reseizing, xexact_l is out of scope
        Quick solution is this here
      */
      Laik_Blob *ptr_to_xexact = 0;

#endif // REPARTITION
#endif // HPCG_NO_LAIK
#endif // HPCG_NO_MPI
};
typedef struct SparseMatrix_STRUCT SparseMatrix;

inline void InitializeSparseMatrix(SparseMatrix & A, Geometry * geom) {
  A.title = 0;
  A.geom = geom;
  A.totalNumberOfRows = 0;
  A.totalNumberOfNonzeros = 0;
  A.localNumberOfRows = 0;
  A.localNumberOfColumns = 0;
  A.localNumberOfNonzeros = 0;
  A.nonzerosInRow = 0;
  A.mtxIndG = 0;
  A.mtxIndL = 0;
  A.matrixValues = 0;
  A.matrixDiagonal = 0;

  // Optimization is ON by default. The code that switches it OFF is in the
  // functions that are meant to be optimized.
  A.isDotProductOptimized = true;
  A.isSpmvOptimized       = true;
  A.isMgOptimized      = true;
  A.isWaxpbyOptimized     = true;

#ifndef HPCG_NO_MPI
  A.numberOfExternalValues = 0;
  A.numberOfSendNeighbors = 0;
  A.totalToBeSent = 0;
  A.elementsToSend = 0;
  A.neighbors = 0;
  A.receiveLength = 0;
  A.sendLength = 0;
  A.sendBuffer = 0;
  A.externalLocalToGlobal.clear();

  #ifndef HPCG_NO_LAIK
  A.inst = 0;
  A.world = 0;
  A.space = 0;
  A.ext = 0;
  A.local = 0;
  A.rowSpacePrefix = 0;
  A.rowP = 0;
  A.rowsP = 0;
  A.rowPrefixP = 0;
  A.rowD = 0;
  A.valD = 0;
  A.colD = 0;
  A.colDIsLocal = false;
  A.extLocalMap.clear();
  A.extMapBuilt = false;
  A.matrixDiagonal_d = 0;
  A.x_blob = 0;
  A.b_blob = 0;

    #ifdef REPARTITION
      A.repartition_me = false;
      A.repartitioned = false;
      A.mapping_ = 0;
      A.offset_ = 0;
      A.space2d = 0;
      A.partitioning_1d = 0;
      A.partitioning_2d = 0;
      A.nonzerosInRow_d = 0;
      A.mtxIndG_d = 0;
      A.matrixValues_d = 0;
      A.ptr_to_xexact = 0;
    #endif // REPARTITION
  #endif // HPCG_NO_LAIK
#endif // HPCG_NO_MPI
  A.mgData = 0; // Fine-to-coarse grid transfer initially not defined.
  A.Ac =0;
  return;
}

inline void CopyMatrixDiagonal(SparseMatrix & A, Vector & diagonal) {

#ifndef HPCG_NO_LAIK
  if (A.matrixDiagonal_d) {
    double *matrixDiagonal;
    laik_get_map_1d(A.matrixDiagonal_d, 0, (void **)&matrixDiagonal, 0);
    double *dia_v = diagonal.values;
    assert(A.localNumberOfRows == diagonal.localLength);
    for (local_int_t i=0; i<A.localNumberOfRows; ++i) dia_v[i] = matrixDiagonal[i];
    return;
  }
#endif

    double ** curDiagA = A.matrixDiagonal;
    double * dv = diagonal.values;
    assert(A.localNumberOfRows==diagonal.localLength);
    for (local_int_t i=0; i<A.localNumberOfRows; ++i) dv[i] = *(curDiagA[i]);
  return;
}

inline void ReplaceMatrixDiagonal(SparseMatrix & A, Vector & diagonal) {

#ifndef HPCG_NO_LAIK
  if (A.matrixDiagonal_d) {
    double *matrixDiagonal;
    laik_get_map_1d(A.matrixDiagonal_d, 0, (void **)&matrixDiagonal, 0);
    double *dia_v = diagonal.values;
    assert(A.localNumberOfRows == diagonal.localLength);
    for (local_int_t i=0; i<A.localNumberOfRows; ++i) matrixDiagonal[i] = dia_v[i];
#ifdef REPARTITION
    replaceMatrixValues(A);
#endif
  }
#endif

    double ** curDiagA = A.matrixDiagonal;
    double * dv = diagonal.values;
    assert(A.localNumberOfRows==diagonal.localLength);
    for (local_int_t i=0; i<A.localNumberOfRows; ++i) *(curDiagA[i]) = dv[i];

#ifndef HPCG_NO_LAIK
  if (A.valD && A.colD && A.rowD && A.rowsP && A.rowP)
    {
      // Keep LAIK CSR values consistent with updated diagonal.
      laik_switchto_partitioning(A.rowD, A.rowP, LAIK_DF_Preserve, LAIK_RO_Single);
      laik_switchto_partitioning(A.valD, A.rowsP, LAIK_DF_Preserve, LAIK_RO_Single);
      laik_switchto_partitioning(A.colD, A.rowsP, LAIK_DF_Preserve, LAIK_RO_Single);

      int mapCount = laik_my_mapcount(A.rowsP);
      for (int mapNo = 0; mapNo < mapCount; ++mapNo)
      {
        double *val = 0; uint64_t val_len = 0;
        int64_t *col = 0; uint64_t col_len = 0;
        laik_get_map_1d(A.valD, mapNo, (void **)&val, &val_len);
        laik_get_map_1d(A.colD, mapNo, (void **)&col, &col_len);

        int mrCount = laik_my_maprangecount(A.rowsP, mapNo);
        int64_t range_val_offset = 0;
        for (int mr = 0; mr < mrCount; ++mr)
        {
          Laik_TaskRange *tr = laik_my_maprange(A.rowsP, mapNo, mr);
          const Laik_Range *s = laik_taskrange_get_range(tr);
          int64_t rf = s->from.i[0];
          int64_t rt = s->to.i[0];
          int64_t rowsHere = rt - rf;

          int row_map_no = -1;
          uint64_t row_lfrom_u = 0;
          Laik_Mapping *row_map = laik_global2maplocal_1d(A.rowD, rf, &row_map_no, &row_lfrom_u);
          if (!row_map) continue;

          int64_t *rp_base = 0; uint64_t rp_len = 0;
          laik_get_map_1d(A.rowD, row_map_no, (void **)&rp_base, &rp_len);
          int64_t *row_ptr = rp_base;
          int64_t base = row_ptr[0] - range_val_offset;

          for (int64_t i = 0; i < rowsHere; ++i)
          {
            global_int_t grow = rf + i;
            auto it = A.globalToLocalMap.find(grow);
            if (it == A.globalToLocalMap.end()) continue;
            double new_diag = dv[it->second];
            int64_t diag_col = (int64_t)grow;

            int64_t beg = row_ptr[i] - base;
            int64_t end = row_ptr[i + 1] - base;
            for (int64_t o = beg; o < end; ++o)
            {
              if (col[o] == diag_col)
              {
                val[o] = new_diag;
                break;
              }
            }
          }
          range_val_offset += (row_ptr[rowsHere] - row_ptr[0]);
        }
      }
    }
#endif
  return;
}

inline void DeleteMatrix(SparseMatrix & A) {

#ifndef HPCG_CONTIGUOUS_ARRAYS
if(A.matrixValues)
  for (local_int_t i = 0; i< A.localNumberOfRows; ++i)
    delete [] A.matrixValues[i];
#else
  delete [] A.matrixValues[0];
  delete [] A.mtxIndG[0];
  delete [] A.mtxIndL[0];
#endif
  if (A.title)                  delete [] A.title;
  if (A.nonzerosInRow)             delete [] A.nonzerosInRow;
  if (A.mtxIndG) delete [] A.mtxIndG;
  if (A.mtxIndL) delete [] A.mtxIndL;
  if (A.matrixValues) delete [] A.matrixValues;
  if (A.matrixDiagonal)           delete [] A.matrixDiagonal;

  if (A.elementsToSend)       delete [] A.elementsToSend;
  if (A.neighbors)              delete [] A.neighbors;
  if (A.receiveLength)            delete [] A.receiveLength;
  if (A.sendLength)            delete [] A.sendLength;
  if (A.sendBuffer)            delete [] A.sendBuffer;

  if (A.geom!=0) { DeleteGeometry(*A.geom); delete A.geom; A.geom = 0;}
  if (A.Ac!=0) { DeleteMatrix(*A.Ac); delete A.Ac; A.Ac = 0;} // Delete coarse matrix
  if (A.mgData!=0) { DeleteMGData(*A.mgData); delete A.mgData; A.mgData = 0;} // Delete MG data
  return;
}

#endif // SPARSEMATRIX_HPP
