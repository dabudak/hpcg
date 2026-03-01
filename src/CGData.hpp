
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
 @file CGData.hpp

 HPCG data structure
 */

#ifndef CGDATA_HPP
#define CGDATA_HPP

#ifndef HPCG_NO_LAIK
#include "laik/hpcg_laik.hpp"
#endif
#include "SparseMatrix.hpp"
#include "Vector.hpp"

struct CGData_STRUCT {

#ifndef HPCG_NO_LAIK
  Laik_Blob * r_blob;  //!< pointer to residual vector
  Laik_Blob * z_blob;  //!< pointer to preconditioned residual vector
  Laik_Blob * p_blob;  //!< pointer to direction vector
  Laik_Blob * Ap_blob; //!< pointer to Krylov vector
#else
  Vector r;  //!< pointer to residual vector
  Vector z;  //!< pointer to preconditioned residual vector
  Vector p;  //!< pointer to direction vector
  Vector Ap; //!< pointer to Krylov vector
#endif
};
typedef struct CGData_STRUCT CGData;

inline void InitializeSparseCGData(SparseMatrix & A, CGData & data) {

#ifndef HPCG_NO_LAIK
  std::string name{""};
  name = "CG_Data_r";
  data.r_blob = init_blob(A, false, name.data());
  name = "CG_Data_z";
  data.z_blob = init_blob(A, true, name.data());
  name = "CG_Data_p";
  data.p_blob = init_blob(A, true, name.data());
  name = "CG_Data_Ap";
  data.Ap_blob = init_blob(A, false, name.data());
#else
  local_int_t nrow = A.localNumberOfRows;
  local_int_t ncol = A.localNumberOfColumns;

  InitializeVector(data.r, nrow);
  InitializeVector(data.z, ncol);
  InitializeVector(data.p, ncol);
  InitializeVector(data.Ap, nrow);
#endif

  return;
}

inline void DeleteCGData(CGData & data) {

#ifndef HPCG_NO_LAIK
  DeleteLaikVector(data.r_blob);
  DeleteLaikVector(data.Ap_blob);
  DeleteLaikVector(data.z_blob);
  DeleteLaikVector(data.p_blob);

#else
  DeleteVector (data.r);
  DeleteVector (data.z);
  DeleteVector (data.p);
  DeleteVector (data.Ap);
#endif
  return;
}

#endif // CGDATA_HPP
