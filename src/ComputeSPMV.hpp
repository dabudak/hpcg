
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

#ifndef COMPUTESPMV_HPP
#define COMPUTESPMV_HPP

#ifndef HPCG_NO_LAIK
#include "laik/hpcg_laik.hpp"
#endif
#include "Vector.hpp"
#include "SparseMatrix.hpp"

#ifndef HPCG_NO_LAIK
int ComputeSPMV_laik(const SparseMatrix &A, Laik_Blob *x_blob, Laik_Blob *y_blob);
#endif
int ComputeSPMV(const SparseMatrix &A, Vector &x, Vector &y);

#endif  // COMPUTESPMV_HPP
