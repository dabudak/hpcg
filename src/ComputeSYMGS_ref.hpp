
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

#ifndef COMPUTESYMGS_REF_HPP
#define COMPUTESYMGS_REF_HPP

#ifndef HPCG_NO_LAIK
#include "laik/hpcg_laik.hpp"
#endif
#include "SparseMatrix.hpp"
#include "Vector.hpp"

#ifndef HPCG_NO_LAIK
int ComputeSYMGS_laik_ref(const SparseMatrix &A, const Laik_Blob *r, Laik_Blob *x);
#endif
int ComputeSYMGS_ref(const SparseMatrix &A, const Vector &r, Vector &x);

#endif // COMPUTESYMGS_REF_HPP
