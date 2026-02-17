
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

#ifndef COMPUTESPMV_REF_HPP
#define COMPUTESPMV_REF_HPP
#include "Vector.hpp"
#include "SparseMatrix.hpp"
#include <vector>

int ComputeSPMV_ref( const SparseMatrix & A, Vector  & x, Vector & y);
int ComputeSPMV_ref_laik(const SparseMatrix& A, std::vector<double>& y);

#endif  // COMPUTESPMV_REF_HPP
