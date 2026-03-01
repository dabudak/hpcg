
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

#ifndef CG_HPP
#define CG_HPP

#include "laik/hpcg_laik.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"

#ifndef HPCG_NO_LAIK
int CG_laik(SparseMatrix &A, CGData &data, Laik_Blob *b, Laik_Blob *x,
            const int max_iter, const double tolerance, int &niters, double &normr, double &normr0,
            double *times, bool doPreconditioning);
#else
int CG(const SparseMatrix & A, CGData & data, const Vector & b, Vector & x,
    const int max_iter, const double tolerance, int & niters, double & normr,  double & normr0,
    double * times, bool doPreconditioning);
#endif

#endif  // CG_HPP
