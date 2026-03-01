
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
 @file CG.cpp

 HPCG routine
 */

#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifndef HPCG_NO_LAIK
#include "laik/hpcg_laik.hpp"
#endif
#include "hpcg.hpp"
#include "CG.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeMG.hpp"
#include "ComputeDotProduct.hpp"
#include "ComputeWAXPBY.hpp"

// Use TICK and TOCK to time a code section in MATLAB-like fashion
#define TICK() t0 = mytimer()       //!< record current time in 't0'
#define TOCK(t) t += mytimer() - t0 //!< store time difference in 't' using time in 't0'


#ifndef HPCG_NO_LAIK
static void cg_vec_stats(const char* label, const Laik_Blob* v)
{
  double* base = 0;
  uint64_t len = 0;
  laik_get_map_1d(v->values, 0, (void**)&base, &len);
  double sumabs = 0.0;
  double maxabs = 0.0;
  for (uint64_t i = 0; i < len; ++i) {
    double a = std::fabs(base[i]);
    sumabs += a;
    if (a > maxabs) maxabs = a;
  }
  std::fprintf(stderr, "[rank %d] CG stats %s len=%llu sumabs=%g maxabs=%g\n",
               v && v->values ? laik_myid(world) : -1,
               label, (unsigned long long)len, sumabs, maxabs);
}
int CG_laik(SparseMatrix &A, CGData &data, Laik_Blob *b, Laik_Blob *x,
       const int max_iter, const double tolerance, int &niters, double &normr, double &normr0,
       double *times, bool doPreconditioning)
{

  double t_begin = mytimer(); // Start timing right away
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;

  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0;
  local_int_t nrow = A.localNumberOfRows;
  Laik_Blob * r = data.r_blob; // Residual vector
  Laik_Blob * z = data.z_blob;  // Preconditioned residual vector
  Laik_Blob * p = data.p_blob;   // Direction vector (in MPI mode ncol>=nrow)
  Laik_Blob * Ap = data.Ap_blob;

  if (!doPreconditioning && A.geom->rank == 0)
    HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

#ifdef HPCG_DEBUG
  int print_freq = 1;
  if (print_freq > 50)
    print_freq = 50;
  if (print_freq < 1)
    print_freq = 1;
#endif
  // copy x to p for sparse MV operation
  CopyLaikVectorToLaikVector(x, p);
  if (A.local) {
    laik_switchto_partitioning(p->values, A.local, LAIK_DF_Preserve, LAIK_RO_Single);
  }
  TICK(); ComputeSPMV_laik(A, p, Ap); TOCK(t3); // Ap = A*p
  if (A.local) {
    laik_switchto_partitioning(b->values, A.local, LAIK_DF_Preserve, LAIK_RO_Single);
    laik_switchto_partitioning(Ap->values, A.local, LAIK_DF_Preserve, LAIK_RO_Single);
    laik_switchto_partitioning(r->values, A.local, LAIK_DF_Preserve, LAIK_RO_Single);
  }
  TICK(); ComputeWAXPBY_laik(nrow, 1.0, b, -1.0, Ap, r, A.isWaxpbyOptimized); TOCK(t2); // r = b - Ax (x stored in p)
  if (A.local) {
    laik_switchto_partitioning(r->values, A.local, LAIK_DF_Preserve, LAIK_RO_Single);
  }
  TICK(); ComputeDotProduct_laik(nrow, r, r, normr, t4, A.isDotProductOptimized); TOCK(t1);
  normr = sqrt(normr);
#ifdef HPCG_DEBUG
  if (A.geom->rank == 0) HPCG_fout << "Initial Residual = " << normr << std::endl;
#endif

  // Record initial residual for convergence testing
  normr0 = normr;

  // Start iterations
  // Convergence check accepts an error of no more than 6 significant digits of tolerance
  for (int k = 1; k <= max_iter && normr / normr0 > tolerance * (1.0 + 1.0e-6); k++)
  {
    const char* cg_state = std::getenv("HPCG_LAIK_CG_STATE");
    static int cg_state_reported = 0;
    TICK();
    if (doPreconditioning)
      ComputeMG_laik(A, r, z); // Apply preconditioner
    else
      CopyLaikVectorToLaikVector(r, z); // copy r to z (no preconditioning)
    // Ensure z/r are in local partitioning before dot products.
    if (A.local) {
      laik_switchto_partitioning(z->values, A.local, LAIK_DF_Preserve, LAIK_RO_Single);
      laik_switchto_partitioning(r->values, A.local, LAIK_DF_Preserve, LAIK_RO_Single);
    }
    TOCK(t5);           // Preconditioner apply time

    if (cg_state && cg_state[0] != '\0' && !cg_state_reported) {
      cg_vec_stats("r(after MG)", r);
      cg_vec_stats("z(after MG)", z);
    }


    if (k == 1)
    {
      TICK();
      if (A.local) {
        laik_switchto_partitioning(z->values, A.local, LAIK_DF_Preserve, LAIK_RO_Single);
        laik_switchto_partitioning(p->values, A.local, LAIK_DF_Preserve, LAIK_RO_Single);
      }
      ComputeWAXPBY_laik(nrow, 1.0, z, 0.0, z, p, A.isWaxpbyOptimized);
      TOCK(t2); // Copy Mr to p



      TICK();
      ComputeDotProduct_laik(nrow, r, z, rtz, t4, A.isDotProductOptimized);

      TOCK(t1); // rtz = r'*z
    }
    else
    {
      oldrtz = rtz;
      TICK();
      ComputeDotProduct_laik(nrow, r, z, rtz, t4, A.isDotProductOptimized);
      TOCK(t1); // rtz = r'*z
      beta = rtz / oldrtz;
      TICK();
      if (A.local) {
        laik_switchto_partitioning(z->values, A.local, LAIK_DF_Preserve, LAIK_RO_Single);
        laik_switchto_partitioning(p->values, A.local, LAIK_DF_Preserve, LAIK_RO_Single);
      }
      ComputeWAXPBY_laik(nrow, 1.0, z, beta, p, p, A.isWaxpbyOptimized);
      TOCK(t2); // p = beta*p + z
    }

    TICK();
    ComputeSPMV_laik(A, p, Ap);
    TOCK(t3); // Ap = A*p
    // Ensure p/Ap and x/r are local before dot product and updates.
    if (A.local) {
      laik_switchto_partitioning(p->values, A.local, LAIK_DF_Preserve, LAIK_RO_Single);
      laik_switchto_partitioning(Ap->values, A.local, LAIK_DF_Preserve, LAIK_RO_Single);
      laik_switchto_partitioning(x->values, A.local, LAIK_DF_Preserve, LAIK_RO_Single);
      laik_switchto_partitioning(r->values, A.local, LAIK_DF_Preserve, LAIK_RO_Single);
    }
    TICK();
    ComputeDotProduct_laik(nrow, p, Ap, pAp, t4, A.isDotProductOptimized);
    TOCK(t1); // alpha = p'*Ap

    if (cg_state && cg_state[0] != '\0' && !cg_state_reported) {
      cg_vec_stats("p", p);
      cg_vec_stats("Ap", Ap);
      cg_state_reported = 1;
    }
    const char* nan_check = std::getenv("HPCG_LAIK_NAN_CHECK");
    static int cg_nan_reported = 0;
    if (nan_check && nan_check[0] != '\0' && !cg_nan_reported) {
      if (!std::isfinite(rtz) || !std::isfinite(pAp) || pAp == 0.0) {
        std::fprintf(stderr, "[rank %d] CG nan check: k=%d rtz=%g pAp=%g\n",
                     A.geom ? A.geom->rank : -1, k, rtz, pAp);
        cg_nan_reported = 1;
      }
    }
    alpha = rtz / pAp;
    TICK();
    ComputeWAXPBY_laik(nrow, 1.0, x, alpha, p, x, A.isWaxpbyOptimized); // x = x + alpha*p
    ComputeWAXPBY_laik(nrow, 1.0, r, -alpha, Ap, r, A.isWaxpbyOptimized);
    TOCK(t2); // r = r - alpha*Ap
    TICK();
    ComputeDotProduct_laik(nrow, r, r, normr, t4, A.isDotProductOptimized);
    TOCK(t1);
    normr = sqrt(normr);
    if (nan_check && nan_check[0] != '\0' && !cg_nan_reported) {
      if (!std::isfinite(normr) || !std::isfinite(alpha) || !std::isfinite(beta)) {
        std::fprintf(stderr, "[rank %d] CG nan check: k=%d alpha=%g beta=%g normr=%g\n",
                     A.geom ? A.geom->rank : -1, k, alpha, beta, normr);
        cg_nan_reported = 1;
      }
    }
#ifdef HPCG_DEBUG
    if (A.geom->rank == 0 && (k % print_freq == 0 || k == max_iter))
      HPCG_fout << "Iteration = " << k << "   Scaled Residual = " << normr / normr0 << std::endl;
#endif
    niters = k;
  }

  // Store times
  times[1] += t1; // dot-product time
  times[2] += t2; // WAXPBY time
  times[3] += t3; // SPMV time
  times[4] += t4; // AllReduce time
  times[5] += t5; // preconditioner apply time
  times[0] += mytimer() - t_begin; // Total time. All done...

  return 0;
}
#else
int CG(const SparseMatrix &A, CGData &data, const Vector &b, Vector &x,
       const int max_iter, const double tolerance, int &niters, double &normr, double &normr0,
       double *times, bool doPreconditioning)
{

  double t_begin = mytimer(); // Start timing right away
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;

  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0;
  local_int_t nrow = A.localNumberOfRows;
  Vector &r = data.r; // Residual vector
  Vector &z = data.z; // Preconditioned residual vector
  Vector &p = data.p; // Direction vector (in MPI mode ncol>=nrow)
  Vector &Ap = data.Ap;

  if (!doPreconditioning && A.geom->rank == 0)
    HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

#ifdef HPCG_DEBUG
  int print_freq = 1;
  if (print_freq > 50)
    print_freq = 50;
  if (print_freq < 1)
    print_freq = 1;
#endif
  // p is of length ncols, copy x to p for sparse MV operation
  CopyVector(x, p);
  TICK();
  ComputeSPMV(A, p, Ap);
  TOCK(t3); // Ap = A*p
  TICK();
  ComputeWAXPBY(nrow, 1.0, b, -1.0, Ap, r, A.isWaxpbyOptimized);
  TOCK(t2); // r = b - Ax (x stored in p)
  TICK();
  ComputeDotProduct(nrow, r, r, normr, t4, A.isDotProductOptimized);
  TOCK(t1);
  normr = sqrt(normr);
#ifdef HPCG_DEBUG
  if (A.geom->rank == 0)
    HPCG_fout << "Initial Residual = " << normr << std::endl;
#endif

  // Record initial residual for convergence testing
  normr0 = normr;

  // Start iterations
  // Convergence check accepts an error of no more than 6 significant digits of tolerance
  for (int k = 1; k <= max_iter && normr / normr0 > tolerance * (1.0 + 1.0e-6); k++)
  {
    TICK();
    if (doPreconditioning)
      ComputeMG(A, r, z); // Apply preconditioner
    else
      CopyVector(r, z); // copy r to z (no preconditioning)
    TOCK(t5);           // Preconditioner apply time

    if (k == 1)
    {
      TICK();
      ComputeWAXPBY(nrow, 1.0, z, 0.0, z, p, A.isWaxpbyOptimized);
      TOCK(t2); // Copy Mr to p

      TICK();
      ComputeDotProduct(nrow, r, z, rtz, t4, A.isDotProductOptimized);
      TOCK(t1); // rtz = r'*z
    }
    else
    {
      oldrtz = rtz;
      TICK();
      ComputeDotProduct(nrow, r, z, rtz, t4, A.isDotProductOptimized);
      TOCK(t1); // rtz = r'*z
      beta = rtz / oldrtz;
      TICK();
      ComputeWAXPBY(nrow, 1.0, z, beta, p, p, A.isWaxpbyOptimized);
      TOCK(t2); // p = beta*p + z
    }

    TICK();
    ComputeSPMV(A, p, Ap);
    TOCK(t3); // Ap = A*p
    TICK();
    ComputeDotProduct(nrow, p, Ap, pAp, t4, A.isDotProductOptimized);
    TOCK(t1); // alpha = p'*Ap
    alpha = rtz / pAp;
    TICK();
    ComputeWAXPBY(nrow, 1.0, x, alpha, p, x, A.isWaxpbyOptimized); // x = x + alpha*p
    ComputeWAXPBY(nrow, 1.0, r, -alpha, Ap, r, A.isWaxpbyOptimized);
    TOCK(t2); // r = r - alpha*Ap
    TICK();
    ComputeDotProduct(nrow, r, r, normr, t4, A.isDotProductOptimized);
    TOCK(t1);
    normr = sqrt(normr);
#ifdef HPCG_DEBUG
    if (A.geom->rank == 0 && (k % print_freq == 0 || k == max_iter))
      HPCG_fout << "Iteration = " << k << "   Scaled Residual = " << normr / normr0 << std::endl;
#endif
    niters = k;
  }

  // Store times
  times[1] += t1; // dot-product time
  times[2] += t2; // WAXPBY time
  times[3] += t3; // SPMV time
  times[4] += t4; // AllReduce time
  times[5] += t5; // preconditioner apply time
  times[0] += mytimer() - t_begin; // Total time. All done...

  return 0;
}
#endif
