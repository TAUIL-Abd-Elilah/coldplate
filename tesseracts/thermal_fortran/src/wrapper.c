/* Copyright 2026 Coldplate contributors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Enzyme entry points for the Fortran thermal residual.
 *
 * After the Enzyme LLVM pass runs, the __enzyme_autodiff and __enzyme_fwddiff
 * calls below are replaced by compiler-generated derivative code. No adjoint
 * is written by hand and no AD library is linked: the derivative of this
 * component is produced by a compiler pass over the Fortran, which is a
 * genuinely different mechanism from the JAX block it can be swapped for and
 * from the C++ block it composes with.
 *
 * Exported ABI:
 *   th_forward  R              = residual(T, u, v, k)
 *   th_jvp      dR             = dR/d(T,u,v,k) . (dT,du,dv,dk)
 *   th_vjp      (Tb,ub,vb,kb) += (dR/d(T,u,v,k))^T . Rb
 *
 * The pipeline uses th_jvp with structured seed vectors to recover the sparse
 * operator dR/dT exactly (see tesseract_api.py), and th_vjp for the parameter
 * derivatives that the implicit-function-theorem step needs.
 */

#if defined(_WIN32)
#define TH_API __declspec(dllexport)
#else
#define TH_API __attribute__((visibility("default")))
#endif

/* Enzyme annotation sentinels, resolved by the Enzyme LLVM pass. */
int enzyme_dup;
int enzyme_const;

/* Fortran ABI: every argument by pointer. */
extern void thermal_residual(int* Nx, int* Ny, double* T, double* u, double* v,
                             double* k, double* q_chip, double* chip_frac,
                             double* R);

extern void __enzyme_autodiff(void*, ...);
extern void __enzyme_fwddiff(void*, ...);

#ifdef __cplusplus
extern "C" {
#endif

TH_API void th_forward(int Nx, int Ny, const double* T, const double* u,
                       const double* v, const double* k, double q_chip,
                       double chip_frac, double* R) {
  int nx = Nx, ny = Ny;
  double q = q_chip, cf = chip_frac;
  thermal_residual(&nx, &ny, (double*)T, (double*)u, (double*)v, (double*)k,
                   &q, &cf, R);
}

/* Forward mode. Integer and scalar-parameter arguments are passed as dup with
 * zero shadows rather than const, matching the calling convention that the
 * upstream Enzyme example uses for forward mode. */
TH_API void th_jvp(int Nx, int Ny, const double* T, const double* dT,
                   const double* u, const double* du, const double* v,
                   const double* dv, const double* k, const double* dk,
                   double q_chip, double chip_frac, double* R, double* dR) {
  int nx = Nx, ny = Ny, dnx = 0, dny = 0;
  double q = q_chip, cf = chip_frac, dq = 0.0, dcf = 0.0;

  __enzyme_fwddiff((void*)thermal_residual,
                   enzyme_dup, &nx, &dnx,
                   enzyme_dup, &ny, &dny,
                   enzyme_dup, (double*)T, (double*)dT,
                   enzyme_dup, (double*)u, (double*)du,
                   enzyme_dup, (double*)v, (double*)dv,
                   enzyme_dup, (double*)k, (double*)dk,
                   enzyme_dup, &q, &dq,
                   enzyme_dup, &cf, &dcf,
                   enzyme_dup, R, dR);
}

/* Reverse mode. Rb carries the incoming cotangent; Enzyme accumulates into
 * Tb, ub, vb and kb, which the caller must zero beforehand. */
TH_API void th_vjp(int Nx, int Ny, const double* T, double* Tb, const double* u,
                   double* ub, const double* v, double* vb, const double* k,
                   double* kb, double q_chip, double chip_frac, double* R,
                   double* Rb) {
  int nx = Nx, ny = Ny;
  double q = q_chip, cf = chip_frac, dq = 0.0, dcf = 0.0;

  __enzyme_autodiff((void*)thermal_residual,
                    enzyme_const, &nx,
                    enzyme_const, &ny,
                    enzyme_dup, (double*)T, Tb,
                    enzyme_dup, (double*)u, ub,
                    enzyme_dup, (double*)v, vb,
                    enzyme_dup, (double*)k, kb,
                    enzyme_dup, &q, &dq,
                    enzyme_dup, &cf, &dcf,
                    enzyme_dup, R, Rb);
}

#ifdef __cplusplus
}
#endif
