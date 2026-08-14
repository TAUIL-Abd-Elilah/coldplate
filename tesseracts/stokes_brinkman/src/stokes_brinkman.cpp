// Copyright 2026 Coldplate contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Steady Stokes-Brinkman flow with Boussinesq buoyancy, on a staggered MAC
// grid, with a HAND-DERIVED DISCRETE ADJOINT.
//
// This component deliberately does not use an AD tool. The system is linear in
// the unknown w = (u, v, p):
//
//     A(alpha) w = b(T)
//
// so its derivatives are available in closed form and cost one extra triangular
// solve against a factorisation we already own:
//
//   forward   w    = A^{-1} b
//   JVP       dw   = A^{-1} ( db/dT dT  -  (dA/dalpha dalpha) w )
//   VJP       lam  = A^{-T} wbar,  then scatter lam against dA/dalpha and db/dT
//
// That is a fundamentally different derivative strategy from the JAX block it
// is composed with, which is the point: the two components disagree about how
// they compute and still compose into one differentiable function.
//
// Index layout (must match the JAX reference exactly):
//   u interior : (Ny, Nx-1)  faces i = 1..Nx-1   flat  j*(Nx-1) + (i-1)
//   v interior : (Ny-1, Nx)  faces j = 1..Ny-1   flat  nu + (j-1)*Nx + i
//   p          : (Ny, Nx)                        flat  nu + nv + j*Nx + i
// Exported arrays are full-size: u is (Ny, Nx+1), v is (Ny+1, Nx), with the
// no-slip wall values written as exact zeros.

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

// Exported either way: MSVC needs dllexport, and the Linux build hides
// everything else so the .so surface is exactly this C ABI.
#if defined(_WIN32)
#define SB_API __declspec(dllexport)
#else
#define SB_API __attribute__((visibility("default")))
#endif

using Scalar = double;
using SpMat = Eigen::SparseMatrix<Scalar>;
using Trip = Eigen::Triplet<Scalar>;
using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using LU = Eigen::SparseLU<SpMat, Eigen::COLAMDOrdering<int>>;

namespace {

struct Solver {
  int Nx = 0, Ny = 0;
  double Pr = 0.0, Ra = 0.0, h = 0.0;
  // Weight on the convective acceleration. 0 recovers the Stokes-Brinkman
  // system exactly -- the operator is then independent of w and everything
  // below collapses to the original single-solve path.
  double inertia = 0.0;
  int nu = 0, nv = 0, np = 0, n = 0;
  std::vector<double> alpha;  // (Ny*Nx) cell-centred Brinkman coefficient

  SpMat A;                      // linear (Stokes) part of the operator
  LU lu;                        // factorisation of A
  std::unique_ptr<LU> luT;      // factorisation of A^T, built lazily for VJPs
  // With inertia the operator that must be inverted for derivatives is the
  // Jacobian at the converged state, J = A + dN/dw, not A. These hold its
  // factorisations; they stay empty in the Stokes case.
  SpMat J;
  std::unique_ptr<LU> luJ, luJT;
  Vec w_star;                   // converged state, needed to build dN/dw
  bool ok = false;

  inline int iu(int j, int i) const { return j * (Nx - 1) + (i - 1); }
  inline int iv(int j, int i) const { return nu + (j - 1) * Nx + i; }
  inline int ip(int j, int i) const { return nu + nv + j * Nx + i; }
  inline int ic(int j, int i) const { return j * Nx + i; }        // cell centre
  inline int iuf(int j, int i) const { return j * (Nx + 1) + i; } // full u array
  inline int ivf(int j, int i) const { return j * Nx + i; }       // full v array
};

// alpha interpolated from cell centres onto the velocity faces.
inline double alpha_u(const Solver& s, int j, int i) {
  return 0.5 * (s.alpha[s.ic(j, i - 1)] + s.alpha[s.ic(j, i)]);
}
inline double alpha_v(const Solver& s, int j, int i) {
  return 0.5 * (s.alpha[s.ic(j - 1, i)] + s.alpha[s.ic(j, i)]);
}

void assemble(Solver& s) {
  const int Nx = s.Nx, Ny = s.Ny;
  const double Pr = s.Pr, h = s.h;
  const double ih2 = Pr / (h * h);
  std::vector<Trip> trips;
  trips.reserve(9 * s.n);

  // ---- x-momentum: -Pr lap(u) + dp/dx + Pr alpha u = 0 ----
  for (int j = 0; j < Ny; ++j) {
    for (int i = 1; i <= Nx - 1; ++i) {
      const int r = s.iu(j, i);
      double diag = 4.0 * ih2 + Pr * alpha_u(s, j, i);
      if (i - 1 >= 1) trips.emplace_back(r, s.iu(j, i - 1), -ih2);
      if (i + 1 <= Nx - 1) trips.emplace_back(r, s.iu(j, i + 1), -ih2);
      // No-slip on the bottom/top walls via the reflected ghost u_ghost = -u,
      // which folds an extra +Pr/h^2 onto the diagonal instead of a neighbour.
      if (j > 0) trips.emplace_back(r, s.iu(j - 1, i), -ih2);
      else diag += ih2;
      if (j < Ny - 1) trips.emplace_back(r, s.iu(j + 1, i), -ih2);
      else diag += ih2;
      trips.emplace_back(r, r, diag);
      trips.emplace_back(r, s.ip(j, i), 1.0 / h);
      trips.emplace_back(r, s.ip(j, i - 1), -1.0 / h);
    }
  }

  // ---- y-momentum: -Pr lap(v) + dp/dy + Pr alpha v = Ra Pr T ----
  for (int j = 1; j <= Ny - 1; ++j) {
    for (int i = 0; i < Nx; ++i) {
      const int r = s.iv(j, i);
      double diag = 4.0 * ih2 + Pr * alpha_v(s, j, i);
      if (j - 1 >= 1) trips.emplace_back(r, s.iv(j - 1, i), -ih2);
      if (j + 1 <= Ny - 1) trips.emplace_back(r, s.iv(j + 1, i), -ih2);
      if (i > 0) trips.emplace_back(r, s.iv(j, i - 1), -ih2);
      else diag += ih2;
      if (i < Nx - 1) trips.emplace_back(r, s.iv(j, i + 1), -ih2);
      else diag += ih2;
      trips.emplace_back(r, r, diag);
      trips.emplace_back(r, s.ip(j, i), 1.0 / h);
      trips.emplace_back(r, s.ip(j - 1, i), -1.0 / h);
    }
  }

  // ---- continuity: -div(u) = 0, with p[0,0] pinned to fix the null space ----
  for (int j = 0; j < Ny; ++j) {
    for (int i = 0; i < Nx; ++i) {
      const int r = s.ip(j, i);
      if (j == 0 && i == 0) {
        trips.emplace_back(r, s.ip(0, 0), 1.0);
        continue;
      }
      if (i + 1 <= Nx - 1) trips.emplace_back(r, s.iu(j, i + 1), -1.0 / h);
      if (i >= 1) trips.emplace_back(r, s.iu(j, i), 1.0 / h);
      if (j + 1 <= Ny - 1) trips.emplace_back(r, s.iv(j + 1, i), -1.0 / h);
      if (j >= 1) trips.emplace_back(r, s.iv(j, i), 1.0 / h);
    }
  }

  s.A.resize(s.n, s.n);
  s.A.setFromTriplets(trips.begin(), trips.end());
  s.A.makeCompressed();
}

// ---------------------------------------------------------------------------
// Convective acceleration (u.grad)u, and its exact Jacobian.
//
// The term is BILINEAR in w, which is what makes a hand-derived adjoint still
// practical once the system stops being linear: the Jacobian entries are just
// the two factors read off in turn, and the alpha/T dependence is untouched --
// convection involves neither. So the whole extension is "invert J = A + dN/dw
// instead of A", and every scatter in the JVP and VJP below is unchanged.
//
// Discretisation matches prototype/reference_jax.py exactly, including the
// reflected ghosts (u_ghost = -u) used for the no-slip walls, so the two can be
// compared to machine precision.
// ---------------------------------------------------------------------------

// Velocity components read out of the flat unknown vector. Faces that lie on a
// wall are not unknowns and are exactly zero.
inline double uval(const Solver& s, const Vec& w, int j, int i) {
  if (i <= 0 || i >= s.Nx || j < 0 || j >= s.Ny) return 0.0;
  return w[s.iu(j, i)];
}
inline double vval(const Solver& s, const Vec& w, int j, int i) {
  if (j <= 0 || j >= s.Ny || i < 0 || i >= s.Nx) return 0.0;
  return w[s.iv(j, i)];
}

// v averaged onto a u-face, and u averaged onto a v-face.
inline double v_at_u(const Solver& s, const Vec& w, int j, int i) {
  return 0.25 * (vval(s, w, j, i - 1) + vval(s, w, j, i) +
                 vval(s, w, j + 1, i - 1) + vval(s, w, j + 1, i));
}
inline double u_at_v(const Solver& s, const Vec& w, int j, int i) {
  return 0.25 * (uval(s, w, j - 1, i) + uval(s, w, j - 1, i + 1) +
                 uval(s, w, j, i) + uval(s, w, j, i + 1));
}

// Derivatives at a face, using the same reflected ghosts as the Laplacian.
inline double dudy(const Solver& s, const Vec& w, int j, int i) {
  const double ym = (j > 0) ? uval(s, w, j - 1, i) : -uval(s, w, 0, i);
  const double yp = (j < s.Ny - 1) ? uval(s, w, j + 1, i)
                                   : -uval(s, w, s.Ny - 1, i);
  return (yp - ym) / (2.0 * s.h);
}
inline double dvdx(const Solver& s, const Vec& w, int j, int i) {
  const double xm = (i > 0) ? vval(s, w, j, i - 1) : -vval(s, w, j, 0);
  const double xp = (i < s.Nx - 1) ? vval(s, w, j, i + 1)
                                   : -vval(s, w, j, s.Nx - 1);
  return (xp - xm) / (2.0 * s.h);
}

// N(w): the convective contribution to the momentum rows.
Vec convect(const Solver& s, const Vec& w) {
  Vec N = Vec::Zero(s.n);
  if (s.inertia == 0.0) return N;
  const double i2h = 1.0 / (2.0 * s.h);

  for (int j = 0; j < s.Ny; ++j)
    for (int i = 1; i <= s.Nx - 1; ++i) {
      const double ux = (uval(s, w, j, i + 1) - uval(s, w, j, i - 1)) * i2h;
      N[s.iu(j, i)] = s.inertia * (uval(s, w, j, i) * ux +
                                   v_at_u(s, w, j, i) * dudy(s, w, j, i));
    }
  for (int j = 1; j <= s.Ny - 1; ++j)
    for (int i = 0; i < s.Nx; ++i) {
      const double vy = (vval(s, w, j + 1, i) - vval(s, w, j - 1, i)) * i2h;
      N[s.iv(j, i)] = s.inertia * (u_at_v(s, w, j, i) * dvdx(s, w, j, i) +
                                   vval(s, w, j, i) * vy);
    }
  return N;
}

// dN/dw, appended to an existing triplet list. Exact, not an approximation:
// every entry is one of the two factors of the bilinear form.
void convection_jacobian(const Solver& s, const Vec& w, std::vector<Trip>& t) {
  if (s.inertia == 0.0) return;
  const double c = s.inertia / (2.0 * s.h);
  const auto u_is_unknown = [&](int j, int i) {
    return i >= 1 && i <= s.Nx - 1 && j >= 0 && j <= s.Ny - 1;
  };
  const auto v_is_unknown = [&](int j, int i) {
    return j >= 1 && j <= s.Ny - 1 && i >= 0 && i <= s.Nx - 1;
  };

  // ---- x-momentum rows: N_u = u du/dx + v_bar du/dy ----
  for (int j = 0; j < s.Ny; ++j)
    for (int i = 1; i <= s.Nx - 1; ++i) {
      const int r = s.iu(j, i);
      const double U = uval(s, w, j, i);
      const double V = v_at_u(s, w, j, i);
      const double ux = (uval(s, w, j, i + 1) - uval(s, w, j, i - 1)) / (2 * s.h);
      const double uy = dudy(s, w, j, i);

      // d/du(j,i): the du/dx factor, plus the ghost's own contribution to du/dy
      double diag = s.inertia * ux;
      if (j == 0) diag += V * c;                 // u_ghost = -u(0,i)
      if (j == s.Ny - 1) diag -= V * c;          // u_ghost = -u(Ny-1,i)
      t.emplace_back(r, r, diag);

      if (u_is_unknown(j, i - 1)) t.emplace_back(r, s.iu(j, i - 1), -c * U);
      if (u_is_unknown(j, i + 1)) t.emplace_back(r, s.iu(j, i + 1), +c * U);
      if (u_is_unknown(j - 1, i)) t.emplace_back(r, s.iu(j - 1, i), -c * V);
      if (u_is_unknown(j + 1, i)) t.emplace_back(r, s.iu(j + 1, i), +c * V);

      // d/dv through the four-point average feeding v_bar
      const double cv = 0.25 * s.inertia * uy;
      if (v_is_unknown(j, i - 1)) t.emplace_back(r, s.iv(j, i - 1), cv);
      if (v_is_unknown(j, i)) t.emplace_back(r, s.iv(j, i), cv);
      if (v_is_unknown(j + 1, i - 1)) t.emplace_back(r, s.iv(j + 1, i - 1), cv);
      if (v_is_unknown(j + 1, i)) t.emplace_back(r, s.iv(j + 1, i), cv);
    }

  // ---- y-momentum rows: N_v = u_bar dv/dx + v dv/dy ----
  for (int j = 1; j <= s.Ny - 1; ++j)
    for (int i = 0; i < s.Nx; ++i) {
      const int r = s.iv(j, i);
      const double V = vval(s, w, j, i);
      const double U = u_at_v(s, w, j, i);
      const double vy = (vval(s, w, j + 1, i) - vval(s, w, j - 1, i)) / (2 * s.h);
      const double vx = dvdx(s, w, j, i);

      double diag = s.inertia * vy;
      if (i == 0) diag += U * c;                 // v_ghost = -v(j,0)
      if (i == s.Nx - 1) diag -= U * c;          // v_ghost = -v(j,Nx-1)
      t.emplace_back(r, r, diag);

      if (v_is_unknown(j, i - 1)) t.emplace_back(r, s.iv(j, i - 1), -c * U);
      if (v_is_unknown(j, i + 1)) t.emplace_back(r, s.iv(j, i + 1), +c * U);
      if (v_is_unknown(j - 1, i)) t.emplace_back(r, s.iv(j - 1, i), -c * V);
      if (v_is_unknown(j + 1, i)) t.emplace_back(r, s.iv(j + 1, i), +c * V);

      const double cu = 0.25 * s.inertia * vx;
      if (u_is_unknown(j - 1, i)) t.emplace_back(r, s.iu(j - 1, i), cu);
      if (u_is_unknown(j - 1, i + 1)) t.emplace_back(r, s.iu(j - 1, i + 1), cu);
      if (u_is_unknown(j, i)) t.emplace_back(r, s.iu(j, i), cu);
      if (u_is_unknown(j, i + 1)) t.emplace_back(r, s.iu(j, i + 1), cu);
    }
}

// J = A + dN/dw at the given state, factorised for the derivative solves.
bool build_jacobian(Solver& s, const Vec& w) {
  std::vector<Trip> trips;
  trips.reserve(9 * s.n);
  for (int k = 0; k < s.A.outerSize(); ++k)
    for (SpMat::InnerIterator it(s.A, k); it; ++it)
      trips.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()),
                         it.value());
  convection_jacobian(s, w, trips);

  s.J.resize(s.n, s.n);
  s.J.setFromTriplets(trips.begin(), trips.end());  // duplicates are summed
  s.J.makeCompressed();

  s.luJ = std::make_unique<LU>();
  s.luJ->analyzePattern(s.J);
  s.luJ->factorize(s.J);
  s.luJT.reset();  // stale once J changes
  if (s.luJ->info() != Eigen::Success) {
    s.luJ.reset();
    return false;
  }
  return true;
}

// Buoyancy right-hand side: only the y-momentum rows are loaded.
Vec build_rhs(const Solver& s, const double* T) {
  Vec b = Vec::Zero(s.n);
  for (int j = 1; j <= s.Ny - 1; ++j)
    for (int i = 0; i < s.Nx; ++i)
      b[s.iv(j, i)] =
          s.Ra * s.Pr * 0.5 * (T[s.ic(j - 1, i)] + T[s.ic(j, i)]);
  return b;
}

void scatter(const Solver& s, const Vec& w, double* u, double* v, double* p) {
  std::memset(u, 0, sizeof(double) * s.Ny * (s.Nx + 1));
  std::memset(v, 0, sizeof(double) * (s.Ny + 1) * s.Nx);
  for (int j = 0; j < s.Ny; ++j)
    for (int i = 1; i <= s.Nx - 1; ++i) u[s.iuf(j, i)] = w[s.iu(j, i)];
  for (int j = 1; j <= s.Ny - 1; ++j)
    for (int i = 0; i < s.Nx; ++i) v[s.ivf(j, i)] = w[s.iv(j, i)];
  if (p)
    for (int j = 0; j < s.Ny; ++j)
      for (int i = 0; i < s.Nx; ++i) p[s.ic(j, i)] = w[s.ip(j, i)];
}

}  // namespace

extern "C" {

// Create and factorise. Returns nullptr if the factorisation fails.
//
// `inertia` weights the convective term: 0 is the original Stokes-Brinkman
// system and leaves every code path below identical to what it was.
SB_API void* sb_create_ns(int Nx, int Ny, double Pr, double Ra, double inertia,
                          const double* alpha) {
  auto* s = new Solver();
  s->Nx = Nx;
  s->Ny = Ny;
  s->Pr = Pr;
  s->Ra = Ra;
  s->inertia = inertia;
  s->h = 1.0 / static_cast<double>(Nx);
  s->nu = Ny * (Nx - 1);
  s->nv = (Ny - 1) * Nx;
  s->np = Nx * Ny;
  s->n = s->nu + s->nv + s->np;
  s->alpha.assign(alpha, alpha + Nx * Ny);

  assemble(*s);
  s->lu.analyzePattern(s->A);
  s->lu.factorize(s->A);
  s->ok = (s->lu.info() == Eigen::Success);
  if (!s->ok) {
    delete s;
    return nullptr;
  }
  return s;
}

// Original entry point, preserved exactly: the Stokes-Brinkman system. Keeping
// it means the pre-existing tests exercise the unmodified path and would catch
// any regression introduced by the inertia work.
SB_API void* sb_create(int Nx, int Ny, double Pr, double Ra, const double* alpha) {
  return sb_create_ns(Nx, Ny, Pr, Ra, 0.0, alpha);
}

SB_API void sb_destroy(void* h) { delete static_cast<Solver*>(h); }

SB_API int sb_n_unknowns(void* h) { return static_cast<Solver*>(h)->n; }

// Forward solve.
//
// Without inertia this is w = A^{-1} b(T), one back-substitution against the
// factorisation built in sb_create. With inertia the system is nonlinear,
// R(w) = A w + N(w) - b(T), and we run Newton from the Stokes solution -- a
// good start precisely because N is quadratic and so vanishes with the flow.
// The Jacobian at the converged state is kept factorised, because that is the
// operator the JVP and VJP must invert.
SB_API int sb_apply(void* h, const double* T, double* u, double* v, double* p) {
  auto& s = *static_cast<Solver*>(h);
  const Vec b = build_rhs(s, T);
  Vec w = s.lu.solve(b);
  if (s.lu.info() != Eigen::Success) return 1;

  if (s.inertia != 0.0) {
    // Scale the convergence test by the load. The buoyancy right-hand side
    // carries a factor Ra*Pr, so at Ra = 1e6 an absolute tolerance of 1e-12 is
    // below the rounding of the residual evaluation itself and Newton would
    // simply run until the line search gave up.
    const double bscale = std::max(1.0, b.lpNorm<Eigen::Infinity>());
    for (int it = 0; it < 60; ++it) {
      Vec R = s.A * w + convect(s, w) - b;
      const double rn = R.lpNorm<Eigen::Infinity>();
      if (rn < 1e-12 * bscale) break;
      if (!build_jacobian(s, w)) return 2;
      Vec dw = s.luJ->solve(R);
      if (s.luJ->info() != Eigen::Success) return 2;

      // Backtracking line search. Only ever accept a step that reduces the
      // residual: at high Ra the undamped Newton step can overshoot into a
      // region where the convective term dominates and never come back.
      double step = 1.0;
      bool accepted = false;
      for (int k = 0; k < 10; ++k) {
        Vec wt = w - step * dw;
        Vec Rt = s.A * wt + convect(s, wt) - b;
        if (Rt.lpNorm<Eigen::Infinity>() < rn) {
          w = wt;
          accepted = true;
          break;
        }
        step *= 0.5;
      }
      if (!accepted) break;  // at the accuracy floor, or genuinely stuck
    }
    // Derivatives are taken about the state we actually converged to.
    if (!build_jacobian(s, w)) return 2;
    s.w_star = w;
  }

  scatter(s, w, u, v, p);
  return 0;
}

// Residual at the last converged state, RELATIVE to the load, for callers that
// want to verify the nonlinear solve rather than trust it. Relative because the
// buoyancy right-hand side scales with Ra*Pr: an absolute number here would say
// more about the Rayleigh number than about the quality of the solve.
SB_API double sb_residual(void* h, const double* T) {
  auto& s = *static_cast<Solver*>(h);
  if (s.inertia == 0.0 || s.w_star.size() != s.n) return 0.0;
  const Vec b = build_rhs(s, T);
  Vec R = s.A * s.w_star + convect(s, s.w_star) - b;
  return R.lpNorm<Eigen::Infinity>() / std::max(1.0, b.lpNorm<Eigen::Infinity>());
}

// Forward-mode: dw = A^{-1} ( db/dT dT - (dA/dalpha dalpha) w ).
// Reuses the existing factorisation, so a tangent costs one back-substitution.
SB_API int sb_jvp(void* h, const double* u, const double* v, const double* dalpha,
           const double* dT, double* du, double* dv) {
  auto& s = *static_cast<Solver*>(h);
  Vec r = Vec::Zero(s.n);

  if (dT) {
    for (int j = 1; j <= s.Ny - 1; ++j)
      for (int i = 0; i < s.Nx; ++i)
        r[s.iv(j, i)] +=
            s.Ra * s.Pr * 0.5 * (dT[s.ic(j - 1, i)] + dT[s.ic(j, i)]);
  }
  if (dalpha) {
    // -(dA/dalpha . dalpha) w : the Brinkman term is the only alpha dependence.
    for (int j = 0; j < s.Ny; ++j)
      for (int i = 1; i <= s.Nx - 1; ++i) {
        const double da =
            0.5 * (dalpha[s.ic(j, i - 1)] + dalpha[s.ic(j, i)]);
        r[s.iu(j, i)] -= s.Pr * da * u[s.iuf(j, i)];
      }
    for (int j = 1; j <= s.Ny - 1; ++j)
      for (int i = 0; i < s.Nx; ++i) {
        const double da =
            0.5 * (dalpha[s.ic(j - 1, i)] + dalpha[s.ic(j, i)]);
        r[s.iv(j, i)] -= s.Pr * da * v[s.ivf(j, i)];
      }
  }

  // With inertia the operator is the Jacobian at the converged state, not A.
  // Neither alpha nor T enters the convective term, so the right-hand side
  // assembled above is unchanged -- only what we invert it against differs.
  Vec dw;
  if (s.inertia != 0.0) {
    if (!s.luJ) return 1;  // sb_apply must have been called first
    dw = s.luJ->solve(r);
    if (s.luJ->info() != Eigen::Success) return 1;
  } else {
    dw = s.lu.solve(r);
    if (s.lu.info() != Eigen::Success) return 1;
  }
  scatter(s, dw, du, dv, nullptr);
  return 0;
}

// Reverse-mode: lam = A^{-T} wbar, then scatter against dA/dalpha and db/dT.
SB_API int sb_vjp(void* h, const double* u, const double* v, const double* ubar,
           const double* vbar, const double* pbar, double* alphabar,
           double* Tbar) {
  auto& s = *static_cast<Solver*>(h);

  // The operator is nonsymmetric (the pressure pin breaks the saddle-point
  // symmetry, and convection is nonsymmetric outright), so the adjoint needs
  // its own factorisation. Cached: A^T is design-dependent only, while J^T is
  // rebuilt whenever sb_apply converges to a new state.
  LU* adj = nullptr;
  if (s.inertia != 0.0) {
    if (!s.luJ) return 1;  // sb_apply must have been called first
    if (!s.luJT) {
      SpMat Jt = SpMat(s.J.transpose());
      Jt.makeCompressed();
      s.luJT = std::make_unique<LU>();
      s.luJT->analyzePattern(Jt);
      s.luJT->factorize(Jt);
      if (s.luJT->info() != Eigen::Success) {
        s.luJT.reset();
        return 1;
      }
    }
    adj = s.luJT.get();
  } else {
    if (!s.luT) {
      SpMat At = SpMat(s.A.transpose());
      At.makeCompressed();
      s.luT = std::make_unique<LU>();
      s.luT->analyzePattern(At);
      s.luT->factorize(At);
      if (s.luT->info() != Eigen::Success) {
        s.luT.reset();
        return 1;
      }
    }
    adj = s.luT.get();
  }

  Vec wbar = Vec::Zero(s.n);
  if (ubar)
    for (int j = 0; j < s.Ny; ++j)
      for (int i = 1; i <= s.Nx - 1; ++i) wbar[s.iu(j, i)] = ubar[s.iuf(j, i)];
  if (vbar)
    for (int j = 1; j <= s.Ny - 1; ++j)
      for (int i = 0; i < s.Nx; ++i) wbar[s.iv(j, i)] = vbar[s.ivf(j, i)];
  if (pbar)
    for (int j = 0; j < s.Ny; ++j)
      for (int i = 0; i < s.Nx; ++i) wbar[s.ip(j, i)] = pbar[s.ic(j, i)];

  Vec lam = adj->solve(wbar);
  if (adj->info() != Eigen::Success) return 1;

  if (Tbar) {
    std::memset(Tbar, 0, sizeof(double) * s.Nx * s.Ny);
    // b_v[j,i] = Ra Pr (T[j-1,i] + T[j,i]) / 2  =>  split the seed both ways.
    for (int j = 1; j <= s.Ny - 1; ++j)
      for (int i = 0; i < s.Nx; ++i) {
        const double c = s.Ra * s.Pr * 0.5 * lam[s.iv(j, i)];
        Tbar[s.ic(j - 1, i)] += c;
        Tbar[s.ic(j, i)] += c;
      }
  }

  if (alphabar) {
    std::memset(alphabar, 0, sizeof(double) * s.Nx * s.Ny);
    // dw/dalpha = -A^{-1} (dA/dalpha) w, hence the minus sign here.
    for (int j = 0; j < s.Ny; ++j)
      for (int i = 1; i <= s.Nx - 1; ++i) {
        const double c = -0.5 * s.Pr * lam[s.iu(j, i)] * u[s.iuf(j, i)];
        alphabar[s.ic(j, i - 1)] += c;
        alphabar[s.ic(j, i)] += c;
      }
    for (int j = 1; j <= s.Ny - 1; ++j)
      for (int i = 0; i < s.Nx; ++i) {
        const double c = -0.5 * s.Pr * lam[s.iv(j, i)] * v[s.ivf(j, i)];
        alphabar[s.ic(j - 1, i)] += c;
        alphabar[s.ic(j, i)] += c;
      }
  }
  return 0;
}

}  // extern "C"
