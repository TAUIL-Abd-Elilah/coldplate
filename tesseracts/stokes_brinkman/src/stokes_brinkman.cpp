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
  int nu = 0, nv = 0, np = 0, n = 0;
  std::vector<double> alpha;  // (Ny*Nx) cell-centred Brinkman coefficient

  SpMat A;
  LU lu;                        // factorisation of A
  std::unique_ptr<LU> luT;      // factorisation of A^T, built lazily for VJPs
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
SB_API void* sb_create(int Nx, int Ny, double Pr, double Ra, const double* alpha) {
  auto* s = new Solver();
  s->Nx = Nx;
  s->Ny = Ny;
  s->Pr = Pr;
  s->Ra = Ra;
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

SB_API void sb_destroy(void* h) { delete static_cast<Solver*>(h); }

SB_API int sb_n_unknowns(void* h) { return static_cast<Solver*>(h)->n; }

// Forward solve: w = A^{-1} b(T).
SB_API int sb_apply(void* h, const double* T, double* u, double* v, double* p) {
  auto& s = *static_cast<Solver*>(h);
  Vec w = s.lu.solve(build_rhs(s, T));
  if (s.lu.info() != Eigen::Success) return 1;
  scatter(s, w, u, v, p);
  return 0;
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

  Vec dw = s.lu.solve(r);
  if (s.lu.info() != Eigen::Success) return 1;
  scatter(s, dw, du, dv, nullptr);
  return 0;
}

// Reverse-mode: lam = A^{-T} wbar, then scatter against dA/dalpha and db/dT.
SB_API int sb_vjp(void* h, const double* u, const double* v, const double* ubar,
           const double* vbar, const double* pbar, double* alphabar,
           double* Tbar) {
  auto& s = *static_cast<Solver*>(h);

  if (!s.luT) {
    // A is nonsymmetric (the pressure pin breaks the saddle-point symmetry),
    // so the adjoint needs its own factorisation. Built once and cached.
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

  Vec lam = s.luT->solve(wbar);
  if (s.luT->info() != Eigen::Success) return 1;

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
