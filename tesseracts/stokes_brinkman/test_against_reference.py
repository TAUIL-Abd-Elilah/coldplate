# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Validate the C++/Eigen solver and its hand-derived adjoint against JAX.

The C++ block computes derivatives by hand (transpose solve + analytic
scatter); the JAX reference computes them by autodiff. They share no code, so
agreement to machine precision is strong evidence both are right -- and it is
what licenses us to compose them and trust the end-to-end gradient.

Checks:
  1. forward   u, v      vs jax reference
  2. JVP       du, dv    vs jax.jvp of the reference
  3. VJP       dalpha,dT vs jax.vjp of the reference
  4. adjoint consistency <J dx, ybar> == <dx, J^T ybar>  (internal to C++)
"""

import ctypes
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from reference_jax import Config, solve_fluid  # noqa: E402

jax.config.update("jax_enable_x64", True)

_here = Path(__file__).parent
_cand = [_here / "lib" / "stokes_brinkman.dll", _here / "lib" / "libstokes_brinkman.so"]
_LIB = next(p for p in _cand if p.exists())
lib = ctypes.CDLL(str(_LIB))
dp = ctypes.POINTER(ctypes.c_double)
lib.sb_create.restype = ctypes.c_void_p
lib.sb_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, dp]
lib.sb_destroy.argtypes = [ctypes.c_void_p]
lib.sb_apply.restype = ctypes.c_int
lib.sb_apply.argtypes = [ctypes.c_void_p, dp, dp, dp, dp]
lib.sb_jvp.restype = ctypes.c_int
lib.sb_jvp.argtypes = [ctypes.c_void_p, dp, dp, dp, dp, dp, dp]
lib.sb_vjp.restype = ctypes.c_int
lib.sb_vjp.argtypes = [ctypes.c_void_p, dp, dp, dp, dp, dp, dp, dp]

P = lambda a: a.ctypes.data_as(dp)  # noqa: E731
C = lambda a: np.ascontiguousarray(a, dtype=np.float64)  # noqa: E731


def relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = max(np.max(np.abs(b)), 1e-300)
    return float(np.max(np.abs(a - b)) / d)


def main():
    N = 14
    cfg = Config(Nx=N, Ny=N, Ra=3.0e4, Pr=7.0)
    rng = np.random.default_rng(7)

    # A design with real contrast: some near-solid cells, some open fluid.
    alpha = C(10.0 ** rng.uniform(-2, 4, size=(N, N)))
    T = C(rng.normal(size=(N, N)))

    h = lib.sb_create(N, N, ctypes.c_double(cfg.Pr), ctypes.c_double(cfg.Ra), P(alpha))
    assert h, "sb_create returned null"
    h = ctypes.c_void_p(h)

    # ---- 1. forward ----
    u = np.zeros((N, N + 1)); v = np.zeros((N + 1, N)); p = np.zeros((N, N))
    assert lib.sb_apply(h, P(T), P(u), P(v), P(p)) == 0
    u_ref, v_ref, _ = solve_fluid(jnp.asarray(T), jnp.asarray(alpha), cfg)
    e_u, e_v = relerr(u, u_ref), relerr(v, v_ref)
    print(f"[forward] u rel err {e_u:.3e}   v rel err {e_v:.3e}")

    # ---- 2. JVP ----
    d_alpha = C(rng.normal(size=(N, N)) * alpha)  # scale-aware perturbation
    d_T = C(rng.normal(size=(N, N)))
    du = np.zeros((N, N + 1)); dv = np.zeros((N + 1, N))
    assert lib.sb_jvp(h, P(u), P(v), P(d_alpha), P(d_T), P(du), P(dv)) == 0

    f = lambda a, t: solve_fluid(t, a, cfg)[:2]  # noqa: E731
    (_, _), (du_ref, dv_ref) = jax.jvp(
        f, (jnp.asarray(alpha), jnp.asarray(T)), (jnp.asarray(d_alpha), jnp.asarray(d_T))
    )
    print(f"[jvp    ] du rel err {relerr(du, du_ref):.3e}   dv rel err {relerr(dv, dv_ref):.3e}")

    # ---- 3. VJP ----
    ubar = C(rng.normal(size=(N, N + 1)))
    vbar = C(rng.normal(size=(N + 1, N)))
    abar = np.zeros((N, N)); Tbar = np.zeros((N, N))
    assert lib.sb_vjp(h, P(u), P(v), P(ubar), P(vbar), None, P(abar), P(Tbar)) == 0

    _, vjp_fn = jax.vjp(f, jnp.asarray(alpha), jnp.asarray(T))
    abar_ref, Tbar_ref = vjp_fn((jnp.asarray(ubar), jnp.asarray(vbar)))
    print(f"[vjp    ] dalpha rel err {relerr(abar, abar_ref):.3e}   dT rel err {relerr(Tbar, Tbar_ref):.3e}")

    # ---- 4. adjoint consistency, purely inside the C++ component ----
    lhs = float(np.sum(du * ubar) + np.sum(dv * vbar))
    rhs = float(np.sum(d_alpha * abar) + np.sum(d_T * Tbar))
    print(f"[adjoint] <J dx, ybar> = {lhs:.12e}")
    print(f"[adjoint] <dx, J^T yb> = {rhs:.12e}")
    print(f"[adjoint] rel mismatch = {abs(lhs-rhs)/max(abs(lhs),1e-300):.3e}")

    lib.sb_destroy(h)

    worst = max(e_u, e_v, relerr(du, du_ref), relerr(dv, dv_ref),
                relerr(abar, abar_ref), relerr(Tbar, Tbar_ref),
                abs(lhs - rhs) / max(abs(lhs), 1e-300))
    print(f"\nworst rel err across all checks: {worst:.3e}")
    print("PASS" if worst < 1e-9 else "FAIL")
    return 0 if worst < 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main())
