# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Validate the sparse thermal assembly and its derivatives.

The sparse operator is assembled by hand for speed, but the physics is defined
by the JAX residual. If those two ever disagree the whole pipeline is silently
wrong, so check them directly:

  1. A T - b == residual(T)  for random T   (assembly matches the physics)
  2. residual(apply()) == 0                 (the solve is consistent)
  3. JVP / VJP vs autodiff through a dense reference solve
  4. <J dx, ybar> == <dx, J^T ybar>         (adjoint consistency)
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import tesseract_api as api  # noqa: E402

jax.config.update("jax_enable_x64", True)


def relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def dense_solve(u, v, k, q, cf):
    """Reference: build the operator with jacfwd and solve densely, in pure JAX."""
    Ny, Nx = k.shape
    res = lambda t: api.residual(t.reshape(Ny, Nx), u, v, k, q, cf).ravel()  # noqa: E731
    A = jax.jacfwd(res)(jnp.zeros(Nx * Ny))
    b = -res(jnp.zeros(Nx * Ny))
    return jnp.linalg.solve(A, b).reshape(Ny, Nx)


def main():
    N = 12
    q, cf = 1.0, 0.4
    rng = np.random.default_rng(3)
    u = jnp.asarray(rng.normal(size=(N, N + 1)) * 5.0).at[:, 0].set(0.0).at[:, N].set(0.0)
    v = jnp.asarray(rng.normal(size=(N + 1, N)) * 5.0).at[0, :].set(0.0).at[N, :].set(0.0)
    k = jnp.asarray(rng.uniform(0.02, 1.0, size=(N, N)))

    # ---- 1. assembly matches the JAX residual ----
    A = api.assemble(u, v, k, N, N)
    b = api.rhs(N, N, q, cf)
    T_rand = jnp.asarray(rng.normal(size=(N, N)))
    lhs = A @ np.asarray(T_rand).ravel() - b
    ref = np.asarray(api.residual(T_rand, u, v, k, q, cf)).ravel()
    print(f"[assembly] ||A T - b - residual(T)|| rel = {relerr(lhs, ref):.3e}")

    # ---- 2. the solve is consistent ----
    inputs = api.InputSchema(u=u, v=v, k=k, q_chip=q, chip_frac=cf)
    T = api.apply(inputs).T
    r = np.asarray(api.residual(jnp.asarray(T), u, v, k, q, cf))
    print(f"[solve   ] ||residual(T_solved)|| = {np.max(np.abs(r)):.3e}")
    print(f"[solve   ] vs dense JAX reference rel = {relerr(T, dense_solve(u, v, k, q, cf)):.3e}")

    # ---- 3. JVP / VJP vs autodiff through the dense reference ----
    du = jnp.asarray(rng.normal(size=u.shape))
    dv = jnp.asarray(rng.normal(size=v.shape))
    dk = jnp.asarray(rng.normal(size=k.shape))
    jvp = api.jacobian_vector_product(
        inputs, {"u", "v", "k"}, {"T"}, {"u": du, "v": dv, "k": dk}
    )["T"]
    f = lambda a, bb, c: dense_solve(a, bb, c, q, cf)  # noqa: E731
    _, jvp_ref = jax.jvp(f, (u, v, k), (du, dv, dk))
    print(f"[jvp     ] rel err = {relerr(jvp, jvp_ref):.3e}")

    Tbar = jnp.asarray(rng.normal(size=(N, N)))
    vjp = api.vector_jacobian_product(inputs, {"u", "v", "k"}, {"T"}, {"T": Tbar})
    _, vjp_fn = jax.vjp(f, u, v, k)
    gu, gv, gk = vjp_fn(Tbar)
    e = max(relerr(vjp["u"], gu), relerr(vjp["v"], gv), relerr(vjp["k"], gk))
    print(f"[vjp     ] rel err = {e:.3e}")

    # ---- 4. adjoint consistency ----
    lhs_d = float(jnp.sum(jvp * Tbar))
    rhs_d = float(
        np.sum(np.asarray(du) * vjp["u"])
        + np.sum(np.asarray(dv) * vjp["v"])
        + np.sum(np.asarray(dk) * vjp["k"])
    )
    print(f"[adjoint ] <J dx, ybar> = {lhs_d:.12e}")
    print(f"[adjoint ] <dx, J^T yb> = {rhs_d:.12e}")
    mismatch = abs(lhs_d - rhs_d) / max(abs(lhs_d), 1e-300)
    print(f"[adjoint ] rel mismatch = {mismatch:.3e}")

    worst = max(relerr(lhs, ref), relerr(jvp, jvp_ref), e, mismatch)
    print(f"\nworst rel err: {worst:.3e}")
    print("PASS" if worst < 1e-9 else "FAIL")
    return 0 if worst < 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main())
