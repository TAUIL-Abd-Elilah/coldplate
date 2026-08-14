# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Pin down the headline claim at the operating point Ra = 3e4.

The sweep showed that at Ra = 3e4 the frozen-flow gradient has cosine
similarity -0.29 against the coupled gradient: it points uphill. That claim is
only worth making if OUR gradient is demonstrably the correct one, so here we
check it against central finite differences on the full coupled solve.

The adjoint system (I - Phi_T)^T lambda = dJ/dT is solved with GMRES rather
than a Neumann series. Neumann needs loop gain < 1; GMRES does not, so the
same code keeps working in the strongly coupled regime where the naive
approach has already fallen apart.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from coupling_strength import anderson_solve, frozen_grad, spectral_radius
from reference_jax import Config, coupled_step, objective

jax.config.update("jax_enable_x64", True)


def adjoint_grad_gmres(T_star, rho, cfg, tol=1e-12, restart=40, maxiter=200):
    """dJ/drho by implicit differentiation, adjoint solved with GMRES.

    Each GMRES matvec applies (I - Phi_T)^T, and every application of Phi_T^T
    is a VJP back through the thermal block followed by a VJP back through the
    fluid block. The Krylov iteration therefore bounces across the component
    boundary once per matvec. One could assemble the component Jacobians; this
    implementation instead composes their matrix-free actions.
    """
    phi = lambda T, r: coupled_step(T, cfg, r)  # noqa: E731
    _, vjp_fn = jax.vjp(phi, T_star, rho)
    g = jax.grad(lambda T: objective(T, cfg))(T_star)

    n_matvec = [0]

    def A_T(lam):
        n_matvec[0] += 1
        return lam - vjp_fn(lam)[0]

    lam, info = jax.scipy.sparse.linalg.gmres(
        A_T, g, tol=tol, atol=0.0, restart=restart, maxiter=maxiter
    )
    resid = float(jnp.linalg.norm(A_T(lam) - g) / jnp.linalg.norm(g))
    return vjp_fn(lam)[1], {"matvecs": n_matvec[0], "gmres_rel_resid": resid}


def fd_entry(rho, cfg, j, i, eps):
    """Central difference of the fully converged coupled objective."""
    Jp = objective(anderson_solve(rho.at[j, i].add(eps), cfg, tol=1e-14)[0], cfg)
    Jm = objective(anderson_solve(rho.at[j, i].add(-eps), cfg, tol=1e-14)[0], cfg)
    return float((Jp - Jm) / (2 * eps))


if __name__ == "__main__":
    N = 16
    cfg = Config(Nx=N, Ny=N, Ra=3.0e4)
    rng = np.random.default_rng(0)
    rho = jnp.asarray(rng.uniform(0.25, 0.75, size=(N, N)))

    print(f"grid {N}x{N}, Ra = {cfg.Ra:.0e}\n")

    t0 = time.time()
    T_star, info = anderson_solve(rho, cfg, tol=1e-14, max_iter=600)
    print(
        f"forward: converged={info['ok']} iters={info['iters']} "
        f"resid={info['residual']:.2e}  ({time.time()-t0:.1f}s)"
    )
    sr = spectral_radius(T_star, rho, cfg)
    print(f"coupling loop gain rho(Phi_T) = {sr:.4f}")
    print(f"chip temperature J = {float(objective(T_star, cfg)):.6f}\n")

    t0 = time.time()
    g_adj, ginfo = adjoint_grad_gmres(T_star, rho, cfg)
    print(
        f"adjoint (GMRES): {ginfo['matvecs']} matvecs, "
        f"rel resid {ginfo['gmres_rel_resid']:.2e}  ({time.time()-t0:.1f}s)"
    )
    g_frozen = frozen_grad(T_star, rho, cfg)

    idx = [(3, 4), (8, 8), (12, 5), (5, 11), (14, 2)]
    print(f"\n{'idx':>9} {'adjoint':>14} {'finite-diff':>14} {'rel err':>10} "
          f"{'frozen-flow':>14} {'frozen err':>11}")
    errs, ferrs = [], []
    for j, i in idx:
        fd = fd_entry(rho, cfg, j, i, 1e-5)
        a = float(g_adj[j, i])
        f = float(g_frozen[j, i])
        e = abs(a - fd) / max(abs(fd), 1e-30)
        fe = abs(f - fd) / max(abs(fd), 1e-30)
        errs.append(e)
        ferrs.append(fe)
        print(f"{str((j,i)):>9} {a:14.6e} {fd:14.6e} {e:10.2e} {f:14.6e} {fe:11.2e}")

    ga, gf = np.asarray(g_adj).ravel(), np.asarray(g_frozen).ravel()
    cos = float(ga @ gf / (np.linalg.norm(ga) * np.linalg.norm(gf)))
    print(f"\n  adjoint    vs finite-diff : max rel err {max(errs):.2e}   <-- our gradient")
    print(f"  frozen-flow vs finite-diff : max rel err {max(ferrs):.2e}   <-- naive gradient")
    print(f"  cosine(frozen, adjoint) = {cos:+.4f}")
    print(f"  descent test: <g_frozen, g_adjoint> = {float(ga @ gf):+.6e} "
          f"({'UPHILL - naive step increases J' if ga @ gf < 0 else 'still downhill'})")
