# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Three-way validation of the end-to-end gradient.

The whole submission rests on one claim: differentiating the *coupled* fixed
point gives the right sensitivity, while differentiating the blocks with the
feedback cut does not. This script tests that by computing dJ/drho three
independent ways and requiring them to agree:

  1. implicit differentiation of the fixed point (what we will ship)
  2. reverse-mode AD through the unrolled Picard loop (brute force)
  3. central finite differences on the whole coupled solve (ground truth)

It also computes a deliberately *wrong* gradient -- the one you get by
freezing the flow and ignoring that T feeds back into buoyancy -- to quantify
how badly naive component-wise differentiation fails.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from reference_jax import (
    Config,
    assemble_fluid,
    coupled_step,
    fluid_rhs,
    material_maps,
    objective,
    solve_coupled,
    solve_thermal,
    _unpack_fluid,
)

jax.config.update("jax_enable_x64", True)


def adjoint_grad(rho, cfg, relax=0.5, n_iter=400, tol=1e-13):
    """dJ/drho by implicit differentiation of the coupled fixed point.

    At the fixed point T* = Phi(T*, rho), the implicit function theorem gives
        dJ/drho = Phi_rho^T lambda,   (I - Phi_T)^T lambda = dJ/dT.
    We solve the adjoint system by the same damped iteration used forward,
    which converges under exactly the same spectral condition.

    Every application of Phi_T^T runs a VJP back through the thermal block and
    then through the fluid block -- i.e. the adjoint bounces across the
    component boundary once per iteration.
    """
    T_star, info = solve_coupled(rho, cfg, max_iter=600, relax=relax)

    phi = lambda T, r: coupled_step(T, cfg, r)  # noqa: E731
    _, vjp_fn = jax.vjp(phi, T_star, rho)

    g = jax.grad(lambda T: objective(T, cfg))(T_star)

    lam = g
    for i in range(n_iter):
        lam_new = (1.0 - relax) * lam + relax * (vjp_fn(lam)[0] + g)
        delta = float(jnp.max(jnp.abs(lam_new - lam)))
        lam = lam_new
        if delta < tol:
            break
    return vjp_fn(lam)[1], {"fwd_iters": info["iters"], "adj_iters": i + 1}


def frozen_flow_grad(rho, cfg, relax=0.5):
    """The WRONG gradient: differentiate the thermal block with the flow held fixed.

    This is what you get if you treat the pipeline as feed-forward and forget
    that temperature drives buoyancy which drives the flow which advects
    temperature. It is a common shortcut when the two solvers do not exchange
    derivatives; external derivative estimates would be another, costlier
    option.
    """
    T_star, _ = solve_coupled(rho, cfg, max_iter=600, relax=relax)
    _, alpha = material_maps(rho, cfg)
    w = jnp.linalg.solve(assemble_fluid(alpha, cfg), fluid_rhs(T_star, alpha, cfg))
    u_f, v_f, _ = _unpack_fluid(w, cfg)  # frozen, treated as constant

    def J_frozen(r):
        k, _ = material_maps(r, cfg)
        return objective(solve_thermal(u_f, v_f, k, cfg), cfg)

    return jax.grad(J_frozen)(rho)


def fd_grad(rho, cfg, idx, eps=1e-5, relax=0.5):
    """Central finite differences on the full coupled solve, at given indices."""
    out = []
    for j, i in idx:
        rp = rho.at[j, i].add(eps)
        rm = rho.at[j, i].add(-eps)
        Jp = objective(solve_coupled(rp, cfg, max_iter=600, relax=relax)[0], cfg)
        Jm = objective(solve_coupled(rm, cfg, max_iter=600, relax=relax)[0], cfg)
        out.append(float((Jp - Jm) / (2 * eps)))
    return np.array(out)


if __name__ == "__main__":
    cfg = Config(Nx=12, Ny=12, Ra=1.0e3)
    rng = np.random.default_rng(0)
    rho = jnp.asarray(rng.uniform(0.25, 0.75, size=(cfg.Ny, cfg.Nx)))

    print(f"grid {cfg.Nx}x{cfg.Ny}, Ra={cfg.Ra:.0e}\n")

    t0 = time.time()
    g_adj, info = adjoint_grad(rho, cfg)
    t_adj = time.time() - t0
    print(
        f"[1] implicit adjoint   {t_adj:6.1f}s  "
        f"(fwd {info['fwd_iters']} iters, adj {info['adj_iters']} iters)"
    )

    t0 = time.time()
    from reference_jax import loss_unrolled

    g_unroll = jax.grad(lambda r: loss_unrolled(r, cfg, 300, 0.5))(rho)
    t_unroll = time.time() - t0
    print(f"[2] unrolled AD        {t_unroll:6.1f}s")

    # a handful of random coordinates is enough to pin the gradient down
    idx = [(int(j), int(i)) for j, i in rng.integers(0, cfg.Nx, size=(6, 2))]
    t0 = time.time()
    g_fd = fd_grad(rho, cfg, idx)
    t_fd = time.time() - t0
    print(f"[3] finite differences {t_fd:6.1f}s  ({len(idx)} entries)\n")

    g_frozen = frozen_flow_grad(rho, cfg)

    a = np.array([float(g_adj[j, i]) for j, i in idx])
    u = np.array([float(g_unroll[j, i]) for j, i in idx])
    f = np.array([float(g_frozen[j, i]) for j, i in idx])

    print(f"{'idx':>8} {'implicit':>13} {'unrolled':>13} {'finite-diff':>13} {'frozen-flow':>13}")
    for n, (j, i) in enumerate(idx):
        print(f"{str((j,i)):>8} {a[n]:13.6e} {u[n]:13.6e} {g_fd[n]:13.6e} {f[n]:13.6e}")

    def relerr(x, y):
        return float(np.max(np.abs(x - y)) / max(np.max(np.abs(y)), 1e-30))

    print()
    print(f"  max rel err  implicit vs finite-diff : {relerr(a, g_fd):.3e}")
    print(f"  max rel err  unrolled vs finite-diff : {relerr(u, g_fd):.3e}")
    print(f"  max rel err  implicit vs unrolled    : {relerr(a, u):.3e}")
    print()
    cos = float(
        np.dot(np.asarray(g_frozen).ravel(), np.asarray(g_adj).ravel())
        / (np.linalg.norm(g_frozen) * np.linalg.norm(g_adj))
    )
    print(f"  FROZEN-FLOW (naive) vs true: rel err {relerr(f, g_fd):.3e}, "
          f"cosine similarity {cos:.4f}")
