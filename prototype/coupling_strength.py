# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""How strong is the two-way coupling, and where does naive composition break?

The submission claims the end-to-end gradient cannot be assembled from the
blocks in isolation. That claim is only interesting in a regime where the
feedback T -> buoyancy -> u -> advection -> T actually bites. This script
measures, as a function of Rayleigh number:

  * rho(Phi_T), the spectral radius of the fixed-point Jacobian -- i.e. the
    gain of one trip around the coupling loop;
  * the error of the "frozen-flow" gradient, which is what you are forced to
    use when the fluid solver cannot hand you derivatives.

Plain Picard stalls once rho(Phi_T) approaches 1, so we use Anderson
acceleration to reach the strongly coupled regime.
"""

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
    solve_thermal,
    _unpack_fluid,
)

jax.config.update("jax_enable_x64", True)


def anderson_solve(rho, cfg, m=6, max_iter=400, tol=1e-12, beta=0.6):
    """Anderson-accelerated fixed-point solve for T = Phi(T, rho).

    Keeps the last m residuals and takes the affine combination that minimises
    the residual norm. This converges well into the regime where damped Picard
    stalls, which is exactly the regime we need for the coupling to matter.
    """
    phi = lambda T: coupled_step(T, cfg, rho)  # noqa: E731
    shape = (cfg.Ny, cfg.Nx)
    T = jnp.zeros(shape)

    G, F = [], []  # history of Phi(T_k) and residuals Phi(T_k) - T_k
    for it in range(max_iter):
        g = phi(T)
        f = (g - T).ravel()
        res = float(jnp.max(jnp.abs(f)))
        if res < tol:
            return T, {"iters": it + 1, "residual": res, "ok": True}

        G.append(np.asarray(g.ravel()))
        F.append(np.asarray(f))
        if len(G) > m:
            G.pop(0)
            F.pop(0)

        if len(F) == 1:
            T = jnp.asarray((T.ravel() + beta * f).reshape(shape))
            continue

        # min || sum_i a_i F_i ||  s.t.  sum_i a_i = 1, solved in difference form
        dF = np.stack(F, axis=1)
        A = np.concatenate([dF, np.ones((1, dF.shape[1]))], axis=0)
        b = np.zeros(dF.shape[0] + 1)
        b[-1] = 1.0
        try:
            a, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            T = jnp.asarray((T.ravel() + beta * f).reshape(shape))
            continue
        T_new = (np.stack(G, axis=1) @ a).reshape(shape)
        T = jnp.asarray(T_new)

    return T, {"iters": max_iter, "residual": res, "ok": False}


def spectral_radius(T_star, rho, cfg, n_power=60):
    """Power-iterate Phi_T at the fixed point to get the coupling loop gain."""
    phi = lambda T: coupled_step(T, cfg, rho)  # noqa: E731
    v = jnp.asarray(np.random.default_rng(1).normal(size=(cfg.Ny, cfg.Nx)))
    v = v / jnp.linalg.norm(v)
    lam = 0.0
    for _ in range(n_power):
        w = jax.jvp(phi, (T_star,), (v,))[1]
        nrm = jnp.linalg.norm(w)
        if float(nrm) < 1e-300:
            return 0.0
        lam = float(nrm)
        v = w / nrm
    return lam


def true_grad(T_star, rho, cfg, n_iter=2000, tol=1e-13):
    """Implicit-diff gradient, using a Neumann/Richardson sweep on the adjoint."""
    phi = lambda T, r: coupled_step(T, cfg, r)  # noqa: E731
    _, vjp_fn = jax.vjp(phi, T_star, rho)
    g = jax.grad(lambda T: objective(T, cfg))(T_star)
    lam, w = g, g
    # Neumann series lambda = sum_k (Phi_T^T)^k g, accumulated term by term so
    # it works whenever the loop gain is < 1 without needing a relaxation knob.
    for _ in range(n_iter):
        w = vjp_fn(w)[0]
        lam = lam + w
        if float(jnp.max(jnp.abs(w))) < tol:
            break
    return vjp_fn(lam)[1]


def frozen_grad(T_star, rho, cfg):
    """Gradient with the velocity field held constant (no derivative from fluid)."""
    _, alpha = material_maps(rho, cfg)
    w = jnp.linalg.solve(assemble_fluid(alpha, cfg), fluid_rhs(T_star, alpha, cfg))
    u_f, v_f, _ = _unpack_fluid(w, cfg)

    def J_frozen(r):
        k, _ = material_maps(r, cfg)
        return objective(solve_thermal(u_f, v_f, k, cfg), cfg)

    return jax.grad(J_frozen)(rho)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 16
    rho = jnp.asarray(rng.uniform(0.25, 0.75, size=(N, N)))

    print(f"{'Ra':>9} {'conv':>5} {'iters':>6} {'rho(Phi_T)':>11} "
          f"{'Tchip':>9} {'|g_frozen-g|/|g|':>17} {'cos':>8}")
    for Ra in (1e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5):
        cfg = Config(Nx=N, Ny=N, Ra=Ra)
        T_star, info = anderson_solve(rho, cfg)
        if not info["ok"]:
            print(f"{Ra:9.0e} {'NO':>5} {info['iters']:6d}  residual={info['residual']:.2e}")
            continue
        sr = spectral_radius(T_star, rho, cfg)
        g = true_grad(T_star, rho, cfg)
        gf = frozen_grad(T_star, rho, cfg)
        gn, gfn = np.asarray(g).ravel(), np.asarray(gf).ravel()
        rel = np.linalg.norm(gfn - gn) / np.linalg.norm(gn)
        cos = float(gn @ gfn / (np.linalg.norm(gn) * np.linalg.norm(gfn)))
        print(f"{Ra:9.0e} {'yes':>5} {info['iters']:6d} {sr:11.4f} "
              f"{float(objective(T_star, cfg)):9.4f} {rel:17.4f} {cos:8.4f}")
