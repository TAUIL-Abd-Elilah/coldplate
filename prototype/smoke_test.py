# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Forward-solve smoke test for the reference model."""

import time

import jax.numpy as jnp
from reference_jax import (
    Config,
    material_maps,
    n_fluid_unknowns,
    objective,
    solve_coupled,
    solve_fluid,
    solve_thermal,
)

cfg = Config(Nx=20, Ny=20)
print(f"grid {cfg.Nx}x{cfg.Ny}, fluid unknowns = {n_fluid_unknowns(cfg)}")
rho_fluid = jnp.zeros((cfg.Ny, cfg.Nx))
k, alpha = material_maps(rho_fluid, cfg)

x = (jnp.arange(cfg.Nx) + 0.5) / cfg.Nx
y = (jnp.arange(cfg.Ny) + 0.5) / cfg.Ny

# --- 1. hydrostatic test: purely vertical dT must produce NO flow ---
# A body force that is a function of y alone is conservative, so pressure
# absorbs it exactly. Zero velocity here is the correct answer, not a bug.
T_vert = jnp.tile((1.0 - y)[:, None], (1, cfg.Nx))
u, v, _ = solve_fluid(T_vert, alpha, cfg)
print(f"[hydrostatic] max|v| = {jnp.abs(v).max():.3e}   (should be ~0)")

# --- 2. differentially heated: horizontal dT must drive a circulation ---
T_horiz = jnp.tile((1.0 - x)[None, :], (cfg.Ny, 1))
t0 = time.time()
u, v, _ = solve_fluid(T_horiz, alpha, cfg)
print(
    f"[convection ] max|u| = {jnp.abs(u).max():.4e}  max|v| = {jnp.abs(v).max():.4e}"
    f"   ({time.time()-t0:.2f}s)"
)
# hot on the left should rise, cold on the right should sink
print(
    f"[convection ] v(left half) = {v[:, : cfg.Nx // 2].mean():+.4e}  "
    f"v(right half) = {v[:, cfg.Nx // 2 :].mean():+.4e}   (expect + then -)"
)
div = (u[:, 1:] - u[:, :-1]) / cfg.h + (v[1:, :] - v[:-1, :]) / cfg.h
print(f"[convection ] max|div u| = {jnp.abs(div).max():.3e}   (should be ~0)")

# --- 3. Brinkman: solid domain must kill the flow ---
_, alpha_s = material_maps(jnp.ones((cfg.Ny, cfg.Nx)), cfg)
us, vs, _ = solve_fluid(T_horiz, alpha_s, cfg)
print(
    f"[brinkman   ] solid max|u| = {jnp.abs(us).max():.3e}  vs fluid "
    f"{jnp.abs(u).max():.3e}  (ratio {jnp.abs(us).max()/jnp.abs(u).max():.1e})"
)

# --- 4. pure conduction energy balance: flux in must equal flux out ---
zero_u = jnp.zeros((cfg.Ny, cfg.Nx + 1))
zero_v = jnp.zeros((cfg.Ny + 1, cfg.Nx))
T_cond = solve_thermal(zero_u, zero_v, k, cfg)
q_in = float(cfg.q_chip * cfg.chip_mask().sum() * cfg.h)
q_out = float((k[cfg.Ny - 1, :] * T_cond[cfg.Ny - 1, :] / (0.5 * cfg.h)).sum() * cfg.h)
print(
    f"[conduction ] T>0? min={T_cond.min():+.4f} max={T_cond.max():+.4f} | "
    f"q_in={q_in:.5f} q_out={q_out:.5f}  rel.err={abs(q_in-q_out)/q_in:.2e}"
)

# --- 5. coupled fixed point at a few Rayleigh numbers ---
for Ra in (1e2, 1e3, 1e4):
    c = Config(Nx=20, Ny=20, Ra=Ra)
    t0 = time.time()
    T, info = solve_coupled(jnp.zeros((c.Ny, c.Nx)), c, max_iter=200, relax=0.5)
    print(
        f"[coupled Ra={Ra:>7.0e}] iters={info['iters']:3d} resid={info['residual']:.2e} "
        f"J={objective(T, c):.5f} Tmax={T.max():.4f}  ({time.time()-t0:.1f}s)"
    )
