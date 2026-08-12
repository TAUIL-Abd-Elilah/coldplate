# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""How does the thermal block scale? Decides dense-vs-sparse for that Tesseract.

The coupled solve runs ~60 Picard sweeps and the optimisation runs a few
hundred design steps, so the per-sweep thermal cost gets multiplied by ~10^4.
Measure it before committing to an implementation.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from reference_jax import Config, material_maps, solve_thermal, thermal_residual

jax.config.update("jax_enable_x64", True)


def timeit(fn, n=3):
    fn()  # warm up / compile
    t0 = time.time()
    for _ in range(n):
        jax.block_until_ready(fn())
    return (time.time() - t0) / n


print(f"{'N':>5} {'n_unknowns':>11} {'assemble+solve':>15} {'x10^4 sweeps':>14}")
for N in (16, 24, 32, 48, 64):
    cfg = Config(Nx=N, Ny=N)
    rng = np.random.default_rng(0)
    rho = jnp.asarray(rng.uniform(0, 1, size=(N, N)))
    k, _ = material_maps(rho, cfg)
    u = jnp.asarray(rng.normal(size=(N, N + 1)))
    v = jnp.asarray(rng.normal(size=(N + 1, N)))

    t = timeit(lambda: solve_thermal(u, v, k, cfg))
    print(f"{N:>5} {N*N:>11} {t:>13.4f}s {t*1e4/3600:>12.2f}h")

# Sanity: how sparse is the operator actually? (5-point stencil => ~5 nnz/row)
cfg = Config(Nx=24, Ny=24)
rng = np.random.default_rng(0)
k, _ = material_maps(jnp.asarray(rng.uniform(0, 1, size=(24, 24))), cfg)
u = jnp.asarray(rng.normal(size=(24, 25)))
v = jnp.asarray(rng.normal(size=(25, 24)))
res = lambda T: thermal_residual(T.reshape(24, 24), u, v, k, cfg)  # noqa: E731
A = jax.jacfwd(res)(jnp.zeros(576))
nnz = int(jnp.sum(jnp.abs(A) > 0))
print(f"\nthermal operator at N=24: {nnz} nonzeros of {576*576} "
      f"({100*nnz/576**2:.2f}% dense, {nnz/576:.1f} nnz/row)")
