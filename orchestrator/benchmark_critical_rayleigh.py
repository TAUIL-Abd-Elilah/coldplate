# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Validate the coupled physics against the classical critical Rayleigh number.

Rayleigh-Benard convection: a fluid layer heated from below is motionless until
buoyancy overcomes viscous and thermal diffusion, and then convects. For rigid
(no-slip) top and bottom walls the onset sits at a precisely known value,

    Ra_c = 1707.762                      (unbounded horizontal layer)

This is a good test for *this* solver specifically, because Stokes flow is the
infinite-Prandtl limit and that is exactly the regime the classical result is
derived in. A Navier-Stokes benchmark such as de Vahl Davis would not be: it
runs at Pr = 0.71 where the inertia term this solver omits is not small, so
disagreement there would prove nothing.

The onset is also exactly where our own machinery says it should be. At the
conduction state the coupling loop is

    dT -> buoyancy -> du -> advection of the base gradient -> dT

which is the linear stability operator. So convection onsets precisely when the
coupling loop gain reaches one:

    rho(Phi_T) = 1   <=>   Ra = Ra_c

The classical value is for an unbounded layer. Our box has no-slip side walls,
which stabilise it, so a confined box must have a *higher* critical Rayleigh
number -- and widening the box must bring it down towards 1707.762. That
convergence is the test.

Ra is reported in the classical convention, based on layer depth. This code
non-dimensionalises on the x-extent (h = 1/Nx), so with Ny cells the depth is
d = Ny/Nx and  Ra_classical = Ra_code * d^3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pipeline import ColdPlate, Params

RA_C_UNBOUNDED = 1707.762  # Chandrasekhar, rigid-rigid


def loop_gain_at_conduction(cp: ColdPlate, alpha, k, n_power=60) -> float:
    """rho(Phi_T) at the motionless conduction state."""
    p = cp.params
    # conduction profile: linear from t_hot at the bottom wall to 0 at the top
    d = p.Ny / p.Nx
    y = (np.arange(p.Ny) + 0.5) * (1.0 / p.Nx)
    T0 = jnp.asarray(np.tile((p.t_hot * (1.0 - y / d))[:, None], (1, p.Nx)))

    phi = lambda T: cp.phi(T, alpha, k)  # noqa: E731
    v = jnp.asarray(np.random.default_rng(1).normal(size=(p.Ny, p.Nx)))
    v = v / jnp.linalg.norm(v)
    lam = 0.0
    for _ in range(n_power):
        w = jax.jvp(phi, (T0,), (v,))[1]
        nrm = float(jnp.linalg.norm(w))
        if nrm < 1e-300:
            return 0.0
        lam, v = nrm, w / nrm
    return lam


def critical_ra(Nx: int, Ny: int, lo: float, hi: float, tol=2e-3) -> tuple[float, int]:
    """Bisect in Ra_code for the value where the loop gain crosses one."""
    alpha = jnp.zeros((Ny, Nx))  # pure fluid, no solid
    k = jnp.ones((Ny, Nx))  # uniform conductivity
    calls = 0

    def gain(Ra):
        nonlocal calls
        calls += 1
        p = Params(Nx=Nx, Ny=Ny, Ra=Ra, Pr=7.0, bc_mode=1.0, t_hot=1.0)
        with ColdPlate(params=p) as cp:
            return loop_gain_at_conduction(cp, alpha, k)

    g_lo, g_hi = gain(lo), gain(hi)
    if not (g_lo < 1.0 < g_hi):
        raise RuntimeError(f"onset not bracketed: gain({lo:.0f})={g_lo:.3f}, "
                           f"gain({hi:.0f})={g_hi:.3f}")
    while (hi - lo) / lo > tol:
        mid = 0.5 * (lo + hi)
        if gain(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), calls


def main(Ny=16, out="results/critical_rayleigh.json"):
    rows = []
    print(f"classical value for an unbounded layer: Ra_c = {RA_C_UNBOUNDED}")
    print("a confined box is more stable, so Ra_c should exceed it and fall "
          "toward it as the box widens\n")
    print(f"{'aspect':>7} {'grid':>10} {'depth d':>9} {'Ra_c (code)':>13} "
          f"{'Ra_c (classical)':>17} {'excess':>8}")

    for aspect in (1, 2, 4, 8):
        Nx = Ny * aspect
        d = Ny / Nx
        # bracket in classical units, converted to code units
        lo, hi = 1.2e3 / d**3, 4.0e4 / d**3
        try:
            ra_code, _ = critical_ra(Nx, Ny, lo, hi)
        except RuntimeError as e:
            print(f"{aspect:>7} {f'{Nx}x{Ny}':>10}   {e}")
            continue
        ra_cl = ra_code * d**3
        excess = ra_cl / RA_C_UNBOUNDED - 1.0
        rows.append({"aspect": aspect, "Nx": Nx, "Ny": Ny, "d": d,
                     "ra_code": ra_code, "ra_classical": ra_cl, "excess": excess})
        print(f"{aspect:>7} {f'{Nx}x{Ny}':>10} {d:9.4f} {ra_code:13.1f} "
              f"{ra_cl:17.2f} {100*excess:+7.1f}%", flush=True)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ny", type=int, default=16, help="cells across the layer depth")
    ap.add_argument("--out", default="results/critical_rayleigh.json")
    main(**vars(ap.parse_args()))
