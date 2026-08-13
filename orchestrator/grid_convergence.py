# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Observed order of accuracy of the coupled discretisation.

Verification rather than validation: this asks whether the code solves its own
equations correctly and at the rate the discretisation implies, independently
of whether those equations describe reality. (The critical-Rayleigh benchmark
does the validation half.)

Three details matter for this to mean anything, and the first two cost real
attempts to get right:

* The design must be grid-independent, so it is an analytic function of (x, y)
  sampled at cell centres rather than a random field.
* The density filter must be *bypassed entirely*. Its radius is specified in
  cells, so refining the mesh changes the physical problem unless the radius is
  scaled -- and even scaled, its discrete cone support jumps between kernel
  sizes and the clamp at small N pins it to a different physical width. An
  earlier version of this study kept the filter and produced a non-monotone J
  sequence, i.e. no asymptotic range and therefore no meaningful order. The
  filter is design regularisation, not physics; the properties here are mapped
  analytically instead.
* Every grid must actually converge. Extrapolating through a stalled solve
  measures the solver, not the discretisation, so non-converged trios are
  skipped rather than reported.

Richardson extrapolation on three grids in constant refinement ratio r gives

    p = ln( (J1 - J2) / (J2 - J3) ) / ln(r)

for the observed order, and an extrapolated J to compare the sequence against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from pipeline import ColdPlate, Params

def analytic_properties(N: int, p: Params):
    """Smooth grid-independent (alpha, k), mapped analytically.

    Same SIMP/RAMP formulas the material_map Tesseract uses, applied directly
    to a smooth density so the filter and projection -- both of which are
    grid-dependent regularisations -- stay out of a study of the PDE
    discretisation.
    """
    c = (np.arange(N) + 0.5) / N
    x, y = np.meshgrid(c, c, indexing="xy")
    rho = 0.5 + 0.22 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    k = p.k_fluid + (p.k_solid - p.k_fluid) * rho**p.penal
    alpha = p.alpha_max * rho / (1.0 + 8.0 * (1.0 - rho))
    return jnp.asarray(alpha), jnp.asarray(k)


def solve_at(N: int, Ra: float) -> dict:
    """Solve at the target Ra by continuation, warm-starting up the ramp.

    A cold start from T = 0 at the target Ra makes Newton stall on some grids,
    which would put non-converged values into a convergence study. Walking Ra
    up and reusing the previous solution as the initial guess fixes that; the
    converged state is the same either way.
    """
    # chip_frac = 1 heats the whole bottom wall. The chip mask is a binary test
    # (0.3N <= i < 0.7N), so a partial strip snaps to cell edges and the total
    # imposed heat varies non-monotonically with N -- 0.4375, 0.375, 0.40625,
    # ... -- which shows up as a non-monotone J and destroys the asymptotic
    # range. That is the boundary condition being measured, not the solver.
    p = Params(Nx=N, Ny=N, Ra=Ra, objective="domain_mean", chip_frac=1.0)
    alpha, k = analytic_properties(N, p)
    ramp = [Ra / 8.0, Ra / 4.0, Ra / 2.0, Ra]

    with ColdPlate(params=p) as cp:
        for step in ramp:
            cp.params.Ra = step
            T, info = cp.solve_coupled(alpha, k)
            if not info["ok"]:
                break
        J = float(cp.objective(T))
    return {"N": N, "J": J, "iters": info["iters"],
            "residual": info["residual"], "ok": bool(info["ok"])}


def main(Ra=3.0e3, out="results/grid_convergence.json"):
    grids = [16, 24, 32, 48, 64, 96]
    rows = []
    print(f"Ra = {Ra:.0e}, smooth analytic properties, filter bypassed "
          f"(objective: domain-mean temperature)\n")
    print(f"{'N':>5} {'h':>9} {'J':>14} {'fp iters':>9} {'residual':>11} {'ok':>5}")
    for N in grids:
        r = solve_at(N, Ra)
        rows.append(r)
        print(f"{N:>5} {1.0/N:9.5f} {r['J']:14.9f} {r['iters']:9d} "
              f"{r['residual']:11.2e} {str(r['ok']):>5}", flush=True)

    # Richardson on the grids that sit in a constant ratio of 2
    print()
    for trio in ((16, 32, 64), (24, 48, 96)):
        try:
            sel = [next(r for r in rows if r["N"] == n) for n in trio]
        except StopIteration:
            continue
        # Refuse to extrapolate through a solve that did not converge -- the
        # resulting "order of accuracy" would be an artefact of the solver
        # stalling, not a property of the discretisation.
        bad = [r["N"] for r in sel if not r["ok"]]
        if bad:
            print(f"  {trio}: skipped, fixed point did not converge at N={bad}")
            continue
        J = [r["J"] for r in sel]
        d1, d2 = J[0] - J[1], J[1] - J[2]
        if d2 == 0 or d1 / d2 <= 0:
            print(f"  {trio}: not in the asymptotic range (differences "
                  f"{d1:.2e}, {d2:.2e})")
            continue
        p_obs = np.log(d1 / d2) / np.log(2.0)
        J_ext = J[2] + d2 / (2.0**p_obs - 1.0)
        err = abs(J[2] - J_ext) / abs(J_ext)
        print(f"  grids {trio}: observed order p = {p_obs:.3f}, "
              f"extrapolated J = {J_ext:.9f}, finest-grid error = {100*err:.3f}%")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # 3e3 rather than 1e4: the study needs every grid to reach a converged
    # steady state, and at 1e4 Newton stalls on some of them from a cold start.
    # The observed order is a property of the discretisation, not of how hard
    # the fixed point is, so there is nothing to gain from pushing Ra here.
    ap.add_argument("--Ra", type=float, default=3.0e3)
    ap.add_argument("--out", default="results/grid_convergence.json")
    main(**vars(ap.parse_args()))
