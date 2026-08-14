# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Validate nonlinear natural convection against de Vahl Davis (1983).

The square cavity has no-slip walls, Pr=0.71, a hot left wall (T=1), a cold
right wall (T=0), and adiabatic horizontal walls. Unlike the critical-Rayleigh
check, this exercises the fully nonlinear Navier-Stokes path (inertia=1).

Reference values are from G. de Vahl Davis, *Natural convection of air in a
square cavity: a bench mark numerical solution*, Int. J. Numer. Meth. Fluids
3 (1983), 249-264, Table 2. The finite-volume grid here is deliberately much
coarser than the extrapolated benchmark. Therefore this script reports errors;
it only marks a case "within coarse-grid tolerance" when all three primary
metrics are within 15% at N>=32. It never hides the raw values or hard-fails a
coarse exploratory run.

Usage: python benchmark_de_vahl_davis.py --N 32 --Ra 1000 10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from tesseract_jax import apply_tesseract

from pipeline import ColdPlate, Params


REFERENCE = {
    1.0e3: {"Nu_mean": 1.118, "u_max": 3.649, "v_max": 3.697},
    1.0e4: {"Nu_mean": 2.243, "u_max": 16.178, "v_max": 19.617},
}


def cavity_metrics(T, u, v) -> dict[str, float]:
    """de Vahl Davis observables for a cell-centred/MAC-grid state."""
    T, u, v = np.asarray(T), np.asarray(u), np.asarray(v)
    Ny, Nx = T.shape
    h = 1.0 / Nx
    # Wall-gradient Nusselt number, integrated along the unit-height hot wall.
    nu_mean = float(np.mean(2.0 * (1.0 - T[:, 0]) / h))
    u_mid = u[:, Nx // 2]
    v_mid = v[Ny // 2, :]
    ju, iv = int(np.argmax(u_mid)), int(np.argmax(v_mid))
    return {
        "Nu_mean": nu_mean,
        "u_max": float(u_mid[ju]),
        "u_max_y": float((ju + 0.5) / Ny),
        "v_max": float(v_mid[iv]),
        "v_max_x": float((iv + 0.5) / Nx),
    }


def fluid_solver_status(flow) -> dict[str, float | int | bool]:
    """Extract the fail-closed nonlinear diagnostics exported by the fluid block."""
    residual = float(np.asarray(flow["nonlinear_residual"]))
    converged = bool(round(float(np.asarray(flow["nonlinear_converged"]))))
    iterations = int(round(float(np.asarray(flow["nonlinear_iterations"]))))
    return {
        "converged": bool(converged and np.isfinite(residual)),
        "relative_residual": residual,
        "newton_iterations": iterations,
    }


def run_case(N: int, Ra: float, verbose: bool = False) -> dict:
    p = Params(
        Nx=N, Ny=N, Ra=Ra, Pr=0.71, inertia=1.0,
        bc_mode=2.0, t_hot=1.0,
    )
    alpha = jnp.zeros((N, N))
    k = jnp.ones((N, N))
    with ColdPlate(params=p, verbose=verbose) as cp:
        # The exact no-flow conduction field is a much better nonlinear start
        # than the cold zero field used for topology-optimisation runs.
        x = (np.arange(N) + 0.5) / N
        cp._T_warm = jnp.asarray(np.tile(1.0 - x, (N, 1)))
        T, info = cp.solve_coupled(alpha, k, tol=1e-9, max_newton=40)
        flow = apply_tesseract(
            cp._t["fluid"],
            {"alpha": alpha, "T": T, "Ra": Ra, "Pr": p.Pr, "inertia": 1.0},
        )
        metrics = cavity_metrics(T, flow["u"], flow["v"])
        fluid_info = fluid_solver_status(flow)

    ref = REFERENCE[float(Ra)]
    errors = {name: abs(metrics[name] / ref[name] - 1.0) for name in ref}
    solver = dict(info)
    solver["coupled_converged"] = bool(info["ok"])
    solver["fluid"] = fluid_info
    # Preserve the existing top-level `ok` key, but strengthen its meaning:
    # both the outer temperature fixed point and the inner nonlinear momentum
    # solve must have converged.
    solver["ok"] = bool(info["ok"] and fluid_info["converged"])
    return {
        "N": N,
        "Ra": Ra,
        "Pr": p.Pr,
        "inertia": p.inertia,
        "solver": solver,
        **metrics,
        "reference": ref,
        "relative_error": errors,
        "coarse_grid_tolerance": 0.15,
        "within_coarse_grid_tolerance": (
            solver["ok"] and N >= 32 and max(errors.values()) <= 0.15
        ),
    }


def main(N=32, Ra=(1.0e3, 1.0e4), out="results/de_vahl_davis.json", verbose=False):
    rows = []
    print(f"de Vahl Davis cavity; grid {N}x{N}, Pr=0.71, inertia=1")
    print(f"{'Ra':>8} {'Nu':>9} {'err':>8} {'u_max':>9} {'err':>8} "
          f"{'v_max':>9} {'err':>8} {'solve':>9}")
    for ra in Ra:
        if float(ra) not in REFERENCE:
            raise ValueError(f"no stored de Vahl Davis reference for Ra={ra:g}")
        row = run_case(N, float(ra), verbose)
        rows.append(row)
        e = row["relative_error"]
        solve = "ok" if row["solver"]["ok"] else "stalled"
        print(f"{ra:8.0f} {row['Nu_mean']:9.3f} {100*e['Nu_mean']:7.2f}% "
              f"{row['u_max']:9.3f} {100*e['u_max']:7.2f}% "
              f"{row['v_max']:9.3f} {100*e['v_max']:7.2f}% {solve:>9}",
              flush=True)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))
    print(f"wrote {path}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--Ra", type=float, nargs="+", default=[1.0e3, 1.0e4])
    ap.add_argument("--out", default="results/de_vahl_davis.json")
    ap.add_argument("--verbose", action="store_true")
    main(**vars(ap.parse_args()))
