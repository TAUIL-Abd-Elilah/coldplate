# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""When does dropping inertia cost you? The same question, a second time.

This repository is about when a cheap approximation to a coupled system is
safe. Everywhere else the approximation under test is *cutting the feedback
loop*. Here it is a different one, on a different axis: modelling the fluid as
Stokes flow rather than Navier-Stokes -- dropping (u.grad)u.

That approximation is not a detail. It is the reason the fluid block was linear
in w, which is what made a single sparse factorisation serve the solve, the
tangent and the adjoint. Turning inertia on costs a Newton iteration in the
forward solve and a refactorisation for the derivatives, so it is worth knowing
when it buys anything.

Two knobs govern the answer and they pull in opposite directions:

* the Prandtl number, because in this scaling the convective term is weighted
  by roughly 1/Pr -- water (Pr = 7) suppresses inertia, air (Pr = 0.71) does
  not;
* the design density, because Brinkman drag is a linear sink that damps exactly
  the velocities inertia feeds on. A solid-heavy design is its own
  regularisation.

For each configuration we solve the coupled fixed point both ways and report
the change in the flow, in the objective, and -- the quantity that actually
matters for optimisation -- in the design gradient.

Usage:  python inertia_study.py [--N 16]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline import ColdPlate, Params


def relative(a, b):
    """||a - b|| / ||b||, guarded."""
    a, b = np.asarray(a), np.asarray(b)
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float("nan")


def run_case(N, Ra, Pr, rho_mean, seed=0, verbose=False, spread=0.25):
    rng = np.random.default_rng(seed)
    # Heterogeneous, not near-uniform. A nearly constant intermediate density
    # is the worst case for this coupling -- nothing damps the flow and
    # convection cells span the cavity, so the loop gain sits near one and the
    # adjoint Krylov solve stalls. That is documented behaviour of this problem
    # (see ColdPlate.solve_coupled), not something to rediscover per study.
    lo = max(0.0, rho_mean - spread)
    hi = min(1.0, rho_mean + spread)
    rho = rng.uniform(lo, hi, size=(N, N))

    out = {}
    for inertia in (0.0, 1.0):
        p = Params(Nx=N, Ny=N, Ra=Ra, Pr=Pr, inertia=inertia)
        with ColdPlate(params=p, verbose=verbose) as cp:
            # An open cavity at high Rayleigh number may simply have no steady
            # state, in which case the coupled solve stalls and the adjoint
            # GMRES raises. That is a fact about the physics, not a failure of
            # the study: report it and move on rather than losing every other
            # configuration to one exception.
            try:
                res = cp.value_and_grad(rho)
            except Exception:  # noqa: BLE001 - any non-convergence
                return None
            if not res["info"]["ok"]:
                return None
            out[inertia] = {
                "J": res["J"],
                "grad": np.asarray(res["grad"]),
                "u": np.asarray(res["u"]),
                "v": np.asarray(res["v"]),
            }

    stokes, ns = out[0.0], out[1.0]
    speed_s = np.sqrt(np.mean(stokes["u"] ** 2) + np.mean(stokes["v"] ** 2))
    g_s, g_n = stokes["grad"].ravel(), ns["grad"].ravel()
    cos = float(g_s @ g_n / (np.linalg.norm(g_s) * np.linalg.norm(g_n)))
    return {
        "Ra": Ra,
        "Pr": Pr,
        "rho_mean": rho_mean,
        "rms_speed_stokes": float(speed_s),
        "J_stokes": stokes["J"],
        "J_navier_stokes": ns["J"],
        "J_rel_change": abs(ns["J"] - stokes["J"]) / max(abs(stokes["J"]), 1e-30),
        "flow_rel_change": relative(
            np.concatenate([ns["u"].ravel(), ns["v"].ravel()]),
            np.concatenate([stokes["u"].ravel(), stokes["v"].ravel()]),
        ),
        "grad_rel_change": relative(g_n, g_s),
        "grad_cosine": cos,
        "grad_sign_flip": float(np.mean(np.sign(g_n) != np.sign(g_s))),
    }


def main(N: int = 16, out: str = "results/inertia_study.json") -> int:
    # Ordered so the story is readable: the design regime first, then the
    # conditions that make inertia matter.
    cases = [
        # (Ra, Pr, mean density) -- what it represents
        (3.0e4, 7.0, 0.50),   # the gradient-study operating point, water
        (3.0e4, 0.71, 0.50),  # same, air
        (1.0e5, 0.71, 0.35),  # optimiser's volume fraction, hotter
        (1.0e5, 0.71, 0.10),  # open cavity
        (3.0e5, 0.71, 0.05),  # open cavity, strongly driven
    ]

    print(f"grid {N}x{N}; solving each configuration twice, Stokes and "
          f"Navier-Stokes\n")
    header = (f"{'Ra':>8} {'Pr':>6} {'rho':>6} {'rms |u|':>10} "
              f"{'flow chg':>10} {'J chg':>10} {'grad chg':>10} {'cos':>9}")
    print(header)
    print("-" * len(header))

    rows = []
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for Ra, Pr, rho_mean in cases:
        r = run_case(N, Ra, Pr, rho_mean)
        if r is None:
            print(f"{Ra:8.0e} {Pr:6.2f} {rho_mean:6.2f}   "
                  f"(no reachable steady state)", flush=True)
            continue
        rows.append(r)
        print(f"{Ra:8.0e} {Pr:6.2f} {rho_mean:6.2f} "
              f"{r['rms_speed_stokes']:10.2f} "
              f"{100*r['flow_rel_change']:9.2f}% "
              f"{100*r['J_rel_change']:9.3f}% "
              f"{100*r['grad_rel_change']:9.2f}% "
              f"{r['grad_cosine']:+9.5f}", flush=True)
        # Checkpoint after every configuration. Each one is two full coupled
        # solves plus two adjoints, and the open-cavity cases take minutes, so
        # an interrupted run must not lose the ones already paid for.
        Path(out).write_text(json.dumps(rows, indent=2))

    Path(out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")

    if rows:
        worst = max(rows, key=lambda r: r["grad_rel_change"])
        best = min(rows, key=lambda r: r["grad_rel_change"])
        print(f"\ngradient change from dropping inertia ranges "
              f"{100*best['grad_rel_change']:.3f}% "
              f"(Ra={best['Ra']:.0e}, Pr={best['Pr']}, rho={best['rho_mean']}) "
              f"to {100*worst['grad_rel_change']:.1f}% "
              f"(Ra={worst['Ra']:.0e}, Pr={worst['Pr']}, "
              f"rho={worst['rho_mean']})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--out", default="results/inertia_study.json")
    raise SystemExit(main(**vars(ap.parse_args())))
