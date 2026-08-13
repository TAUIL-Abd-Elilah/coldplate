# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Is the error of a component-wise gradient directional, or spectral?

This is the decisive test of the claim, and it works by holding the physics
completely fixed and changing only what is being measured.

At one design and one Rayleigh number there is a single coupled state, a single
Phi, and therefore a single spectral radius rho(Phi_T). If rho governed the
error of a component-wise gradient, every objective evaluated at that state
would suffer the same error -- rho cannot distinguish between them.

The directional gain can:

    gamma = || Phi_T^T g || / || g ||,     g = dJ/dT

because g differs from objective to objective. So: sweep the objective, keep
everything else identical, and see which statistic moves with the error.

rho is constant across every row of this table by construction. Any variation
in the error is variation rho is structurally blind to.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pipeline import ColdPlate, Params
from sweep_coupling import spectral_radius

OBJECTIVES = [
    "chip_mean",
    "chip_peak",
    "domain_mean",
    "top_half_mean",
    "left_column_mean",
    "outlet_mean",
]


def main(N=20, Ra=1.0e4, out="results/objective_sweep.json"):
    rho = np.random.default_rng(0).uniform(0.25, 0.75, size=(N, N))
    rows = []
    rho_phi = None

    print(f"grid {N}x{N}, Ra={Ra:.0e}, one fixed design and one fixed coupled state.")
    print("rho(Phi_T) is a property of the state alone, so it is identical on "
          "every row.\n")
    print(f"{'objective':>18} {'gamma':>10} {'naive rel err':>14} {'cos':>9} "
          f"{'wrong sign':>11}")

    for obj in OBJECTIVES:
        p = Params(Nx=N, Ny=N, Ra=Ra, objective=obj)
        with ColdPlate(params=p) as cp:
            res = cp.value_and_grad(rho)
            if not res["info"]["ok"]:
                print(f"{obj:>18}   fixed point did not converge")
                continue
            mat = cp.material(rho)
            T = jnp.asarray(res["T"])

            if rho_phi is None:  # same for every objective; measure once
                rho_phi = spectral_radius(cp, T, mat["alpha"], mat["k"])

            phi = lambda t: cp.phi(t, mat["alpha"], mat["k"])  # noqa: E731
            _, vjp_fn = jax.vjp(phi, T)
            g = jax.grad(cp.objective)(T)
            (w,) = vjp_fn(g)
            gamma = float(jnp.linalg.norm(w) / jnp.linalg.norm(g))

            gn = cp.one_way_grad(rho)

        a, b = res["grad"].ravel(), gn.ravel()
        rel = float(np.linalg.norm(b - a) / np.linalg.norm(a))
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        flip = float(np.mean(np.sign(a) != np.sign(b)))
        rows.append({"objective": obj, "gamma": gamma, "rel_err": rel,
                     "cos": cos, "sign_flip": flip, "rho_phi": rho_phi})
        print(f"{obj:>18} {gamma:10.4f} {rel:14.4f} {cos:9.4f} {100*flip:10.0f}%",
              flush=True)

    if rows:
        g = np.array([r["gamma"] for r in rows])
        e = np.array([r["rel_err"] for r in rows])
        print(f"\nrho(Phi_T) = {rho_phi:.4f} for every row above (variation: none, "
              f"by construction)")
        print(f"gamma spans {g.min():.4f} to {g.max():.4f}  "
              f"({g.max()/max(g.min(),1e-12):.1f}x)")
        print(f"error spans {e.min():.4f} to {e.max():.4f}  "
              f"({e.max()/max(e.min(),1e-12):.1f}x)")
        if len(rows) > 2:
            c = np.corrcoef(np.log10(np.maximum(g, 1e-12)),
                            np.log10(np.maximum(e, 1e-12)))[0, 1]
            print(f"correlation log(gamma) vs log(error): {c:+.4f}")
            print("correlation log(rho)   vs log(error): undefined "
                  "(rho is constant)")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--Ra", type=float, default=1.0e4)
    ap.add_argument("--out", default="results/objective_sweep.json")
    main(**vars(ap.parse_args()))
