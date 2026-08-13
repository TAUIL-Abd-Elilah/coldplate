# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Measure coupling strength vs. failure of component-wise differentiation.

For a range of Rayleigh numbers, through the real Tesseracts:

  * rho(Phi_T), the spectral radius of the fixed-point Jacobian -- the gain of
    one trip around the coupling loop (power iteration using JVPs, which run
    forward through the C++ block and then the JAX block);
  * the error and cosine similarity of the naive one-way gradient against the
    composed one.

Within this sweep the two track each other: component-wise differentiation is
accurate to six digits when the loop gain is small and carries ~86% error when
it exceeds 1. It stays a descent direction throughout -- it does not become an
ascent direction at any point we measured.

The tracking is also specific to holding the design fixed and varying Ra. See
predict_error.py: across different designs the loop gain can order two states
backwards, and the quantity that actually predicts the error is the
*directional* gain ||Phi_T^T g|| / ||g||.

Writes coupling_sweep.json for figure 3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from pipeline import ColdPlate, Params


def spectral_radius(cp: ColdPlate, T_star, alpha, k, n_power=40):
    """Power-iterate Phi_T at the fixed point."""
    import jax.numpy as jnp

    phi_T = lambda T: cp.phi(T, alpha, k)  # noqa: E731
    rng = np.random.default_rng(1)
    v = jnp.asarray(rng.normal(size=T_star.shape))
    v = v / jnp.linalg.norm(v)
    lam = 0.0
    for _ in range(n_power):
        w = jax.jvp(phi_T, (T_star,), (v,))[1]
        nrm = float(jnp.linalg.norm(w))
        if nrm < 1e-300:
            return 0.0
        lam, v = nrm, w / nrm
    return lam


def main(N=16, out="results"):
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.25, 0.75, size=(N, N))
    rows = []

    print(f"{'Ra':>9} {'fp it':>6} {'rho(Phi_T)':>11} {'J':>9} "
          f"{'naive rel err':>14} {'cos':>8} {'wrong sign':>11}")
    for Ra in (1e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5):
        p = Params(Nx=N, Ny=N, Ra=Ra)
        with ColdPlate(params=p) as cp:
            try:
                res = cp.value_and_grad(rho)
            except RuntimeError as e:
                print(f"{Ra:9.0e}  adjoint failed: {e}", flush=True)
                continue
            if not res["info"]["ok"]:
                print(f"{Ra:9.0e}  fixed point did not converge "
                      f"(residual {res['info']['residual']:.1e})", flush=True)
                continue
            mat = cp.material(rho)
            sr = spectral_radius(cp, res["T"], mat["alpha"], mat["k"])
            # the strong naive baseline: full chain, feedback loop cut
            gf = cp.one_way_grad(rho)

            g, f = res["grad"].ravel(), gf.ravel()
            rel = float(np.linalg.norm(f - g) / np.linalg.norm(g))
            cos = float(g @ f / (np.linalg.norm(g) * np.linalg.norm(f)))
            flip = float(np.mean(np.sign(f) != np.sign(g)))
            rows.append({
                "Ra": float(Ra), "rho_phi": sr, "J": res["J"],
                "rel_err": rel, "cos": cos, "sign_flip_frac": flip,
                "fp_iters": res["info"]["iters"],
            })
            print(f"{Ra:9.0e} {res['info']['iters']:6d} {sr:11.4f} {res['J']:9.4f} "
                  f"{rel:14.4f} {cos:8.4f} {100*flip:10.0f}%", flush=True)

    Path(out).mkdir(parents=True, exist_ok=True)
    Path(out, "coupling_sweep.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}/coupling_sweep.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--out", default="results")
    main(**vars(ap.parse_args()))
