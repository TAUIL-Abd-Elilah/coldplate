# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""What actually predicts whether you can skip composing across the boundary?

The Rayleigh sweep suggested the coupling loop gain rho(Phi_T) governs how
wrong a component-wise gradient is. Across *different designs* it does not:
two states with essentially the same loop gain (0.759 and 0.769) differ
ten-fold in error. So rho is not the right statistic, and the implicit function
theorem says why.

The exact adjoint solves (I - Phi_T)^T lambda = g. Cutting the feedback loop
uses lambda_0 = g, whose residual in that equation is exactly Phi_T^T g. This
depends on the *direction* g, the objective's own sensitivity to the coupled
state. rho(Phi_T) is an objective-blind asymptotic modal rate and does not
encode how g aligns with the operator's modes. The residual is well-defined
even when rho(Phi_T) >= 1, where a Neumann expansion would not converge.

That suggests a directional gain,

    gamma = || Phi_T^T g || / || g ||

which costs exactly one VJP -- far less than the gradient it is screening. It
is a residual diagnostic, not a universal error bound. This
script measures gamma, rho(Phi_T) and the actual naive-gradient error across a
range of designs and Rayleigh numbers, and asks which one predicts the error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter

from pipeline import ColdPlate, Params
from sweep_coupling import spectral_radius


def directional_gain(cp: ColdPlate, T_star, alpha, k) -> float:
    """gamma = ||Phi_T^T g|| / ||g||, one VJP through both physics blocks."""
    phi = lambda T: cp.phi(T, alpha, k)  # noqa: E731
    _, vjp_fn = jax.vjp(phi, T_star)
    g = jax.grad(cp.objective)(T_star)
    (w,) = vjp_fn(g)
    return float(jnp.linalg.norm(w) / jnp.linalg.norm(g))


def designs(N: int, rng):
    """A spread of designs that differ in smoothness but not in statistics.

    Every field is normalised to the same mean and range as the rough one, so
    the only thing varying across the family is spatial correlation length.
    An earlier version let the smooth fields drift fluid-heavy, which made them
    unsolvable at higher Ra and confounded smoothness with volume fraction.
    """
    def band(z):
        z = (z - z.mean()) / (z.std() + 1e-12)
        return np.clip(0.5 + 0.15 * z, 0.25, 0.75)

    out = {"rough": band(rng.normal(size=(N, N)))}
    for sigma, name in ((1.5, "smooth"), (3.0, "very_smooth")):
        out[name] = band(gaussian_filter(rng.normal(size=(N, N)), sigma))
    out["near_uniform"] = np.clip(0.5 + 0.03 * rng.normal(size=(N, N)), 0.25, 0.75)
    return out


def main(N=20, out="results/predict_error.json"):
    rng = np.random.default_rng(0)
    ds = designs(N, rng)
    rows = []

    print(f"{'design':>13} {'Ra':>8} {'rho(Phi_T)':>11} {'gamma':>9} "
          f"{'naive err':>10} {'cos':>8}")
    for name, rho in ds.items():
        for Ra in (1.0e3, 3.0e3, 1.0e4, 2.0e4, 3.0e4):
            p = Params(Nx=N, Ny=N, Ra=Ra)
            try:
                with ColdPlate(params=p) as cp:
                    res = cp.value_and_grad(rho)
                    if not res["info"]["ok"]:
                        print(f"{name:>13} {Ra:8.0e}   fixed point did not converge")
                        continue
                    mat = cp.material(rho)
                    T = jnp.asarray(res["T"])
                    sr = spectral_radius(cp, T, mat["alpha"], mat["k"])
                    gam = directional_gain(cp, T, mat["alpha"], mat["k"])
                    gn = cp.one_way_grad(rho)
            except Exception as exc:  # noqa: BLE001
                print(f"{name:>13} {Ra:8.0e}   failed: {type(exc).__name__}")
                continue

            a, b = res["grad"].ravel(), gn.ravel()
            rel = float(np.linalg.norm(b - a) / np.linalg.norm(a))
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            rows.append({"design": name, "Ra": Ra, "rho_phi": sr, "gamma": gam,
                         "rel_err": rel, "cos": cos})
            print(f"{name:>13} {Ra:8.0e} {sr:11.4f} {gam:9.4f} {rel:10.4f} {cos:8.4f}",
                  flush=True)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows, indent=2))

    # A high in-sample correlation from fourteen points deserves a stability
    # check. Ship leave-one-design-family-out and seeded bootstrap statistics
    # beside the raw measurements every time this experiment runs.
    if len(rows) > 2:
        from predictor_statistics import summarize

        stats_path = Path(out).with_name("predictor_statistics.json")
        stats_path.write_text(json.dumps(summarize(rows), indent=2))

    if len(rows) > 2:
        r = np.array([x["rel_err"] for x in rows])
        sr = np.array([x["rho_phi"] for x in rows])
        gm = np.array([x["gamma"] for x in rows])
        keep = r > 0
        lr = np.log10(np.maximum(r[keep], 1e-12))
        print("\ncorrelation with log10(naive error):")
        print(f"  rho(Phi_T)      {np.corrcoef(sr[keep], lr)[0,1]:+.4f}")
        print(f"  gamma           {np.corrcoef(gm[keep], lr)[0,1]:+.4f}")
        print(f"  log10(gamma)    {np.corrcoef(np.log10(np.maximum(gm[keep],1e-12)), lr)[0,1]:+.4f}")
    print(f"\nwrote {out}")
    if len(rows) > 2:
        print(f"wrote {stats_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--out", default="results/predict_error.json")
    main(**vars(ap.parse_args()))
