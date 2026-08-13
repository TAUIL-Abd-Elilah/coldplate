# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Where component-wise differentiation goes wrong, as coupling strengthens.

Holds one design fixed and raises the Rayleigh number, recording the exact
gradient, the naive one, and where they disagree in sign. This is the honest
form of the claim: the damage is regime-dependent, so the interesting picture
is not a single snapshot but the progression.

Writes gradient_maps.npz for figure 7.
"""

from __future__ import annotations

import argparse

import numpy as np

from pipeline import ColdPlate, Params


def main(N=24, out="results/gradient_maps.npz"):
    rho = np.random.default_rng(0).uniform(0.25, 0.75, size=(N, N))
    Ras = [1.0e3, 1.0e4, 3.0e4]

    rows = []
    print(f"{'Ra':>9} {'converged':>10} {'loop gain':>10} {'rel err':>9} "
          f"{'cos':>8} {'wrong sign':>11}")
    for Ra in Ras:
        p = Params(Nx=N, Ny=N, Ra=Ra)
        with ColdPlate(params=p) as cp:
            res = cp.value_and_grad(rho)
            gn = cp.one_way_grad(rho)
            mat = cp.material(rho)
            from sweep_coupling import spectral_radius

            sr = spectral_radius(cp, res["T"], mat["alpha"], mat["k"])

        ge = res["grad"]
        a, b = ge.ravel(), gn.ravel()
        rel = float(np.linalg.norm(b - a) / np.linalg.norm(a))
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        flip = float(np.mean(np.sign(a) != np.sign(b)))
        print(f"{Ra:9.0e} {str(res['info']['ok']):>10} {sr:10.4f} {rel:9.4f} "
              f"{cos:8.4f} {100*flip:10.0f}%", flush=True)

        rows.append({
            "Ra": Ra, "g_exact": ge, "g_naive": gn, "T": res["T"],
            "rho_phys": res["rho_phys"], "loop_gain": sr, "rel": rel,
            "cos": cos, "flip": flip, "ok": res["info"]["ok"],
        })

    np.savez_compressed(
        out,
        Ra=np.array([r["Ra"] for r in rows]),
        g_exact=np.stack([r["g_exact"] for r in rows]),
        g_naive=np.stack([r["g_naive"] for r in rows]),
        rho_phys=np.stack([r["rho_phys"] for r in rows]),
        loop_gain=np.array([r["loop_gain"] for r in rows]),
        rel=np.array([r["rel"] for r in rows]),
        cos=np.array([r["cos"] for r in rows]),
        flip=np.array([r["flip"] for r in rows]),
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=24)
    ap.add_argument("--out", default="results/gradient_maps.npz")
    main(**vars(ap.parse_args()))
