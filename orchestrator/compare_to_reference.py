# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Differential test: composed Tesseracts vs the monolithic JAX reference.

The reference implementation in prototype/reference_jax.py is independently
validated (energy balance to 1e-15, adjoint to 1e-8 against finite
differences). If the containerised composition disagrees with it on the same
inputs, the bug is in the Tesseracts or the wiring, not the physics.

Compares block by block so a discrepancy localises immediately:
    fluid    alpha, T   -> u, v
    thermal  u, v, k    -> T
    phi      T          -> T          (one full trip round the loop)
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prototype"))

from pipeline import ColdPlate, Params  # noqa: E402
from reference_jax import (  # noqa: E402
    Config,
    material_maps,
    solve_fluid,
    solve_thermal,
)
from reference_jax import coupled_step as ref_phi  # noqa: E402


def relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def main(N=16):
    Ra, Pr = 3.0e4, 7.0
    cfg = Config(Nx=N, Ny=N, Ra=Ra, Pr=Pr)
    p = Params(Nx=N, Ny=N, Ra=Ra, Pr=Pr)

    rng = np.random.default_rng(0)
    rho = jnp.asarray(rng.uniform(0.25, 0.75, size=(N, N)))
    # bypass material_map so both sides see identical properties
    k, alpha = material_maps(rho, cfg)
    T0 = jnp.asarray(rng.normal(size=(N, N)) * 0.5)

    print(f"grid {N}x{N}, Ra={Ra:.0e}\n")
    print(f"alpha range [{float(alpha.min()):.4g}, {float(alpha.max()):.4g}]")
    print(f"k     range [{float(k.min()):.4g}, {float(k.max()):.4g}]\n")

    with ColdPlate(params=p) as cp:
        from tesseract_jax import apply_tesseract

        # ---- fluid block ----
        flow = apply_tesseract(cp._t["fluid"], {"alpha": alpha, "T": T0, "Ra": Ra, "Pr": Pr})
        u_ref, v_ref, _ = solve_fluid(T0, alpha, cfg)
        print(f"[fluid  ] u rel err {relerr(flow['u'], u_ref):.3e}   "
              f"v rel err {relerr(flow['v'], v_ref):.3e}")
        print(f"[fluid  ] max|u| tess {float(jnp.abs(flow['u']).max()):.6g}  "
              f"ref {float(jnp.abs(u_ref).max()):.6g}")

        # ---- thermal block, fed the REFERENCE velocities so it is isolated ----
        th = apply_tesseract(
            cp._t["thermal"],
            {"u": u_ref, "v": v_ref, "k": k, "q_chip": p.q_chip, "chip_frac": p.chip_frac},
        )
        T_ref = solve_thermal(u_ref, v_ref, k, cfg)
        print(f"[thermal] T rel err {relerr(th['T'], T_ref):.3e}")
        print(f"[thermal] max T tess {float(jnp.abs(th['T']).max()):.6g}  "
              f"ref {float(jnp.abs(T_ref).max()):.6g}")

        # ---- one full trip round the coupling loop ----
        phi_t = cp.phi(T0, alpha, k)
        phi_r = ref_phi(T0, cfg, rho)
        print(f"[phi    ] rel err {relerr(phi_t, phi_r):.3e}")

        # ---- and the converged fixed point ----
        T_star, info = cp.solve_coupled(alpha, k)
        print(f"\n[fixed point] iters={info['iters']} residual={info['residual']:.3e} "
              f"ok={info['ok']}")
        print(f"[fixed point] J = {float(cp.objective(T_star)):.6f}")

    worst = max(relerr(flow["u"], u_ref), relerr(th["T"], T_ref), relerr(phi_t, phi_r))
    print(f"\nworst block-level rel err: {worst:.3e}")
    print("PASS" if worst < 1e-9 else "FAIL")
    return 0 if worst < 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 16))
