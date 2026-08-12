# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Find the Rayleigh number at which the optimiser's designs stay well posed.

The gradient study uses a heterogeneous design where solid regions block the
flow. The optimiser instead *starts* from a near-uniform intermediate density,
which is the worst case for coupling strength: low Brinkman drag (little damping)
and low conductivity (high Peclet) at the same time. So the Ra that is
comfortable for the gradient study can still be intractable here.

Rather than guess, measure: for the optimiser's actual initial design, report
convergence and coupling loop gain as a function of Ra.
"""

from __future__ import annotations

import numpy as np

from pipeline import ColdPlate, Params
from sweep_coupling import spectral_radius


def main(N=16, seed=0):
    rng = np.random.default_rng(seed)
    vf = Params().volume_fraction
    rho = np.clip(vf + 0.05 * rng.normal(size=(N, N)), 0.0, 1.0)
    print(f"optimiser start: rho_raw mean {rho.mean():.3f}, "
          f"std {rho.std():.3f}  (grid {N}x{N})\n")

    print(f"{'Ra':>9} {'converged':>10} {'iters':>6} {'residual':>11} "
          f"{'rho(Phi_T)':>11} {'J':>9}")
    for Ra in (1e3, 3e3, 6e3, 1e4, 3e4):
        p = Params(Nx=N, Ny=N, Ra=Ra, beta=1.0)
        with ColdPlate(params=p) as cp:
            mat = cp.material(rho)
            T, info = cp.solve_coupled(mat["alpha"], mat["k"])
            sr = spectral_radius(cp, T, mat["alpha"], mat["k"]) if info["ok"] else float("nan")
            print(f"{Ra:9.0e} {str(info['ok']):>10} {info['iters']:6d} "
                  f"{info['residual']:11.2e} {sr:11.4f} "
                  f"{float(cp.objective(T)):9.4f}", flush=True)


if __name__ == "__main__":
    main()
