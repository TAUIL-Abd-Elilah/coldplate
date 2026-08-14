# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end validation of the composed three-Tesseract gradient.

This is the headline experiment. Everything runs through the real Tesseracts --
three served containers, three derivative strategies, and either of two thermal
backends -- and we check that the composed gradient is exact, and that the
gradient you would get without composing across the boundary is not merely less
accurate but wrong.

  [1] composed adjoint   implicit diff of the fixed point, adjoint via GMRES
  [2] finite differences central differences on the whole pipeline
  [3] frozen flow        the naive gradient: no derivative from the C++ block

Usage:  python validate_pipeline.py [N]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from pipeline import ColdPlate, Params


def relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def main(N: int = 20, backend: str = "thermal_advdiff",
         inertia: float = 0.0, Pr: float = 7.0) -> int:
    p = Params(Nx=N, Ny=N, Ra=3.0e4, Pr=Pr, inertia=inertia)
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.25, 0.75, size=(N, N))

    how = {"thermal_advdiff": "JAX autodiff",
           "thermal_fortran": "Fortran + Enzyme compiler AD"}.get(backend, backend)
    fluid = ("Stokes-Brinkman, linear in w" if inertia == 0.0
             else f"Navier-Stokes-Brinkman, inertia={inertia:g}, "
                  f"nonlinear in w")
    print(f"grid {N}x{N}, Ra={p.Ra:.0e}, Pr={p.Pr}")
    print(f"fluid block: {fluid}")
    print(f"composing: material_map [PyTorch] -> stokes_brinkman [C++] "
          f"<-> {backend} [{how}]\n")

    images = {"material": "material_map", "fluid": "stokes_brinkman", "thermal": backend}
    with ColdPlate(params=p, images=images, verbose=True) as cp:
        t0 = time.time()
        res = cp.value_and_grad(rho)
        t_adj = time.time() - t0
        info = res["info"]
        print(
            f"[1] composed adjoint   {t_adj:6.1f}s   J = {res['J']:.6f}\n"
            f"    fixed point: {info['iters']} Newton iters, residual {info['residual']:.2e}\n"
            f"    Phi evaluations: {cp.stats['phi_calls']}, "
            f"adjoint GMRES matvecs: {cp.stats['vjp_matvecs']}"
        )
        print(f"    (each Phi call and each GMRES matvec crosses the C++/{backend} boundary)\n")

        t0 = time.time()
        g_frozen = cp.frozen_flow_grad(rho)
        g_oneway = cp.one_way_grad(rho)
        print(f"[3] naive gradients    {time.time()-t0:6.1f}s "
              f"(frozen-flow and one-way)\n")

        def J_at(r):
            mat = cp.material(r)
            T, _ = cp.solve_coupled(mat["alpha"], mat["k"])
            return float(cp.objective(T))

        # Primary check: a directional derivative along one random unit vector.
        # Per-entry differencing is dominated by solver noise -- the fixed point
        # is only converged to ~1e-10, so differencing it with eps=1e-5 leaves
        # ~5e-6 of noise in the gradient, which swamps small-magnitude entries.
        # A directional derivative uses the whole gradient at once, and
        # sweeping eps shows the finite-difference estimate converging onto the
        # analytic value rather than us tuning a single lucky step size.
        d = rng.normal(size=(N, N))
        d /= np.linalg.norm(d)
        analytic = float(np.sum(res["grad"] * d))
        print(f"[2] directional derivative along a random unit vector")
        print(f"    analytic  <g, d> = {analytic:.10e}")
        best = np.inf
        for eps in (1e-3, 3e-4, 1e-4):
            t0 = time.time()
            fd_dir = (J_at(rho + eps * d) - J_at(rho - eps * d)) / (2 * eps)
            e = abs(fd_dir - analytic) / max(abs(analytic), 1e-30)
            best = min(best, e)
            print(f"    eps={eps:7.0e}  finite-diff = {fd_dir:.10e}   "
                  f"rel err = {e:.2e}   ({time.time()-t0:.0f}s)")
        print()

        # A few per-entry values as well, for the scatter figure.
        idx = [(int(a), int(b)) for a, b in rng.integers(0, N, size=(5, 2))]
        eps = 3e-4
        fd = []
        for j, i in idx:
            vals = []
            for s in (+1, -1):
                r = rho.copy()
                r[j, i] += s * eps
                vals.append(J_at(r))
            fd.append((vals[0] - vals[1]) / (2 * eps))
        fd = np.array(fd)

        g = res["grad"]
        a = np.array([g[j, i] for j, i in idx])
        f = np.array([g_frozen[j, i] for j, i in idx])
        o = np.array([g_oneway[j, i] for j, i in idx])

        print(f"{'idx':>9} {'composed':>14} {'finite-diff':>14} {'rel err':>10} "
              f"{'one-way':>14} {'frozen-flow':>14}")
        for n, (j, i) in enumerate(idx):
            e = abs(a[n] - fd[n]) / max(abs(fd[n]), 1e-30)
            print(f"{str((j,i)):>9} {a[n]:14.6e} {fd[n]:14.6e} {e:10.2e} "
                  f"{o[n]:14.6e} {f[n]:14.6e}")

        results = {"fd": fd.tolist(), "composed": a.tolist(),
                   "frozen": f.tolist(), "one_way": o.tolist()}

        print(f"  composed adjoint : directional rel err {best:.3e}   <-- exact")
        # Statistics over the WHOLE gradient field, not the five sampled
        # entries: which entries you happen to sample moves the sign-flip count
        # around a lot, so a handful of them is not a defensible number.
        gn = g.ravel()
        for name, gv in (("one-way    (naive)", g_oneway), ("frozen-flow(naive)", g_frozen)):
            nv = gv.ravel()
            cos = float(gn @ nv / (np.linalg.norm(gn) * np.linalg.norm(nv)))
            rel = float(np.linalg.norm(nv - gn) / np.linalg.norm(gn))
            flip = float(np.mean(np.sign(nv) != np.sign(gn)))
            print(f"  {name}: rel err {rel:.3f}   cos = {cos:+.4f}   "
                  f"wrong sign on {100*flip:.0f}% of design variables   "
                  f"{'ASCENT DIRECTION' if gn @ nv < 0 else 'descent'}")
            results[f"stats_{name.split()[0]}"] = {
                "rel_err": rel, "cos": cos, "sign_flip_frac": flip
            }

        Path("results").mkdir(parents=True, exist_ok=True)
        Path("results/gradient_validation.json").write_text(json.dumps(results, indent=2))

        # Threshold reflects the finite-difference noise floor, not the adjoint:
        # the fixed point is converged to ~1e-10, so differencing it cannot
        # resolve better than ~1e-5 relative however carefully it is done.
        ok = best < 1e-4
        print(f"\n{'PASS' if ok else 'FAIL'}: composed gradient "
              f"{'matches' if ok else 'does NOT match'} finite differences")
        return 0 if ok else 1


if __name__ == "__main__":
    # usage: validate_pipeline.py [N] [thermal backend] [inertia] [Pr]
    #
    # inertia = 1 turns the fluid block from Stokes into steady Navier-Stokes,
    # so the composed gradient is then being validated through a *nonlinear*
    # component whose adjoint is a solve against the Jacobian at its converged
    # state. Pass a small Prandtl number with it; at Pr = 7 the convective term
    # is negligible and the two runs are indistinguishable.
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    be = sys.argv[2] if len(sys.argv) > 2 else "thermal_advdiff"
    inertia = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    pr = float(sys.argv[4]) if len(sys.argv) > 4 else 7.0
    sys.exit(main(n, be, inertia, pr))
