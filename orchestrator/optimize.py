# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Topology optimisation of the cold plate, driven by the composed gradient.

Minimise the mean chip temperature subject to a solid-material budget, with the
design field passed through filtering and projection so the result is a real
solid/fluid layout rather than grey mush.

Update rule is projected Adam: an Adam step on the raw design, then a bisection
on a uniform shift that restores the volume constraint. Beta continuation
sharpens the projection as the design settles, which is standard practice --
starting sharp freezes in whatever the initial guess happened to look like.

The only thing that makes this work is that dJ/drho is the *coupled* gradient.
Run with --frozen to drive the same loop with the naive component-wise gradient
and watch it fail.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pipeline import ColdPlate, Params
from sweep_coupling import spectral_radius


def cp_alpha_k(cp, rho_raw):
    """(alpha, k) for a design, for the loop-gain diagnostic."""
    mat = cp.material(rho_raw)
    return mat["alpha"], mat["k"]


def volume_project(cp: ColdPlate, rho_raw, target, n_bisect=25):
    """Shift the design uniformly until the projected volume hits the budget."""
    lo, hi = -1.0, 1.0

    def vol(shift):
        r = np.clip(rho_raw + shift, 0.0, 1.0)
        return float(np.mean(np.asarray(cp.material(r)["rho_phys"]))), r

    v_hi, _ = vol(hi)
    if v_hi <= target:  # cannot exceed the budget even fully solid
        return np.clip(rho_raw + hi, 0.0, 1.0)
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        v, _ = vol(mid)
        if v > target:
            hi = mid
        else:
            lo = mid
    return np.clip(rho_raw + 0.5 * (lo + hi), 0.0, 1.0)


def run(
    N=48,
    iters=120,
    lr=0.05,
    outdir="results",
    mode="composed",
    seed=0,
    snapshot_every=4,
    verbose=False,
    Ra=1.0e3,
    diagnose=0,
    gamma_gate=0.01,
    result_tag=None,
):
    # Ra = 1e3 by default, not the 3e4 used for the gradient study, and this is
    # measured rather than guessed (see probe_startpoint.py). A near-uniform
    # intermediate density -- exactly where topology optimisation has to start --
    # is the worst case for coupling: low Brinkman drag means nothing damps the
    # flow, and low conductivity means a high Peclet number, both at once. That
    # steady solver does not converge for this design above Ra ~ 1e3, whereas the
    # heterogeneous design used for the gradient study stays solvable to 3e5.
    #
    # This is not a weak-coupling cop-out: at Ra = 1e3 the starting design still
    # has a loop gain of 0.75, comparable to Ra = 3e5 in the gradient study.
    p = Params(Nx=N, Ny=N, Ra=Ra, beta=1.0)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    tag = result_tag or mode

    rng = np.random.default_rng(seed)
    # start from a mild perturbation of the volume fraction so the filter has
    # something to break symmetry with
    rho = np.clip(p.volume_fraction + 0.05 * rng.normal(size=(N, N)), 0.0, 1.0)

    m = np.zeros_like(rho)
    v = np.zeros_like(rho)
    b1, b2, eps = 0.9, 0.999, 1e-8

    history = []
    snaps = []
    grad_snaps = []

    with ColdPlate(params=p, verbose=verbose) as cp:
        for it in range(1, iters + 1):
            # beta continuation: 1 -> 8 over the run, doubling every 30 steps
            cp.params.beta = min(8.0, 1.0 * 2 ** (it // 30))

            t0 = time.time()
            # A served Tesseract occasionally drops its connection part-way
            # through a long run (observed once at iteration 27 of 120, with no
            # OOM and plenty of free memory). Re-serving the containers and
            # retrying the iteration is cheap insurance against losing an hour
            # of optimisation to a transient.
            mv_before = cp.stats["vjp_matvecs"]
            for attempt in range(3):
                try:
                    res = cp.value_and_grad(rho, mode=mode, gamma_gate=gamma_gate)
                    break
                except Exception as exc:  # noqa: BLE001 - any transport failure
                    if attempt == 2:
                        raise
                    print(f"  [warn] iteration {it} failed ({type(exc).__name__}: "
                          f"{str(exc)[:80]}); re-serving Tesseracts and retrying",
                          flush=True)
                    cp.__exit__(None, None, None)
                    cp._T_warm = None  # stale warm start after a restart
                    cp.__enter__()
            # Keep the unmodified gradient: the diagnostics below must compare
            # like with like, and the optimiser is about to project this one.
            g_raw = res["grad"]
            rho_prev = rho.copy()  # design this gradient belongs to

            # Project the gradient onto the volume-preserving subspace before
            # stepping. Without this the optimiser stalls: Adam's per-coordinate
            # normalisation makes every step roughly +-lr, so when the gradient
            # has a consistent sign the step is nearly uniform -- and the
            # uniform shift applied by volume_project then cancels it almost
            # exactly. The design freezes while J drifts upward.
            g = g_raw - g_raw.mean()

            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g**2
            mh = m / (1 - b1**it)
            vh = v / (1 - b2**it)
            step = lr * mh / (np.sqrt(vh) + eps)
            step -= step.mean()  # keep the step itself volume-neutral
            rho = np.clip(rho - step, 0.0, 1.0)
            rho = volume_project(cp, rho, p.volume_fraction)

            # Optional diagnostic: how wrong is the naive gradient at *this*
            # design? The coupling weakens as the design solidifies (solid
            # blocks the flow), so this is what explains why an optimiser can
            # still succeed with a gradient that is badly wrong early on.
            diag = {}
            if diagnose and (it % diagnose == 0 or it == 1):
                gn = cp.one_way_grad(rho_prev)
                # Compare the RAW gradients. Using the projected `g` here
                # against a raw naive gradient would be comparing two different
                # things and would misstate every number below.
                a, b = g_raw.ravel(), gn.ravel()
                diag = {
                    "naive_rel_err": float(np.linalg.norm(b - a) / np.linalg.norm(a)),
                    "naive_cos": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
                    "naive_sign_flip": float(np.mean(np.sign(a) != np.sign(b))),
                    "loop_gain": float(
                        spectral_radius(cp, res["T"], *cp_alpha_k(cp, rho_prev))
                    ),
                }
                # The same comparison inside the volume-preserving subspace the
                # optimiser actually steps in, which is what governs whether it
                # still descends.
                ap, bp = a - a.mean(), b - b.mean()
                diag["naive_cos_projected"] = float(
                    ap @ bp / (np.linalg.norm(ap) * np.linalg.norm(bp))
                )
                diag["naive_sign_flip_projected"] = float(np.mean(np.sign(ap) != np.sign(bp)))
                grad_snaps.append({
                    "iter": it,
                    "g_exact": g_raw.copy(),
                    "g_naive": gn.copy(),
                    "rho_phys": res["rho_phys"].copy(),
                })

            vol = float(np.mean(res["rho_phys"]))
            # gamma-gated mode: record what the gate decided and what the
            # decision cost, so the saving can be audited rather than asserted.
            # Always record the adjoint cost, not just in gated mode: the whole
            # point of the gate is a cost comparison, and that needs the
            # always-exact run to have been measured the same way.
            gate = {"adjoint_matvecs": cp.stats["vjp_matvecs"] - mv_before}
            if mode == "gamma_gated":
                gate |= {
                    "gamma": res.get("gamma"),
                    "gate": res.get("gate", "exact"),
                }
            rec = {
                "iter": it,
                "J": res["J"],
                **diag,
                **gate,
                "volume": vol,
                "beta": cp.params.beta,
                "grad_norm": float(np.linalg.norm(g)),
                "fp_iters": res["info"]["iters"],
                "fp_residual": res["info"]["residual"],
                "seconds": time.time() - t0,
            }
            history.append(rec)
            print(
                f"[{it:4d}] J={res['J']:.6f}  vol={vol:.3f}  beta={cp.params.beta:.1f}  "
                f"|g|={rec['grad_norm']:.3e}  fp={rec['fp_iters']:3d}  "
                f"({rec['seconds']:.1f}s)",
                flush=True,
            )

            # Checkpoint every iteration so a crash costs one step, not the run.
            (out / f"history_{tag}_N{N}.json").write_text(json.dumps(history, indent=2))

            if it % snapshot_every == 0 or it == 1 or it == iters:
                snaps.append(
                    {
                        "iter": it,
                        "rho_phys": res["rho_phys"].copy(),
                        "T": res["T"].copy(),
                        "u": res["u"].copy(),
                        "v": res["v"].copy(),
                    }
                )

        np.savez_compressed(
            out / f"run_{tag}_N{N}.npz",
            rho_raw=rho,
            snapshots_rho=np.stack([s["rho_phys"] for s in snaps]),
            snapshots_T=np.stack([s["T"] for s in snaps]),
            snapshots_u=np.stack([s["u"] for s in snaps]),
            snapshots_v=np.stack([s["v"] for s in snaps]),
            snapshot_iters=np.array([s["iter"] for s in snaps]),
            **(
                {
                    "grad_iters": np.array([s["iter"] for s in grad_snaps]),
                    "grad_exact": np.stack([s["g_exact"] for s in grad_snaps]),
                    "grad_naive": np.stack([s["g_naive"] for s in grad_snaps]),
                    "grad_rho": np.stack([s["rho_phys"] for s in grad_snaps]),
                }
                if grad_snaps
                else {}
            ),
        )
        (out / f"history_{tag}_N{N}.json").write_text(json.dumps(history, indent=2))
        print(f"\nsaved -> {out}/run_{tag}_N{N}.npz")
        print(f"J: {history[0]['J']:.6f} -> {history[-1]['J']:.6f} "
              f"({100*(history[0]['J']-history[-1]['J'])/history[0]['J']:.1f}% reduction)")

        if mode == "gamma_gated":
            cheap = cp.stats["gate_cheap"]
            exact = cp.stats["gate_exact"]
            spent = cp.stats["vjp_matvecs"]          # adjoint GMRES matvecs
            gate_cost = cp.stats["gamma_vjps"]        # one VJP per decision
            # What always-exact would have cost: the same adjoint solve on
            # every iteration. Estimate the skipped ones at the mean observed
            # cost of the ones actually paid for, which is the only honest
            # figure available without running the counterfactual.
            per_solve = spent / exact if exact else float("nan")
            would_have = per_solve * (cheap + exact) if exact else float("nan")
            total = spent + gate_cost
            print(
                f"\ngamma gate (threshold {gamma_gate:g}):\n"
                f"  cheap gradient taken on {cheap}/{cheap+exact} iterations\n"
                f"  adjoint VJPs actually spent : {spent}\n"
                f"  gate VJPs (1 per iteration) : {gate_cost}\n"
                f"  always-exact would have cost: {would_have:.0f} "
                f"(at {per_solve:.1f} VJPs/solve)\n"
                f"  net VJPs: {total:.0f} vs {would_have:.0f} "
                f"-> {100*(1-total/would_have):.0f}% saved"
                if exact else
                f"\ngamma gate: took the cheap gradient on all {cheap} "
                f"iterations; no adjoint solve was ever needed"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--outdir", default="results")
    ap.add_argument(
        "--result-tag", default=None,
        help="filename tag for this run (for example 'diag'); defaults to --mode",
    )
    ap.add_argument(
        "--mode",
        default="composed",
        choices=["composed", "one_way", "frozen", "gamma_gated"],
        help="which gradient drives the optimiser; the naive ones are for "
             "comparison, gamma_gated decides per iteration",
    )
    ap.add_argument(
        "--gamma-gate", dest="gamma_gate", type=float, default=0.01,
        help="in gamma_gated mode, take the cheap gradient when gamma is below "
             "this (default 0.01, calibrated on this benchmark; validate it "
             "before transferring to another problem)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="print Newton progress")
    ap.add_argument("--Ra", type=float, default=1.0e3, help="Rayleigh number")
    ap.add_argument(
        "--diagnose", type=int, default=0,
        help="every K iterations, also measure the naive gradient's error and "
             "the coupling loop gain at the current design",
    )
    run(**vars(ap.parse_args()))
