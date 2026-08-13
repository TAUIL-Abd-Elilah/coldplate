# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Does the naive gradient survive being used as a *sensitivity*?

Everywhere else in this repository the gradient is used as a search direction,
and the headline finding there is reassuring in a way that is easy to
over-read: a component-wise gradient that is 86% wrong still descends, because
descent only needs a positive inner product with the truth.

That is not the only thing engineers do with a gradient. The other standard use
is attribution -- "which parts of my design actually drive the objective?" --
which is what you run before committing to a manufacturing tolerance, choosing
where to place a sensor, or deciding which region deserves a finer mesh. That
use has no inner-product slack to hide behind. It reads the gradient entry by
entry, so every entry has to be individually right.

So this script runs the attribution task directly. Rank every design cell by
|dJ/drho|, take the cells a designer would actually act on, and ask what the
naive ranking gets:

  * Spearman rank correlation over the whole field
  * recall@k -- of the k genuinely most influential cells, how many appear in
    the naive top k
  * sign agreement restricted to the true top k, since acting on an influential
    cell in the wrong direction is worse than ignoring it
  * phantom hotspots -- cells the naive ranking promotes into the top k that
    are not remotely influential

Both naive gradients are measured: the strong one (full chain, feedback loop
cut) and the weak one (flow frozen). The strong one is the interesting case,
because it is what a competent engineer writes when the components are
separately differentiable but nobody differentiated the loop.

Usage:  python sensitivity_ranking.py [N] [Ra]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from pipeline import ColdPlate, Params


def spearman(a, b):
    """Rank correlation without pulling in scipy.stats, ties averaged."""
    def ranks(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # average ranks within tied groups
        xs = x[order]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[order[i : j + 1]] = 0.5 * (i + j)
            i = j + 1
        return r

    ra, rb = ranks(np.asarray(a, float)), ranks(np.asarray(b, float))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / denom) if denom > 0 else float("nan")


def rank_report(g_exact, g_naive, ks=(5, 10, 25, 50, 100)):
    """Compare the two gradients as *rankings of influence*."""
    a = np.asarray(g_exact).ravel()
    b = np.asarray(g_naive).ravel()
    n = a.size
    mag_a, mag_b = np.abs(a), np.abs(b)

    # descending order of influence
    ord_a = np.argsort(-mag_a, kind="mergesort")
    ord_b = np.argsort(-mag_b, kind="mergesort")
    # true rank of every cell, 0 = most influential
    true_rank = np.empty(n, dtype=np.int64)
    true_rank[ord_a] = np.arange(n)

    out = {
        "n_cells": int(n),
        "spearman_magnitude": spearman(mag_a, mag_b),
        "spearman_signed": spearman(a, b),
        "top1_correct": bool(ord_a[0] == ord_b[0]),
        "top1_true_rank_of_naive_pick": int(true_rank[ord_b[0]]),
        "per_k": [],
    }

    for k in ks:
        if k > n:
            continue
        set_a, set_b = set(ord_a[:k].tolist()), set(ord_b[:k].tolist())
        hit = set_a & set_b
        top_a = ord_a[:k]
        sign_ok = float(np.mean(np.sign(b[top_a]) == np.sign(a[top_a])))
        # Cells the naive ranking promotes into the top k that are not merely
        # misordered but genuinely uninfluential. The cut has to be at least k
        # itself: with a fixed 10%-of-field cut, asking for k above that would
        # brand correctly-ranked cells as phantoms, and a gradient compared
        # against itself would report phantoms.
        cut = max(k, 0.10 * n)
        phantom = [i for i in ord_b[:k] if true_rank[i] > cut]
        # worst promotion: the least genuinely-influential cell naive puts in
        # its top k
        worst = int(max(true_rank[ord_b[:k]]))
        out["per_k"].append(
            {
                "k": int(k),
                "recall": len(hit) / k,
                "sign_agreement_on_true_topk": sign_ok,
                "n_phantom": len(phantom),
                "phantom_frac": len(phantom) / k,
                "worst_true_rank_promoted": worst,
            }
        )
    return out


def main(N: int = 32, Ra: float = 3.0e4) -> int:
    p = Params(Nx=N, Ny=N, Ra=Ra)
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.25, 0.75, size=(N, N))

    print(f"sensitivity attribution, grid {N}x{N}, Ra={Ra:.0e}")
    print("ranking design cells by |dJ/drho| and asking what the naive")
    print("component-wise gradient would have told an engineer.\n")

    results = {"N": N, "Ra": Ra}
    with ColdPlate(params=p) as cp:
        res = cp.value_and_grad(rho)
        g_exact = res["grad"]
        print(f"exact composed gradient: J = {res['J']:.6f}, "
              f"fixed point residual {res['info']['residual']:.1e}")

        g_oneway = cp.one_way_grad(rho)
        g_frozen = cp.frozen_flow_grad(rho)
        print("naive gradients computed\n")

        # What would the shipped diagnostic have advised at this state, before
        # any of the work below was done? It costs one VJP, and the point of
        # reporting it here is that it must refuse: this is the regime where
        # attribution breaks, so a gate that waved it through would be useless.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from coupling_check import coupling_gamma  # noqa: E402

        mat = cp.material(rho)
        T_star, _ = cp.solve_coupled(mat["alpha"], mat["k"])
        import jax

        g_state = jax.grad(cp.objective)(T_star)
        report = coupling_gamma(
            lambda T: cp.phi(T, mat["alpha"], mat["k"]), T_star, g_state
        )
        rel = float(
            np.linalg.norm(np.asarray(g_oneway) - np.asarray(g_exact))
            / np.linalg.norm(np.asarray(g_exact))
        )
        results["gamma"] = report.gamma
        results["gamma_verdict"] = report.verdict
        results["naive_rel_err"] = rel
        print(f"coupling_check at this state: {report}")
        print(f"  measured naive gradient error here: {100*rel:.0f}%\n")

        for name, gn in (("one_way", g_oneway), ("frozen", g_frozen)):
            rep = rank_report(g_exact, gn)
            results[name] = rep
            label = {"one_way": "one-way (strong naive)",
                     "frozen": "frozen-flow (weak naive)"}[name]
            print(f"--- {label} ---")
            print(f"  Spearman rank corr (|g|)      : {rep['spearman_magnitude']:+.4f}")
            print(f"  Spearman rank corr (signed)   : {rep['spearman_signed']:+.4f}")
            print(f"  identifies the single most influential cell: "
                  f"{'yes' if rep['top1_correct'] else 'NO'}"
                  f"  (its pick is truly ranked #{rep['top1_true_rank_of_naive_pick']+1})")
            print(f"  {'k':>5} {'recall@k':>10} {'sign ok':>9} {'phantoms':>9} "
                  f"{'worst rank':>11}")
            for row in rep["per_k"]:
                print(f"  {row['k']:>5} {row['recall']:>9.0%} "
                      f"{row['sign_agreement_on_true_topk']:>8.0%} "
                      f"{row['n_phantom']:>9} {row['worst_true_rank_promoted']+1:>11}")
            print()

    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/sensitivity_ranking.json").write_text(json.dumps(results, indent=2))
    # Keep the fields too: the ranking statistics are much easier to believe as
    # a picture than as a table (see make_figures.py, figure 7).
    np.savez_compressed(
        "results/sensitivity_ranking.npz",
        g_exact=np.asarray(g_exact),
        g_oneway=np.asarray(g_oneway),
        g_frozen=np.asarray(g_frozen),
        rho_phys=np.asarray(res["rho_phys"]),
        T=np.asarray(res["T"]),
    )
    print("saved -> results/sensitivity_ranking.json, results/sensitivity_ranking.npz")
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    ra = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0e4
    sys.exit(main(n, ra))
