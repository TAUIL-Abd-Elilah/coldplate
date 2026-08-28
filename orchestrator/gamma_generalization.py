# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Does gamma predict outside our cold plate? A randomized study.

Every other result in this repository is measured on one physical system. That
is the honest limitation of the gamma claim: the derivation is general, but the
evidence was not. A reader is entitled to ask whether gamma tracks the
component-wise gradient error because that is what the mathematics says, or
because Boussinesq convection on a 96x96 grid happens to be well behaved.

So this script removes the physics entirely and tests the claim on randomly
generated coupled fixed points, where every quantity can be computed exactly:

    x* = Phi(x*, theta),   J = c . x*

For a linear loop Phi(x) = A x + B theta the exact adjoint and the shortcut are
both closed form,

    exact:     (I - A^T) lambda = c,   dJ/dtheta = B^T lambda
    shortcut:  lambda_0 = c,           dJ/dtheta = B^T c

so the error of cutting the loop needs no finite differences and no solver
tolerance. A nonlinear family, Phi(x) = tanh(A x) + b, is included as well,
where the fixed point is found by Newton and Phi_x is taken there.

Four structural families are drawn, because the spectral radius and the
directional residual come apart most sharply for non-normal operators:

    normal        symmetric A -- modal and norm behaviour are better aligned
    nonnormal     upper triangular with a heavy off-diagonal, so rho says
                  little about the transient gain in any given direction
    sparse        random 5% sparsity
    lowrank       rank-3 plus small noise, where most directions are inert

and the spectral radius is swept over 0.05 to 1.9 -- deliberately including
rho > 1, where the fixed point is repelling, the Neumann expansion of
(I - A^T)^-1 diverges, and the residual identity r_0 = Phi_x^T g still holds
exactly. That regime is where our own headline state lives (rho = 1.19), so the
study must expose rather than hide the predictor's limitation there.

gamma is computed by calling the shipped `coupling_check.coupling_gamma`, not a
reimplementation, so this exercises the module a user would actually import.

Usage:  python gamma_generalization.py [--trials 2000] [--n 40]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coupling_check import coupling_gamma  # noqa: E402

FAMILIES = ("normal", "nonnormal", "sparse", "lowrank")


def draw_operator(rng, n: int, family: str, rho_target: float) -> np.ndarray:
    """A random n x n operator of the requested structure, scaled to rho."""
    if family == "normal":
        M = rng.normal(size=(n, n))
        A = 0.5 * (M + M.T)
    elif family == "nonnormal":
        # Upper triangular: the eigenvalues sit on the diagonal, so the strong
        # off-diagonal coupling is invisible to rho by construction.
        A = np.triu(rng.normal(size=(n, n)), 1) * 2.0
        A += np.diag(rng.normal(size=n))
    elif family == "sparse":
        A = rng.normal(size=(n, n)) * (rng.random((n, n)) < 0.05)
    elif family == "lowrank":
        U = rng.normal(size=(n, 3))
        V = rng.normal(size=(3, n))
        A = U @ V + 0.01 * rng.normal(size=(n, n))
    else:
        raise ValueError(f"unknown family {family!r}")

    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    if rho < 1e-12:
        return np.zeros((n, n))
    return A * (rho_target / rho)


def linear_case(rng, n: int, m: int, family: str, rho_target: float):
    """One linear coupled system; returns measured error, gamma and rho."""
    A = draw_operator(rng, n, family, rho_target)
    B = rng.normal(size=(n, m))
    c = rng.normal(size=n)

    I = np.eye(n)
    if np.linalg.cond(I - A.T) > 1e12:
        return None  # adjoint system effectively singular; nothing to measure

    lam = np.linalg.solve(I - A.T, c)
    g_exact = B.T @ lam
    g_naive = B.T @ c
    denom = np.linalg.norm(g_exact)
    if denom < 1e-12:
        return None

    theta = rng.normal(size=m)
    x_star = np.linalg.solve(I - A, B @ theta)

    # gamma through the shipped check, on the real loop map
    Aj, Bj, tj = jnp.asarray(A), jnp.asarray(B), jnp.asarray(theta)
    report = coupling_gamma(
        lambda x: Aj @ x + Bj @ tj, jnp.asarray(x_star), jnp.asarray(c), n_terms=3
    )
    return {
        "family": family,
        "kind": "linear",
        "rho": float(np.max(np.abs(np.linalg.eigvals(A)))),
        "gamma": float(report.gamma),
        "terms": [float(t) for t in report.neumann_terms],
        "rel_err": float(np.linalg.norm(g_naive - g_exact) / denom),
        "cosine": float(
            g_naive @ g_exact
            / (np.linalg.norm(g_naive) * np.linalg.norm(g_exact) + 1e-300)
        ),
    }


def nonlinear_case(rng, n: int, m: int, family: str, rho_target: float):
    """Phi(x) = tanh(A x) + b, fixed point by Newton, derivatives by JAX."""
    A = draw_operator(rng, n, family, rho_target)
    b = 0.3 * rng.normal(size=n)
    B = rng.normal(size=(n, m))
    c = rng.normal(size=n)

    Aj, bj = jnp.asarray(A), jnp.asarray(b)

    def phi(x):
        return jnp.tanh(Aj @ x) + bj

    # Newton on F(x) = phi(x) - x
    x = jnp.zeros(n)
    for _ in range(60):
        F = phi(x) - x
        if float(jnp.max(jnp.abs(F))) < 1e-12:
            break
        Jm = jax.jacobian(phi)(x) - jnp.eye(n)
        if float(jnp.linalg.cond(Jm)) > 1e12:
            return None
        x = x - jnp.linalg.solve(Jm, F)
    else:
        return None
    if not bool(jnp.all(jnp.isfinite(x))):
        return None

    Phi_x = np.asarray(jax.jacobian(phi)(x))
    I = np.eye(n)
    if np.linalg.cond(I - Phi_x.T) > 1e12:
        return None
    lam = np.linalg.solve(I - Phi_x.T, c)
    g_exact = B.T @ lam
    g_naive = B.T @ c
    denom = np.linalg.norm(g_exact)
    if denom < 1e-12:
        return None

    report = coupling_gamma(phi, x, jnp.asarray(c), n_terms=3)
    return {
        "family": family,
        "kind": "nonlinear",
        "rho": float(np.max(np.abs(np.linalg.eigvals(Phi_x)))),
        "gamma": float(report.gamma),
        "terms": [float(t) for t in report.neumann_terms],
        "rel_err": float(np.linalg.norm(g_naive - g_exact) / denom),
        "cosine": float(
            g_naive @ g_exact
            / (np.linalg.norm(g_naive) * np.linalg.norm(g_exact) + 1e-300)
        ),
    }


def corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def report_block(label, rows):
    if not rows:
        print(f"  {label:<22} (no cases)")
        return None
    lg = corr(np.log10([r["gamma"] + 1e-300 for r in rows]),
              np.log10([r["rel_err"] + 1e-300 for r in rows]))
    lr = corr([r["rho"] for r in rows],
              np.log10([r["rel_err"] + 1e-300 for r in rows]))
    print(f"  {label:<22} n={len(rows):>5}   log-gamma {lg:+.4f}   rho {lr:+.4f}")
    return {"n": len(rows), "log_gamma_correlation": lg, "rho_correlation": lr}


def main(trials: int = 2000, n: int = 40, m: int = 25, seed: int = 0,
         out: str = "results/gamma_generalization.json") -> int:
    rng = np.random.default_rng(seed)
    rows = []
    print(f"randomized coupled fixed points: {trials} trials, n={n}, m={m}")
    print(f"families: {', '.join(FAMILIES)}; rho swept over 0.05..1.9 "
          f"(including the repelling regime)\n")

    for t in range(trials):
        family = FAMILIES[t % len(FAMILIES)]
        # Log-uniform, and reaching far below the physical sweep. The SAFE
        # verdict is the one that can actually hurt someone -- it tells a user
        # to skip the adjoint -- so the study has to generate enough weakly
        # coupled systems to test it, not just the dramatic ones.
        rho_target = float(10.0 ** rng.uniform(-3.0, np.log10(1.9)))
        maker = linear_case if (t % 5) else nonlinear_case
        try:
            row = maker(rng, n, m, family, rho_target)
        except Exception:  # noqa: BLE001 - a degenerate draw is not a result
            row = None
        if row is not None:
            rows.append(row)

    if not rows:
        print("no usable trials")
        return 1

    print("=== correlation with log10(relative error of the shortcut) ===")
    overall = report_block("all", rows)
    per_family = {f: report_block(f, [r for r in rows if r["family"] == f])
                  for f in FAMILIES}
    per_kind = {k: report_block(k, [r for r in rows if r["kind"] == k])
                for k in ("linear", "nonlinear")}

    print("\n=== split by spectral radius ===")
    sub = report_block("rho < 1 (attracting)", [r for r in rows if r["rho"] < 1.0])
    rep = report_block("rho >= 1 (repelling)", [r for r in rows if r["rho"] >= 1.0])

    # Why gamma should degrade for a repelling loop, and whether two more VJPs
    # recover it. gamma is the residual of the adjoint equation; the error it
    # causes is (I - Phi_x^T)^-1 applied to that residual. For normal operators
    # with rho < 1 its norm is bounded by 1/(1-rho); non-normal conditioning can
    # amplify more. When rho >= 1 no contraction-based bound applies and a
    # residual of a given size can mean almost anything. The successive terms
    # ||(Phi_x^T)^k g|| are a power iteration, so their ratio estimates rho for
    # free -- the question is whether correcting gamma by it helps.
    def refined(r):
        t = r.get("terms") or [r["gamma"]]
        if len(t) < 2 or t[0] <= 0:
            return r["gamma"]
        ratio = t[1] / t[0]
        # geometric sum of the neglected series when it converges; when it does
        # not, the correction is unbounded and we say so rather than pretend
        return t[0] / (1.0 - ratio) if ratio < 0.95 else float("inf")

    print("\n=== can two extra VJPs rescue the repelling regime? ===")
    for label, subset in (
        ("rho < 1", [r for r in rows if r["rho"] < 1.0]),
        ("rho >= 1", [r for r in rows if r["rho"] >= 1.0]),
    ):
        finite = [r for r in subset if np.isfinite(refined(r))]
        flagged = len(subset) - len(finite)
        if len(finite) >= 3:
            c_ref = corr(np.log10([refined(r) + 1e-300 for r in finite]),
                         np.log10([r["rel_err"] + 1e-300 for r in finite]))
            c_raw = corr(np.log10([r["gamma"] + 1e-300 for r in finite]),
                         np.log10([r["rel_err"] + 1e-300 for r in finite]))
            print(f"  {label:<9} decay-corrected {c_ref:+.4f} vs plain gamma "
                  f"{c_raw:+.4f}   (n={len(finite)}, {flagged} flagged "
                  f"non-decaying)")
        else:
            print(f"  {label:<9} {flagged}/{len(subset)} flagged as "
                  f"non-decaying -- the series does not converge, which is "
                  f"itself the answer: do not screen, compute the adjoint")

    # The operational question: does the SAFE threshold mean what it says?
    safe = [r for r in rows if r["gamma"] < 0.01]
    unsafe = [r for r in rows if r["gamma"] >= 0.10]
    safe_err = max((r["rel_err"] for r in safe), default=None)
    print("\n=== do the shipped thresholds hold up? ===")
    if safe:
        frac = np.mean([r["rel_err"] < 0.05 for r in safe])
        print(f"  gamma < 0.01 (SAFE)   n={len(safe):>5}   "
              f"worst error {safe_err:.4f}   under 5% error in {100*frac:.1f}%")
    if unsafe:
        frac = np.mean([r["rel_err"] > 0.05 for r in unsafe])
        print(f"  gamma >= 0.10 (UNSAFE) n={len(unsafe):>5}   "
              f"over 5% error in {100*frac:.1f}%")

    result = {
        "trials_requested": trials,
        "trials_usable": len(rows),
        "n": n,
        "m": m,
        "seed": seed,
        "overall": overall,
        "per_family": per_family,
        "per_kind": per_kind,
        "attracting": sub,
        "repelling": rep,
        "safe_bucket": {
            "n": len(safe),
            "worst_rel_err": safe_err,
            "frac_under_5pct": float(np.mean([r["rel_err"] < 0.05 for r in safe]))
            if safe else None,
        },
        "unsafe_bucket": {
            "n": len(unsafe),
            "frac_over_5pct": float(np.mean([r["rel_err"] > 0.05 for r in unsafe]))
            if unsafe else None,
        },
    }
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))

    # Per-trial values as well, so the claim can be plotted and re-checked
    # rather than taken from the summary (see make_figures.py, figure 10).
    np.savez_compressed(
        target.with_suffix(".npz"),
        gamma=np.array([r["gamma"] for r in rows]),
        rel_err=np.array([r["rel_err"] for r in rows]),
        rho=np.array([r["rho"] for r in rows]),
        cosine=np.array([r["cosine"] for r in rows]),
        family=np.array([r["family"] for r in rows]),
        kind=np.array([r["kind"] for r in rows]),
    )
    # The .npz is a large intermediate and stays out of the repository, so the
    # per-trial values the results page plots are written as compact JSON too:
    # gamma, the true relative error, the spectral radius, and the two labels,
    # rounded to six figures. Enough to redraw the scatter and recount the
    # screening thresholds in a fresh clone, without shipping a binary.
    points = target.with_name(target.stem + "_points.json")
    points.write_text(
        json.dumps(
            {
                "n": len(rows),
                "columns": ["gamma", "rel_err", "rho", "family", "kind"],
                "rows": [
                    [
                        float(f"{r['gamma']:.6g}"),
                        float(f"{r['rel_err']:.6g}"),
                        float(f"{r['rho']:.6g}"),
                        r["family"],
                        r["kind"],
                    ]
                    for r in rows
                ],
            },
            separators=(",", ":"),
        )
        + "\n",
        newline="\n",
    )

    print(f"\nwrote {target}, {points} and {target.with_suffix('.npz')}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--m", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gamma_generalization.json")
    raise SystemExit(main(**vars(ap.parse_args())))
