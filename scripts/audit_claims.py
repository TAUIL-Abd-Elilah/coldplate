#!/usr/bin/env python3
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Check that the headline numbers quoted in README/PAPER are measured.

Written after several claims in this repository turned out to be stale or wrong
-- a gradient comparison that measured a mean-removal instead of the coupling
and a convergence order extrapolated through a stalled solve. Prose drifts from
data silently, so this re-derives the canonical values from the stored result
files and asserts that the documents agree.

    usage:  python scripts/audit_claims.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "orchestrator" / "results"

FAIL: list[str] = []
NOTE: list[str] = []


def load(name, sub="results"):
    p = ROOT / "orchestrator" / sub / name
    if not p.exists():
        NOTE.append(f"missing result file: {sub}/{name}")
        return None
    return json.loads(p.read_text())


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(label)


def docs_contain(needle: str, *, files=("README.md", "PAPER.md")) -> list[str]:
    """Which documents mention this exact string."""
    out = []
    for f in files:
        if needle in (ROOT / f).read_text(encoding="utf-8"):
            out.append(f)
    return out


def main() -> int:
    print("=== canonical values, re-derived from stored results ===\n")

    # ---- critical Rayleigh number -------------------------------------
    rows = load("critical_rayleigh.json")
    if rows:
        best = min(rows, key=lambda r: abs(r["excess"]))
        ra = best["ra_classical"]
        print(f"critical Rayleigh: finest aspect={best['aspect']} "
              f"Ra_c={ra:.2f} excess={100*best['excess']:+.3f}%")
        check("Ra_c within 0.1% of 1707.762 at the widest box",
              abs(ra / 1707.762 - 1) < 1e-3, f"measured {ra:.2f}")
        check("README/PAPER quote 1707.97", bool(docs_contain("1707.97")))
        # monotone approach from above
        srt = sorted(rows, key=lambda r: r["aspect"])
        check("Ra_c decreases monotonically with aspect ratio",
              all(a["ra_classical"] > b["ra_classical"]
                  for a, b in zip(srt, srt[1:])))
        check("every measured Ra_c exceeds the unbounded value",
              all(r["ra_classical"] > 1707.762 for r in rows))

    # ---- grid convergence ---------------------------------------------
    rows = load("grid_convergence.json")
    if rows:
        Js = {r["N"]: r["J"] for r in rows}
        print(f"grid convergence: J(16)={Js.get(16):.6f} .. J(96)={Js.get(96):.6f}")
        check("all grids converged", all(r["ok"] for r in rows))
        seq = [Js[n] for n in sorted(Js)]
        check("J is monotone in N (asymptotic range)",
              all(a < b for a, b in zip(seq, seq[1:]))
              or all(a > b for a, b in zip(seq, seq[1:])))
        import math
        for trio in ((16, 32, 64), (24, 48, 96)):
            if all(n in Js for n in trio):
                d1, d2 = Js[trio[0]] - Js[trio[1]], Js[trio[1]] - Js[trio[2]]
                p = math.log(d1 / d2) / math.log(2.0)
                print(f"  observed order on {trio}: {p:.3f}")
                check(f"order on {trio} is between 1.5 and 2.5 (2nd order)",
                      1.5 < p < 2.5, f"p={p:.3f}")

    # ---- predictor comparison -----------------------------------------
    rows = load("predict_error.json")
    if rows:
        import numpy as np
        g = np.array([r["gamma"] for r in rows])
        e = np.array([r["rel_err"] for r in rows])
        s = np.array([r["rho_phi"] for r in rows])
        lg = np.corrcoef(np.log10(np.maximum(g, 1e-12)), np.log10(np.maximum(e, 1e-12)))[0, 1]
        ls = np.corrcoef(s, np.log10(np.maximum(e, 1e-12)))[0, 1]
        print(f"predictors: corr(log gamma)={lg:.4f}  corr(rho)={ls:.4f}  n={len(rows)}")
        check("README/PAPER quote 0.995 for log-gamma correlation",
              bool(docs_contain("0.995")) and abs(lg - 0.995) < 0.005,
              f"measured {lg:.4f}")
        check("README/PAPER quote 0.825 for rho correlation",
              bool(docs_contain("0.825")) and abs(ls - 0.825) < 0.005,
              f"measured {ls:.4f}")
        # the ordering-reversal pair
        pair = {r["design"]: r for r in rows
                if r["design"] in ("rough", "smooth") and abs(r["Ra"] - 1e4) < 1}
        if len(pair) == 2:
            ro, sm = pair["rough"], pair["smooth"]
            check("rho orders the Ra=1e4 pair backwards",
                  sm["rho_phi"] < ro["rho_phi"] and sm["rel_err"] > ro["rel_err"],
                  f"rough rho={ro['rho_phi']:.3f} err={ro['rel_err']:.3f} | "
                  f"smooth rho={sm['rho_phi']:.3f} err={sm['rel_err']:.3f}")
            check("gamma orders that pair correctly",
                  sm["gamma"] > ro["gamma"])
            for v in (f"{ro['rho_phi']:.3f}", f"{sm['rho_phi']:.3f}"):
                check(f"docs quote rho={v}", bool(docs_contain(v)))

    stats = load("predictor_statistics.json")
    if stats:
        check("predictor report states 14 converged of 20 attempted",
              stats["n_converged"] == 14
              and stats["attempted_configurations"] == 20)
        holdouts = stats["leave_one_family_out"]
        check("log-gamma correlation survives every family holdout",
              len(holdouts) == 4
              and min(r["log_gamma_correlation"] for r in holdouts.values()) > 0.98)
        lo, hi = stats["bootstrap_95_percent_interval"]
        check("bootstrap interval supports the predictor correlation",
              lo > 0.98 and hi <= 1.0, f"95% interval [{lo:.4f}, {hi:.4f}]")

    # ---- objective sweep ----------------------------------------------
    rows = load("objective_sweep.json")
    if rows:
        import numpy as np
        e = np.array([r["rel_err"] for r in rows])
        rp = {round(r["rho_phi"], 6) for r in rows}
        spread = e.max() / e.min()
        print(f"objective sweep: rho values seen={rp}  error spread={spread:.1f}x")
        check("rho is identical across every objective", len(rp) == 1)
        check("error spread is the ~136x quoted",
              abs(spread - 136) < 8, f"measured {spread:.1f}x")
        check("docs quote 0.5481 for the constant rho",
              bool(docs_contain("0.5481")))

    # ---- optimisation --------------------------------------------------
    for tag, quoted in (("composed", "1.2588"), ("one_way", "1.2576")):
        h = load(f"history_{tag}_N96.json")
        if h:
            J0, J1 = h[0]["J"], h[-1]["J"]
            red = 100 * (J0 - J1) / J0
            print(f"optimisation {tag}: J {J0:.4f} -> {J1:.4f}  ({red:.1f}% reduction)")
            check(f"{tag} final J matches the quoted {quoted}",
                  f"{J1:.4f}" == quoted, f"actual {J1:.4f}")
            check(f"{tag} reduction is the quoted 84.6%", abs(red - 84.6) < 0.15,
                  f"{red:.2f}%")

    # ---- sensitivity attribution ---------------------------------------
    rep = load("sensitivity_ranking.json")
    if rep:
        ow = rep["one_way"]
        by_k = {r["k"]: r for r in ow["per_k"]}
        sp = ow["spearman_magnitude"]
        n = ow["n_cells"]
        print(f"attribution: Spearman(|g|)={sp:+.4f} over {n} cells, "
              f"recall@50={by_k[50]['recall']:.0%}")
        check("naive influence ranking is no better than chance",
              abs(sp) < 0.05, f"Spearman {sp:+.4f}")
        check("README quotes the measured Spearman", bool(docs_contain("0.011")))
        check("naive misses the single most influential cell",
              not ow["top1_correct"],
              f"its pick is truly #{ow['top1_true_rank_of_naive_pick']+1}")
        check("recall of the true top 50 is the quoted 56%",
              abs(by_k[50]["recall"] - 0.56) < 0.01,
              f"{by_k[50]['recall']:.0%}")
        check("signs on the true top 50 are all correct",
              by_k[50]["sign_agreement_on_true_topk"] == 1.0,
              "this is why descent still works")
        worst = by_k[50]["worst_true_rank_promoted"] + 1
        check(f"docs quote the worst promoted cell as #{worst} of {n}",
              bool(docs_contain(str(worst))) and bool(docs_contain(str(n))),
              f"measured #{worst} of {n}")

    # ---- gamma-gated adjoint -------------------------------------------
    # Kept in its own directory: this is an 80-iteration cost comparison, and
    # writing it into results/ would silently overwrite the 120-iteration
    # N=48 runs that back the replication claim in the README.
    gated = load("history_gamma_gated_N48.json", sub="results_gate")
    base = load("history_composed_N48.json", sub="results_gate")
    if gated and base:
        cheap = sum(1 for r in gated if r.get("gate") == "cheap")
        spent = sum(r.get("adjoint_matvecs", 0) for r in gated)
        exact_cost = sum(r.get("adjoint_matvecs", 0) for r in base)
        Jg, Jb = gated[-1]["J"], base[-1]["J"]
        gap = abs(Jg - Jb) / abs(Jb)
        print(f"gamma gate: cheap on {cheap}/{len(gated)} iterations, "
              f"{spent}+{len(gated)} VJPs vs {exact_cost} always-exact, "
              f"final J {Jg:.4f} vs {Jb:.4f} ({100*gap:.2f}%)")
        check("the gate reproduces the exact-gradient design",
              gap < 0.02, f"final J differs by {100*gap:.2f}%")
        if exact_cost:
            saved = 1 - (spent + len(gated)) / exact_cost
            check("the gate costs less than always paying for the adjoint",
                  saved > 0.5, f"{100*saved:.0f}% fewer VJPs "
                  f"({spent + len(gated)} vs {exact_cost})")
        check("every gamma recorded is below the gate that was set",
              all(r["gamma"] < 0.10 for r in gated if r.get("gate") == "cheap"))
        check("gamma stays in the MARGINAL band throughout",
              all(0.001 < r["gamma"] < 0.10 for r in gated),
              f"range {min(r['gamma'] for r in gated):.4f}"
              f"-{max(r['gamma'] for r in gated):.4f}")

    # ---- forward-validated intervention -------------------------------
    intervention = load("intervention_test.json")
    if intervention:
        rows = intervention["rows"]
        check("exact-gradient intervention wins every equal-budget re-solve",
              intervention["exact_wins"] == intervention["n_amplitudes"]
              and all(r["exact_advantage"] > 0 for r in rows))
        check("every exact-gradient intervention reduces the true objective",
              all(r["delta_J_exact_action"] < 0 for r in rows))
        last = max(rows, key=lambda r: r["amplitude"])
        ratio = last["delta_J_exact_action"] / last["delta_J_naive_action"]
        check("largest action realizes the quoted ~58% extra cooling",
              1.55 < ratio < 1.61 and bool(docs_contain("58%")),
              f"measured {100*(ratio-1):.1f}%")

    robustness = load("intervention_robustness.json")
    if robustness and "seeds_converged" not in robustness:
        # Superseded schema: the first version of that sweep hand-picked its
        # seeds and raised on a loss, so it could not report one. Refuse to
        # audit it rather than quietly validate numbers it could not have
        # falsified.
        check("intervention robustness uses the pre-registered sweep", False,
              "results predate the pre-registered rewrite; re-run "
              "orchestrator/intervention_robustness.py")
        robustness = None
    if robustness:
        conv = robustness["seeds_converged"]
        wins = robustness["exact_wins"]
        cases = [c for c in robustness["cases"] if c["outcome"] != "not_converged"]
        print(f"intervention robustness: {wins}/{conv} wins over converged "
              f"designs, {robustness['seeds_attempted']} attempted, "
              f"{robustness['seeds_not_converged']} without a steady state")
        check("the seed range was declared up front, not selected after the fact",
              "declared before running" in robustness.get("selection_note", ""))
        check("the sweep is a contiguous range with no gaps",
              [c["seed"] for c in robustness["cases"]]
              == list(range(robustness["cases"][0]["seed"],
                            robustness["cases"][0]["seed"]
                            + robustness["seeds_attempted"])))
        check("every attempted seed is accounted for",
              conv + robustness["seeds_not_converged"]
              == robustness["seeds_attempted"])
        check("losses were recordable, and are counted if they occurred",
              wins + robustness["shortcut_wins"] == conv)
        check("the composed choice won every converged design",
              wins == conv and conv > 0, f"{wins}/{conv}")
        check("README/PAPER quote the 10-of-10 result",
              bool(docs_contain("10/10")) or bool(docs_contain("10 out of 10")))
        check("every converged design realizes more cooling",
              all(c["extra_cooling_fraction"] > 0 for c in cases))
        med = robustness["median_extra_cooling_when_winning"]
        check("docs quote the median extra cooling as 36%",
              abs(100 * med - 36) < 1.5 and bool(docs_contain("36%")),
              f"measured {100*med:.0f}%")
        best = robustness["max_extra_cooling_when_winning"]
        check("largest robustness advantage is the quoted ~276%",
              2.7 < best < 2.8 and bool(docs_contain("276%")),
              f"measured {100*best:.1f}%")

    # ---- randomized generalization study -------------------------------
    gg = load("gamma_generalization.json")
    if gg:
        o = gg["overall"]
        print(f"generalization: n={gg['trials_usable']} random systems, "
              f"log-gamma {o['log_gamma_correlation']:+.4f} vs rho "
              f"{o['rho_correlation']:+.4f}")
        check("README/PAPER quote 2,377 usable random systems",
              gg["trials_usable"] == 2377
              and (bool(docs_contain("2,377")) or bool(docs_contain("2377"))),
              f"measured {gg['trials_usable']}")
        check("pooled log-gamma correlation is the quoted 0.989",
              abs(o["log_gamma_correlation"] - 0.989) < 0.002
              and bool(docs_contain("0.989")),
              f"measured {o['log_gamma_correlation']:.4f}")
        check("pooled rho correlation is the quoted 0.691",
              abs(o["rho_correlation"] - 0.691) < 0.002
              and bool(docs_contain("0.691")),
              f"measured {o['rho_correlation']:.4f}")
        check("gamma beats rho in every structural family",
              all(b["log_gamma_correlation"] > b["rho_correlation"]
                  for b in gg["per_family"].values()))
        check("gamma beats rho for linear and nonlinear loops alike",
              all(b["log_gamma_correlation"] > b["rho_correlation"]
                  for b in gg["per_kind"].values()))
        safe = gg["safe_bucket"]
        check("no draw called SAFE hid an error above 5%",
              safe["frac_under_5pct"] == 1.0,
              f"n={safe['n']}, worst {100*safe['worst_rel_err']:.1f}%")
        check("docs quote the worst SAFE error as 1.4%",
              abs(100 * safe["worst_rel_err"] - 1.4) < 0.1,
              f"measured {100*safe['worst_rel_err']:.2f}%")
        check("every draw called UNSAFE genuinely exceeded 5%",
              gg["unsafe_bucket"]["frac_over_5pct"] == 1.0,
              f"n={gg['unsafe_bucket']['n']}")
        # The documented limitation must stay documented.
        check("the repelling-regime limitation is real and disclosed",
              gg["repelling"]["log_gamma_correlation"] < 0.75
              and bool(docs_contain("repelling")),
              f"repelling corr {gg['repelling']['log_gamma_correlation']:+.3f}")

    # ---- claims that must NOT appear ----------------------------------
    print("\n=== retracted claims must not reappear ===")
    for bad, why in [
        ("79% of", "artefact of comparing projected against raw gradients"),
    ]:
        hits = [f for f in ("README.md", "PAPER.md")
                if re.search(bad, (ROOT / f).read_text(encoding="utf-8"), re.I)]
        # the README keeps one deliberate mention inside its correction note
        allowed = bad == "79% of" and hits == ["README.md"]
        check(f"no live claim of '{bad}'", not hits or allowed,
              f"({why})" + (f" found in {hits}" if hits and not allowed else ""))

    print("\n=== stale artefacts and unsafe helpers must not return ===")
    source_files = [
        ROOT / "README.md", ROOT / "PAPER.md", ROOT / "DEMO_SCRIPT.md",
        ROOT / "orchestrator" / "make_figures.py",
    ]
    joined = "\n".join(p.read_text(encoding="utf-8") for p in source_files)
    for bad in ("40-150%", "74% wrong sign", "four languages", "four containers"):
        check(f"no stale '{bad}' wording", bad.lower() not in joined.lower())
    check("obsolete N48 trajectory diagnostic removed",
          not (ROOT / "orchestrator" / "results" / "history_diag_N48.json").exists())
    composed = load("history_composed_N96.json")
    stale_fields = {
        "naive_rel_err", "naive_cos", "naive_sign_flip", "loop_gain",
        "naive_cos_projected", "naive_sign_flip_projected",
    }
    if composed:
        check("optimization history contains no retracted diagnostic fields",
              all(not stale_fields.intersection(row) for row in composed))

    for name in ("validate_both_backends.sh", "run_optimisations.sh"):
        body = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        check(f"{name} never removes every running Docker container",
              "docker ps -q |" not in body and "ancestor=$image" in body)

    print()
    if NOTE:
        print("notes:")
        for n in NOTE:
            print(f"  - {n}")
    if FAIL:
        print(f"\n{len(FAIL)} CHECK(S) FAILED:")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print("every audited headline number matches the stored measurements")
    return 0


if __name__ == "__main__":
    sys.exit(main())
