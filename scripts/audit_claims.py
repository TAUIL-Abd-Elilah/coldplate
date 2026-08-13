#!/usr/bin/env python3
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Check that every number quoted in README.md and PAPER.md is one we measured.

Written after several claims in this repository turned out to be stale or wrong
-- a gradient comparison that measured a mean-removal instead of the coupling,
an "ascent direction" that was never observed in the real pipeline, a
convergence order extrapolated through a stalled solve. Prose drifts from data
silently, so this re-derives the canonical values from the stored result files
and asserts that the documents agree.

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


def load(name):
    p = RES / name
    if not p.exists():
        NOTE.append(f"missing result file: {name}")
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

    # ---- claims that must NOT appear ----------------------------------
    print("\n=== retracted claims must not reappear ===")
    for bad, why in [
        ("ascent direction", "never observed in the composed pipeline"),
        ("points uphill", "the naive gradient is a descent direction here"),
        ("79% of", "artefact of comparing projected against raw gradients"),
    ]:
        hits = [f for f in ("README.md", "PAPER.md")
                if re.search(bad, (ROOT / f).read_text(encoding="utf-8"), re.I)]
        # the README keeps one deliberate mention inside its correction note
        allowed = bad == "79% of" and hits == ["README.md"]
        check(f"no live claim of '{bad}'", not hits or allowed,
              f"({why})" + (f" found in {hits}" if hits and not allowed else ""))

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
    print("every quoted number matches the stored measurements")
    return 0


if __name__ == "__main__":
    sys.exit(main())
