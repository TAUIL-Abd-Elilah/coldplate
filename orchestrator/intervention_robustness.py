# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Repeat the equal-budget intervention across independently seeded designs.

The headline experiment in ``intervention_test.py`` sweeps three action sizes at
the strongest converged state. This companion check holds the action size and
Rayleigh number fixed and varies the random design instead. Each case still
uses fresh coupled forward solves to judge the two gradient-selected actions.

Usage:  python intervention_robustness.py [--seeds 1 3 4]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from intervention_test import main as run_intervention


def summarize(reports: list[dict]) -> dict:
    """Reduce single-seed reports to compact, auditable robustness evidence."""
    if not reports:
        raise ValueError("at least one intervention report is required")
    cases = []
    for report in reports:
        if len(report["rows"]) != 1:
            raise ValueError("robustness reports must contain one amplitude")
        row = report["rows"][0]
        exact = row["delta_J_exact_action"]
        naive = row["delta_J_naive_action"]
        cases.append(
            {
                "seed": report["seed"],
                "gamma": report["gamma"],
                "naive_relative_error": report["naive_relative_error"],
                "naive_cosine": report["naive_cosine"],
                "add_set_overlap": report["add_set_overlap"],
                "remove_set_overlap": report["remove_set_overlap"],
                "delta_J_exact_action": exact,
                "delta_J_naive_action": naive,
                "exact_advantage": row["exact_advantage"],
                "extra_cooling_fraction": abs(exact) / abs(naive) - 1,
            }
        )
    return {
        "N": reports[0]["N"],
        "Ra": reports[0]["Ra"],
        "amplitude": reports[0]["rows"][0]["amplitude"],
        "n_seeds": len(cases),
        "exact_wins": sum(c["exact_advantage"] > 0 for c in cases),
        "all_actions_reduce_J": all(
            c["delta_J_exact_action"] < 0 and c["delta_J_naive_action"] < 0
            for c in cases
        ),
        "cases": cases,
    }


def main(
    N: int = 20,
    Ra: float = 2.0e4,
    seeds: tuple[int, ...] = (1, 3, 4),
    amplitude: float = 0.025,
    out: str = "results/intervention_robustness.json",
) -> int:
    reports = []
    with tempfile.TemporaryDirectory(prefix="coldplate-interventions-") as tmp:
        for seed in seeds:
            path = Path(tmp) / f"seed-{seed}.json"
            status = run_intervention(
                N=N, Ra=Ra, seed=seed, amplitudes=(amplitude,), out=str(path)
            )
            if status != 0:
                raise RuntimeError(f"exact intervention did not win for seed {seed}")
            reports.append(json.loads(path.read_text()))

    result = summarize(reports)
    result["selection_note"] = (
        "Seeds 1, 3, and 4 are a fixed convergence-qualified pilot set; "
        "this is robustness evidence, not a population success-rate estimate."
    )
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))

    print("\nseed  gamma  naive error  exact delta J  shortcut delta J  extra cooling")
    for case in result["cases"]:
        print(
            f"{case['seed']:4d}  {case['gamma']:.3f}      "
            f"{case['naive_relative_error']:.3f}      "
            f"{case['delta_J_exact_action']:+.5f}         "
            f"{case['delta_J_naive_action']:+.5f}          "
            f"{100 * case['extra_cooling_fraction']:.0f}%"
        )
    print(f"exact-gradient choice wins {result['exact_wins']}/{result['n_seeds']} seeds")
    print(f"wrote {target}")
    return 0 if result["exact_wins"] == result["n_seeds"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--Ra", type=float, default=2.0e4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 3, 4])
    parser.add_argument("--amplitude", type=float, default=0.025)
    parser.add_argument("--out", default="results/intervention_robustness.json")
    args = vars(parser.parse_args())
    args["seeds"] = tuple(args["seeds"])
    raise SystemExit(main(**args))
