# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Repeat the equal-budget intervention across a fixed contiguous seed range.

``intervention_test.py`` sweeps three action sizes at one state. This companion
holds the action size and Rayleigh number fixed and varies the random design.

The design of this script matters as much as its result. An earlier version took
a hand-picked seed list and *raised* whenever the exact gradient failed to win,
which meant it could not report a loss even in principle: any seed that
disagreed would have crashed the run rather than appear in the table. Selecting
the seeds afterwards and then being unable to record a negative is not evidence,
it is a filter that manufactures the conclusion.

So this version:

* sweeps a contiguous seed range (0, 1, 2, ... by default), so there is no
  choosing which designs to believe after seeing individual outcomes;
* records every seed's outcome, including seeds where the exact gradient loses;
* evaluates both actions independently; one failed solve cannot suppress the
  competing action or be counted as a win;
* distinguishes a failed base solve from an inconclusive action comparison,
  without claiming that solver non-convergence proves non-existence;
* reports the win rate only over comparable designs, and prints the full table.

It exits non-zero only if the sweep could not be run at all. Whether the exact
gradient wins is the measurement, not the pass criterion.

Usage:  python intervention_robustness.py [--n-seeds 16] [--seed-start 0]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
from pathlib import Path

from intervention_test import main as run_intervention

DEFAULT_OUTCOME_ATOL = 1.0e-10
DEFAULT_OUTCOME_RTOL = 1.0e-8


def classify_outcome(delta_exact: float, delta_shortcut: float,
                     absolute_tolerance: float = DEFAULT_OUTCOME_ATOL,
                     relative_tolerance: float = DEFAULT_OUTCOME_RTOL,
                     ) -> tuple[str, float, float]:
    """Classify a comparable action pair without turning roundoff into a win.

    Returns ``(outcome, exact_advantage, equivalence_tolerance)``.  Positive
    advantage means the exact-gradient action produced the lower objective.
    The scale is the observed objective change, rather than one, so the
    relative term does not swamp small but well-resolved interventions.
    """
    values = (delta_exact, delta_shortcut, absolute_tolerance, relative_tolerance)
    if not all(not isinstance(value, bool) and isinstance(value, (int, float))
               and math.isfinite(float(value))
               for value in values):
        raise ValueError("comparison values and tolerances must be finite numbers")
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("comparison tolerances must be non-negative")
    exact = float(delta_exact)
    shortcut = float(delta_shortcut)
    tolerance = float(absolute_tolerance) + float(relative_tolerance) * max(
        abs(exact), abs(shortcut)
    )
    advantage = shortcut - exact
    if abs(advantage) <= tolerance:
        outcome = "tie"
    elif advantage > 0:
        outcome = "exact_wins"
    else:
        outcome = "shortcut_wins"
    return outcome, advantage, tolerance


def summarize(cases: list[dict], N: int, Ra: float, amplitude: float) -> dict:
    """Reduce per-seed outcomes to auditable, population-level evidence."""
    comparable = [
        c for c in cases
        if c["outcome"] in {"exact_wins", "shortcut_wins", "tie"}
    ]
    wins = [c for c in comparable if c["outcome"] == "exact_wins"]
    losses = [c for c in comparable if c["outcome"] == "shortcut_wins"]
    ties = [c for c in comparable if c["outcome"] == "tie"]
    inconclusive = [c for c in cases if c["outcome"] == "inconclusive"]
    base_failures = [c for c in cases if c["outcome"] == "base_not_converged"]
    runner_failures = [c for c in cases if c["outcome"] == "runner_failure"]
    extra = sorted(
        c["extra_cooling_fraction"] for c in wins
        if c.get("extra_cooling_fraction") is not None
    )
    return {
        "N": N,
        "Ra": Ra,
        "amplitude": amplitude,
        "seeds_attempted": len(cases),
        "seeds_comparable": len(comparable),
        "seeds_inconclusive": len(inconclusive),
        "base_state_failures": len(base_failures),
        "runner_failures": len(runner_failures),
        "exact_wins": len(wins),
        "shortcut_wins": len(losses),
        "ties": len(ties),
        "win_rate_over_comparable": (
            len(wins) / len(comparable) if comparable else None
        ),
        "median_extra_cooling_when_winning": (
            statistics.median(extra) if extra else None
        ),
        "min_extra_cooling_when_winning": extra[0] if extra else None,
        "max_extra_cooling_when_winning": extra[-1] if extra else None,
        "all_comparable_actions_reduce_J": all(
            c["delta_J_exact_action"] < 0 and c["delta_J_naive_action"] < 0
            for c in comparable
        ),
        "selection_note": (
            "Fixed contiguous seed range: every seed is reported, including "
            "every within-range loss, failed base solve, inconclusive action "
            "comparison, and runner failure. Whether the range itself was set "
            "before any related observation is disclosed by the enclosing "
            "study protocol rather than inferred from this runner."
        ),
        "cases": cases,
    }


def main(
    N: int = 20,
    Ra: float = 2.0e4,
    seed_start: int = 0,
    n_seeds: int = 16,
    amplitude: float = 0.025,
    outcome_atol: float = DEFAULT_OUTCOME_ATOL,
    outcome_rtol: float = DEFAULT_OUTCOME_RTOL,
    out: str = "results/intervention_robustness.json",
) -> int:
    if N <= 0 or n_seeds <= 0 or amplitude <= 0:
        raise ValueError("N, n_seeds, and amplitude must be positive")
    if not all(not isinstance(value, bool) and isinstance(value, (int, float))
               and math.isfinite(float(value))
               for value in (Ra, outcome_atol, outcome_rtol)):
        raise ValueError("Ra and outcome tolerances must be finite numbers")
    if Ra <= 0 or outcome_atol < 0 or outcome_rtol < 0:
        raise ValueError("Ra must be positive and outcome tolerances non-negative")
    cases: list[dict] = []
    seeds = list(range(seed_start, seed_start + n_seeds))
    print(f"fixed contiguous seed range: {seeds[0]}..{seeds[-1]} "
          f"({len(seeds)} designs), N={N}, Ra={Ra:.0e}, amplitude={amplitude}\n")
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        """Write after every seed.

        A difficult design can burn the full Newton budget, so one seed can take
        many minutes and the sweep can take hours. Writing only at the end means
        an interrupted run yields nothing at all -- and worse, it creates a
        quiet incentive to shorten the range until the script finishes, which
        is the selection effect this rewrite exists to remove.
        """
        partial = summarize(cases, N, Ra, amplitude)
        partial["complete"] = len(cases) == len(seeds)
        partial["seeds_planned"] = len(seeds)
        partial["outcome_equivalence_tolerance"] = {
            "absolute_delta_J": outcome_atol,
            "relative_delta_J": outcome_rtol,
        }
        partial["action_budget_definition"] = (
            "equal zero-sum perturbation of the same number of raw-design "
            "cells; not an equal realised physical-density or runtime budget"
        )
        target.write_text(json.dumps(partial, indent=2))

    with tempfile.TemporaryDirectory(prefix="coldplate-interventions-") as tmp:
        for seed in seeds:
            path = Path(tmp) / f"seed-{seed}.json"
            try:
                # The return status says whether the exact action won; we record
                # it either way rather than treating a loss as an error.
                run_intervention(
                    N=N, Ra=Ra, seed=seed, amplitudes=(amplitude,), out=str(path)
                )
            except Exception as exc:  # noqa: BLE001 - preserve every attempt
                reason = f"{type(exc).__name__}: {str(exc)[:120]}"
                outcome = (
                    "base_not_converged"
                    if "base coupled state did not converge" in str(exc)
                    else "runner_failure"
                )
                cases.append({
                    "seed": seed,
                    "outcome": outcome,
                    "failure_stage": (
                        "base_forward" if outcome == "base_not_converged"
                        else "execution_unit_exception"
                    ),
                    "reason": reason,
                })
                print(f"  seed {seed}: {outcome} ({type(exc).__name__})",
                      flush=True)
                checkpoint()
                continue

            try:
                report = json.loads(path.read_text())
                rows = report["rows"]
                if not isinstance(rows, list) or len(rows) != 1:
                    raise ValueError("execution-unit report must contain one row")
                row = rows[0]
            except Exception as exc:  # noqa: BLE001 - malformed output is runner failure
                cases.append({
                    "seed": seed,
                    "outcome": "runner_failure",
                    "failure_stage": "execution_unit_report",
                    "reason": f"{type(exc).__name__}: {str(exc)[:120]}",
                })
                print(f"  seed {seed}: runner_failure (invalid report)", flush=True)
                checkpoint()
                continue
            if row["outcome"] == "inconclusive":
                cases.append({
                    "seed": seed,
                    "outcome": "inconclusive",
                    "gamma": report["gamma"],
                    "naive_relative_error": report["naive_relative_error"],
                    "naive_cosine": report["naive_cosine"],
                    "add_set_overlap": report["add_set_overlap"],
                    "remove_set_overlap": report["remove_set_overlap"],
                    "exact_action_ok": row["exact_action_ok"],
                    "naive_action_ok": row["naive_action_ok"],
                    "exact_action_reason": row["exact_action_reason"],
                    "naive_action_reason": row["naive_action_reason"],
                    "delta_J_exact_action": row["delta_J_exact_action"],
                    "delta_J_naive_action": row["delta_J_naive_action"],
                })
                print(f"  seed {seed}: inconclusive action comparison", flush=True)
                checkpoint()
                continue
            exact, naive = row["delta_J_exact_action"], row["delta_J_naive_action"]
            try:
                outcome, advantage, tolerance = classify_outcome(
                    exact, naive, outcome_atol, outcome_rtol
                )
            except Exception as exc:  # noqa: BLE001 - bad evidence is runner failure
                cases.append({
                    "seed": seed,
                    "outcome": "runner_failure",
                    "failure_stage": "outcome_classification",
                    "reason": f"{type(exc).__name__}: {str(exc)[:120]}",
                })
                print(f"  seed {seed}: runner_failure (invalid comparison)", flush=True)
                checkpoint()
                continue
            cases.append({
                "seed": seed,
                "outcome": outcome,
                "execution_unit_reported_outcome": row["outcome"],
                "gamma": report["gamma"],
                "naive_relative_error": report["naive_relative_error"],
                "naive_cosine": report["naive_cosine"],
                "add_set_overlap": report["add_set_overlap"],
                "remove_set_overlap": report["remove_set_overlap"],
                "delta_J_exact_action": exact,
                "delta_J_naive_action": naive,
                "exact_advantage": advantage,
                "outcome_equivalence_tolerance": tolerance,
                # negative when the shortcut's action cooled more
                "extra_cooling_fraction": abs(exact) / abs(naive) - 1
                if naive != 0 else None,
            })
            print(f"  seed {seed}: {outcome} "
                  f"(exact {exact:+.5f} vs shortcut {naive:+.5f})", flush=True)
            checkpoint()

    result = summarize(cases, N, Ra, amplitude)
    result["complete"] = len(cases) == len(seeds)
    result["seeds_planned"] = len(seeds)
    result["outcome_equivalence_tolerance"] = {
        "absolute_delta_J": outcome_atol,
        "relative_delta_J": outcome_rtol,
    }
    result["action_budget_definition"] = (
        "equal zero-sum perturbation of the same number of raw-design cells; "
        "not an equal realised physical-density or runtime budget"
    )
    target.write_text(json.dumps(result, indent=2))

    print(f"\n{'seed':>4} {'outcome':>14} {'gamma':>7} {'naive err':>10} "
          f"{'exact dJ':>11} {'shortcut dJ':>12} {'extra cooling':>14}")
    for c in cases:
        if c["outcome"] not in {"exact_wins", "shortcut_wins", "tie"}:
            print(f"{c['seed']:>4} {c['outcome']:>14} {'-':>7} {'-':>10} "
                  f"{'-':>11} {'-':>12} {'-':>14}")
            continue
        print(f"{c['seed']:>4} {c['outcome']:>14} {c['gamma']:>7.3f} "
              f"{c['naive_relative_error']:>10.3f} "
              f"{c['delta_J_exact_action']:>+11.5f} "
              f"{c['delta_J_naive_action']:>+12.5f} "
              f"{100 * c['extra_cooling_fraction']:>13.0f}%")

    comparable, wins = result["seeds_comparable"], result["exact_wins"]
    print(f"\nattempted {result['seeds_attempted']} designs; {comparable} "
          "yielded two comparable action solves")
    if comparable:
        print(f"exact-gradient action won {wins}/{comparable} comparable cases; "
              f"shortcut won {result['shortcut_wins']}, ties {result['ties']}")
        if result["median_extra_cooling_when_winning"] is not None:
            print(f"extra cooling when it wins: median "
                  f"{100 * result['median_extra_cooling_when_winning']:.0f}%, "
                  f"range {100 * result['min_extra_cooling_when_winning']:.0f}%"
                  f"-{100 * result['max_extra_cooling_when_winning']:.0f}%")
    print(f"wrote {target}")
    # The measurement is the deliverable. Only an unrunnable sweep is a failure.
    return 0 if comparable else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--Ra", type=float, default=2.0e4)
    parser.add_argument("--seed-start", dest="seed_start", type=int, default=0)
    parser.add_argument("--n-seeds", dest="n_seeds", type=int, default=16)
    parser.add_argument("--amplitude", type=float, default=0.025)
    parser.add_argument("--outcome-atol", dest="outcome_atol", type=float,
                        default=DEFAULT_OUTCOME_ATOL)
    parser.add_argument("--outcome-rtol", dest="outcome_rtol", type=float,
                        default=DEFAULT_OUTCOME_RTOL)
    parser.add_argument("--out", default="results/intervention_robustness.json")
    raise SystemExit(main(**vars(parser.parse_args())))
