# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Repeat the equal-budget intervention across a pre-declared range of designs.

``intervention_test.py`` sweeps three action sizes at one state. This companion
holds the action size and Rayleigh number fixed and varies the random design.

The design of this script matters as much as its result. An earlier version took
a hand-picked seed list and *raised* whenever the exact gradient failed to win,
which meant it could not report a loss even in principle: any seed that
disagreed would have crashed the run rather than appear in the table. Selecting
the seeds afterwards and then being unable to record a negative is not evidence,
it is a filter that manufactures the conclusion.

So this version:

* sweeps a contiguous seed range declared up front (0, 1, 2, ... by default), so
  there is no choosing which designs to believe;
* records every seed's outcome, including seeds where the exact gradient loses;
* records seeds whose coupled state does not converge as ``not_converged``,
  with the reason, rather than silently dropping them -- a design with no
  reachable steady state is a fact about the physics, not a failed trial to be
  discarded;
* reports the win rate over converged designs, and prints the full table.

It exits non-zero only if the sweep could not be run at all. Whether the exact
gradient wins is the measurement, not the pass criterion.

Usage:  python intervention_robustness.py [--n-seeds 16] [--seed-start 0]
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path

from intervention_test import main as run_intervention


def summarize(cases: list[dict], N: int, Ra: float, amplitude: float) -> dict:
    """Reduce per-seed outcomes to auditable, population-level evidence."""
    converged = [c for c in cases if c["outcome"] != "not_converged"]
    wins = [c for c in converged if c["outcome"] == "exact_wins"]
    losses = [c for c in converged if c["outcome"] == "shortcut_wins"]
    extra = sorted(c["extra_cooling_fraction"] for c in wins)
    return {
        "N": N,
        "Ra": Ra,
        "amplitude": amplitude,
        "seeds_attempted": len(cases),
        "seeds_converged": len(converged),
        "seeds_not_converged": len(cases) - len(converged),
        "exact_wins": len(wins),
        "shortcut_wins": len(losses),
        "win_rate_over_converged": (len(wins) / len(converged)) if converged else None,
        "median_extra_cooling_when_winning": (
            statistics.median(extra) if extra else None
        ),
        "min_extra_cooling_when_winning": extra[0] if extra else None,
        "max_extra_cooling_when_winning": extra[-1] if extra else None,
        "all_converged_actions_reduce_J": all(
            c["delta_J_exact_action"] < 0 and c["delta_J_naive_action"] < 0
            for c in converged
        ),
        "selection_note": (
            "Contiguous seed range declared before running; every seed is "
            "reported, including losses and designs with no reachable steady "
            "state. No seed was chosen after seeing its result."
        ),
        "cases": cases,
    }


def main(
    N: int = 20,
    Ra: float = 2.0e4,
    seed_start: int = 0,
    n_seeds: int = 16,
    amplitude: float = 0.025,
    out: str = "results/intervention_robustness.json",
) -> int:
    cases: list[dict] = []
    seeds = list(range(seed_start, seed_start + n_seeds))
    print(f"pre-declared seed range: {seeds[0]}..{seeds[-1]} "
          f"({len(seeds)} designs), N={N}, Ra={Ra:.0e}, amplitude={amplitude}\n")
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        """Write after every seed.

        A design with no reachable steady state burns the full Newton budget
        before it can be declared non-convergent, so a single seed can take
        many minutes and the sweep can take hours. Writing only at the end
        means an interrupted run yields nothing at all -- and worse, it creates
        a quiet incentive to shorten the seed range until the script finishes,
        which is the selection effect this rewrite exists to remove.
        """
        partial = summarize(cases, N, Ra, amplitude)
        partial["complete"] = len(cases) == len(seeds)
        partial["seeds_planned"] = len(seeds)
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
            except Exception as exc:  # noqa: BLE001 - any non-convergence
                cases.append({
                    "seed": seed,
                    "outcome": "not_converged",
                    "reason": f"{type(exc).__name__}: {str(exc)[:120]}",
                })
                print(f"  seed {seed}: not converged ({type(exc).__name__})",
                      flush=True)
                checkpoint()
                continue

            report = json.loads(path.read_text())
            row = report["rows"][0]
            exact, naive = row["delta_J_exact_action"], row["delta_J_naive_action"]
            won = row["exact_advantage"] > 0
            cases.append({
                "seed": seed,
                "outcome": "exact_wins" if won else "shortcut_wins",
                "gamma": report["gamma"],
                "naive_relative_error": report["naive_relative_error"],
                "naive_cosine": report["naive_cosine"],
                "add_set_overlap": report["add_set_overlap"],
                "remove_set_overlap": report["remove_set_overlap"],
                "delta_J_exact_action": exact,
                "delta_J_naive_action": naive,
                "exact_advantage": row["exact_advantage"],
                # negative when the shortcut's action cooled more
                "extra_cooling_fraction": abs(exact) / abs(naive) - 1
                if naive != 0 else float("inf"),
            })
            print(f"  seed {seed}: {'exact wins ' if won else 'SHORTCUT WINS'} "
                  f"(exact {exact:+.5f} vs shortcut {naive:+.5f})", flush=True)
            checkpoint()

    result = summarize(cases, N, Ra, amplitude)
    result["complete"] = len(cases) == len(seeds)
    result["seeds_planned"] = len(seeds)
    target.write_text(json.dumps(result, indent=2))

    print(f"\n{'seed':>4} {'outcome':>14} {'gamma':>7} {'naive err':>10} "
          f"{'exact dJ':>11} {'shortcut dJ':>12} {'extra cooling':>14}")
    for c in cases:
        if c["outcome"] == "not_converged":
            print(f"{c['seed']:>4} {'not converged':>14} {'-':>7} {'-':>10} "
                  f"{'-':>11} {'-':>12} {'-':>14}")
            continue
        print(f"{c['seed']:>4} {c['outcome']:>14} {c['gamma']:>7.3f} "
              f"{c['naive_relative_error']:>10.3f} "
              f"{c['delta_J_exact_action']:>+11.5f} "
              f"{c['delta_J_naive_action']:>+12.5f} "
              f"{100 * c['extra_cooling_fraction']:>13.0f}%")

    conv, wins = result["seeds_converged"], result["exact_wins"]
    print(f"\nattempted {result['seeds_attempted']} designs; {conv} had a "
          f"reachable steady state")
    if conv:
        print(f"exact-gradient action won {wins}/{conv} "
              f"({100 * wins / conv:.0f}% of converged designs)")
        if result["median_extra_cooling_when_winning"] is not None:
            print(f"extra cooling when it wins: median "
                  f"{100 * result['median_extra_cooling_when_winning']:.0f}%, "
                  f"range {100 * result['min_extra_cooling_when_winning']:.0f}%"
                  f"-{100 * result['max_extra_cooling_when_winning']:.0f}%")
    print(f"wrote {target}")
    # The measurement is the deliverable. Only an unrunnable sweep is a failure.
    return 0 if conv else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--Ra", type=float, default=2.0e4)
    parser.add_argument("--seed-start", dest="seed_start", type=int, default=0)
    parser.add_argument("--n-seeds", dest="n_seeds", type=int, default=16)
    parser.add_argument("--amplitude", type=float, default=0.025)
    parser.add_argument("--out", default="results/intervention_robustness.json")
    raise SystemExit(main(**vars(parser.parse_args())))
