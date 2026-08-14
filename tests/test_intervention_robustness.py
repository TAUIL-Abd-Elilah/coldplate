# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the fixed-range intervention sweep.

The point of this sweep is that it can report a negative. An earlier version
hand-picked its seeds and raised whenever the exact gradient lost, so a
disagreeing design crashed the run instead of appearing in the table. These
tests pin the properties that prevent that from coming back: losses are
counted, failed base solves and inconclusive action pairs are retained rather
than dropped, and every attempted seed is accounted for somewhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "orchestrator"))

from intervention_robustness import summarize  # noqa: E402


def case(seed, exact, naive):
    return {
        "seed": seed,
        "outcome": "exact_wins" if (naive - exact) > 0 else "shortcut_wins",
        "gamma": 0.3,
        "naive_relative_error": 0.7,
        "naive_cosine": 0.8,
        "add_set_overlap": 0.4,
        "remove_set_overlap": 0.2,
        "delta_J_exact_action": exact,
        "delta_J_naive_action": naive,
        "exact_advantage": naive - exact,
        "extra_cooling_fraction": abs(exact) / abs(naive) - 1,
    }


def base_failure(seed):
    return {
        "seed": seed,
        "outcome": "base_not_converged",
        "reason": "RuntimeError: base solve did not converge",
    }


def inconclusive(seed):
    return {
        "seed": seed,
        "outcome": "inconclusive",
        "reason": "exact action solver did not converge",
    }


def test_counts_forward_validated_wins():
    r = summarize([case(0, -0.04, -0.035), case(1, -0.06, -0.02),
                   case(2, -0.05, -0.03)], 20, 2.0e4, 0.025)
    assert r["exact_wins"] == 3
    assert r["seeds_comparable"] == 3
    assert r["shortcut_wins"] == 0
    assert r["win_rate_over_comparable"] == pytest.approx(1.0)
    assert r["all_comparable_actions_reduce_J"]


def test_a_loss_is_recorded_not_hidden():
    r = summarize([case(0, -0.04, -0.035), case(1, -0.02, -0.06)],
                  20, 2.0e4, 0.025)
    assert r["exact_wins"] == 1
    assert r["shortcut_wins"] == 1
    assert r["win_rate_over_comparable"] == pytest.approx(0.5)


def test_failed_and_inconclusive_designs_are_retained_and_excluded_from_rate():
    r = summarize([case(0, -0.04, -0.02), base_failure(1), inconclusive(2)],
                  20, 2.0e4, 0.025)
    assert r["seeds_attempted"] == 3
    assert r["seeds_comparable"] == 1
    assert r["base_state_failures"] == 1
    assert r["seeds_inconclusive"] == 1
    assert r["win_rate_over_comparable"] == pytest.approx(1.0)


def test_every_attempted_seed_is_accounted_for():
    cases = [case(0, -0.04, -0.02), case(1, -0.02, -0.06), inconclusive(2)]
    r = summarize(cases, 20, 2.0e4, 0.025)
    assert (
        r["seeds_comparable"]
        + r["seeds_inconclusive"]
        + r["base_state_failures"]
        == r["seeds_attempted"]
    )
    assert (
        r["exact_wins"] + r["shortcut_wins"] + r["ties"]
        == r["seeds_comparable"]
    )


def test_extra_cooling_statistics_describe_only_the_wins():
    r = summarize([case(0, -0.04, -0.02), case(1, -0.03, -0.02),
                   case(2, -0.01, -0.05)], 20, 2.0e4, 0.025)
    assert r["exact_wins"] == 2
    assert r["min_extra_cooling_when_winning"] == pytest.approx(0.5)
    assert r["max_extra_cooling_when_winning"] == pytest.approx(1.0)


def test_summary_states_that_seed_range_is_fixed_and_complete():
    r = summarize([case(0, -0.04, -0.02)], 20, 2.0e4, 0.025)
    assert "Fixed contiguous seed range" in r["selection_note"]
    assert "every seed is reported" in r["selection_note"]


def test_all_inconclusive_sweep_reports_no_rate_rather_than_zero():
    r = summarize([base_failure(0), inconclusive(1)], 20, 2.0e4, 0.025)
    assert r["seeds_comparable"] == 0
    assert r["win_rate_over_comparable"] is None
    assert r["median_extra_cooling_when_winning"] is None


def test_empty_sweep_is_not_a_silent_success():
    r = summarize([], 20, 2.0e4, 0.025)
    assert r["seeds_attempted"] == 0
    assert r["win_rate_over_comparable"] is None
