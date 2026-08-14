# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "orchestrator"))

from strong_coupling_showdown import (  # noqa: E402
    initial_design,
    load_protocol,
    summarize,
    trajectory_metrics,
)


def test_locked_showdown_protocol_is_deterministic_and_fair():
    protocol = load_protocol(
        ROOT / "orchestrator" / "protocols" / "strong_coupling_showdown_v1.json"
    )
    p = protocol["parameters"]
    a = initial_design(p)
    b = initial_design(p)
    assert a.shape == (20, 20)
    assert np.array_equal(a, b)
    assert np.all((a >= 0.25) & (a < 0.75))
    assert protocol["fairness"]["same_true_candidate_forward_budget"] is True


def test_trajectory_metrics_use_true_objective_sequence():
    metrics = trajectory_metrics([10.0, 9.0, 9.5, 8.0])
    assert metrics["initial_J"] == 10.0
    assert metrics["final_J"] == 8.0
    assert metrics["reduction_percent"] == pytest.approx(20.0)
    assert metrics["trajectory_auc"] == pytest.approx(27.5)
    assert metrics["improving_steps"] == 2


def test_summary_requires_all_three_completed_and_strict_composed_win():
    def branch(method, final, complete=True):
        return {
            "method": method,
            "complete": complete,
            "metrics": {"final_J": final, "reduction_percent": 1.0},
        }

    result = summarize([
        branch("composed", 1.0),
        branch("one_way", 1.1),
        branch("frozen", 1.2),
    ])
    assert result["preregistered_success_condition_met"] is True
    incomplete = summarize([
        branch("composed", 1.0),
        branch("one_way", 1.1, complete=False),
        branch("frozen", 1.2),
    ])
    assert incomplete["preregistered_success_condition_met"] is False
