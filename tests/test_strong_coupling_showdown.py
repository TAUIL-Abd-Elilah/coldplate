# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from copy import deepcopy
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


def test_frozen_showdown_protocol_is_deterministic_and_discloses_prior_overlap():
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
    assert protocol["status"] == "retrospectively_frozen_design"
    assert protocol["outer_steps"] == p["iterations"] == 8
    assert "prior favourable single-step evidence" in (
        protocol["prior_observation_disclosure"]["interpretation"]
    )


def test_trajectory_metrics_use_true_objective_sequence():
    metrics = trajectory_metrics([10.0, 9.0, 9.5, 8.0])
    assert metrics["initial_J"] == 10.0
    assert metrics["final_J"] == 8.0
    assert metrics["reduction_percent"] == pytest.approx(20.0)
    assert metrics["trajectory_auc"] == pytest.approx(27.5)
    assert metrics["improving_steps"] == 2


def test_summary_requires_all_three_completed_and_strict_composed_win():
    def branch(method, final, complete=True, initial=2.0):
        objectives = np.linspace(initial, final, 9).tolist()
        rows = [
            {
                "iteration": index + 1,
                "status": "accepted",
                "raw_design_sha256": f"{index:064x}",
                "J_before": objectives[index],
                "J_after": objectives[index + 1],
                "delta_J": objectives[index + 1] - objectives[index],
            }
            for index in range(8)
        ]
        value = {
            "method": method,
            "complete": complete,
            "failure": None if complete else {"stage": "test"},
            "planned_iterations": 8,
            "completed_iterations": 8 if complete else 7,
            "objectives": objectives if complete else objectives[:-1],
            "rows": rows if complete else rows[:-1],
            "proposals": deepcopy(rows if complete else rows[:-1]),
            "metrics": trajectory_metrics(objectives if complete else objectives[:-1]),
        }
        return value

    result = summarize([
        branch("composed", 1.0),
        branch("one_way", 1.1),
        branch("frozen", 1.2),
    ])
    assert result["frozen_success_condition_met"] is True
    assert result["all_branches_complete"] is True
    assert result["common_initial_objective_verified"] is True
    assert all(
        comparison["relation"] == "composed_lower"
        for comparison in result["final_objective_comparisons"]
    )
    incomplete = summarize([
        branch("composed", 1.0),
        branch("one_way", 1.1, complete=False),
        branch("frozen", 1.2),
    ])
    assert incomplete["frozen_success_condition_met"] is False
    assert incomplete["all_branches_complete"] is False


def test_summary_rejects_fewer_than_eight_steps_and_mismatched_initial_state():
    def branch(method, initial=2.0):
        objectives = np.linspace(initial, 1.0, 9).tolist()
        rows = [
            {
                "iteration": index + 1,
                "status": "accepted",
                "raw_design_sha256": f"{index:064x}",
                "J_before": objectives[index],
                "J_after": objectives[index + 1],
                "delta_J": objectives[index + 1] - objectives[index],
            }
            for index in range(8)
        ]
        return {
            "method": method,
            "complete": True,
            "failure": None,
            "planned_iterations": 8,
            "completed_iterations": 8,
            "objectives": objectives,
            "rows": rows,
            "proposals": deepcopy(rows),
            "metrics": trajectory_metrics(objectives),
        }

    too_short = [branch(name) for name in ("composed", "one_way", "frozen")]
    too_short[0]["rows"].pop()
    too_short[0]["proposals"].pop()
    too_short[0]["completed_iterations"] = 7
    assert summarize(too_short)["all_branches_complete"] is False

    mismatched = [
        branch("composed"),
        branch("one_way", initial=2.01),
        branch("frozen"),
    ]
    result = summarize(mismatched)
    assert result["common_initial_objective_verified"] is False
    assert result["frozen_success_condition_met"] is False


def test_summary_treats_roundoff_scale_final_difference_as_equivalent():
    def branch(method, final):
        objectives = np.linspace(2.0, final, 9).tolist()
        rows = [
            {
                "iteration": index + 1,
                "status": "accepted",
                "raw_design_sha256": f"{index:064x}",
                "J_before": objectives[index],
                "J_after": objectives[index + 1],
                "delta_J": objectives[index + 1] - objectives[index],
            }
            for index in range(8)
        ]
        return {
            "method": method, "complete": True, "failure": None,
            "planned_iterations": 8, "completed_iterations": 8,
            "objectives": objectives, "rows": rows, "proposals": deepcopy(rows),
            "metrics": trajectory_metrics(objectives),
        }

    result = summarize([
        branch("composed", 1.0),
        branch("one_way", 1.0 + 1.0e-9),
        branch("frozen", 1.2),
    ])
    assert result["final_objective_comparisons"][0]["relation"] == "numerically_equivalent"
    assert result["frozen_success_condition_met"] is False
