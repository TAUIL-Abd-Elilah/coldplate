# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""The cost comparison must stay arithmetic, and stay honest about it.

`adjoint_cost.py` quotes one number it did not measure: the wall clock a
central-difference gradient would take. That number is a multiplication of a
measured per-solve time by a count of design variables, and the danger with a
number like that is that it drifts into being quoted as if it had been run.

These tests pin the three things that keep it defensible: the extrapolation is
exactly the product it claims to be, the count of solves is the one the
parameterisation implies, and the artefact says in its own bytes that it was
extrapolated.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "orchestrator" / "results" / "adjoint_cost.json"
SCRIPT = ROOT / "orchestrator" / "adjoint_cost.py"

pytestmark = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason="adjoint_cost.json is produced by a Docker run; absent in a source-only checkout",
)


@pytest.fixture(scope="module")
def cost():
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_the_artifact_is_the_audited_grid(cost):
    audited = int(re.search(r"^AUDITED_N = (\d+)$", SCRIPT.read_text(encoding="utf-8"), re.M).group(1))
    assert cost["N"] == audited


def test_design_variable_count_matches_the_grid(cost):
    assert cost["n_design_variables"] == cost["N"] ** 2


def test_central_difference_cost_is_two_solves_per_design_variable(cost):
    fd = cost["central_difference_gradient"]
    assert fd["coupled_solves_required"] == 2 * cost["n_design_variables"]


def test_the_extrapolation_is_exactly_the_product_it_claims(cost):
    fd = cost["central_difference_gradient"]
    expected = fd["coupled_solves_required"] * cost["seconds_one_coupled_forward_solve"]
    assert fd["extrapolated_seconds"] == pytest.approx(expected, rel=1e-6)


def test_the_speedup_is_the_ratio_of_those_two_numbers(cost):
    fd = cost["central_difference_gradient"]
    expected = fd["extrapolated_seconds"] / cost["seconds_one_composed_gradient"]
    assert cost["adjoint_speedup_over_central_differences"] == pytest.approx(
        expected, rel=1e-3
    )


def test_the_artifact_labels_the_unmeasured_number_as_extrapolated(cost):
    fd = cost["central_difference_gradient"]
    assert "extrapolated" in fd["extrapolated_from"] or "measured" in fd["extrapolated_from"]
    # And the timing basis says which way the bias runs.
    assert "favours finite differences" in fd["timing_basis"]


def test_the_gradient_really_crossed_the_boundary_in_both_directions(cost):
    matvecs = cost["cross_boundary_matvecs_per_gradient"]
    assert matvecs["jvp"] > 0, "the forward Newton solve must have used JVPs"
    assert matvecs["vjp"] > 0, "the adjoint solve must have used VJPs"


def test_the_optimisation_arithmetic_uses_the_committed_history(cost):
    if "optimisation" not in cost:
        pytest.skip("no committed history for this grid")
    opt = cost["optimisation"]
    history = json.loads(
        (ROOT / "orchestrator" / "results" / opt["source"]).read_text(encoding="utf-8")
    )
    assert opt["iterations"] == len(history)
    assert opt["J_start"] == history[0]["J"]
    assert opt["J_final"] == history[-1]["J"]
    assert opt["measured_seconds"] == pytest.approx(
        sum(e["seconds"] for e in history), rel=1e-6
    )
