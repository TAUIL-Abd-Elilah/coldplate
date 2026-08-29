# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""The independent-checker artefact has to keep saying what it means.

`check_gradients.py` records what Tesseract's own gradient checker makes of all
four components. The record separates two findings that must not be averaged
together:

  * **agreement**, measured on live comparisons as the largest absolute gap
    divided by the largest entry of the endpoint's own row. Scale-free, and not
    contaminated by the checker's hard-coded `atol=1e-8`.
  * **phantom sensitivities**, where the finite difference is *exactly* zero
    because the forward map never reads the perturbed input, yet the derivative
    endpoint reports a real number. That is a defect, not a tolerance miss, and
    this project's JAX thermal block has some.

Three things about the record are easy to corrupt quietly. The tolerance could
drift back towards the checker's 0.1 default, which measures the sampler rather
than the derivative. The per-input step scaling could be dropped, and the
large-magnitude inputs would then fail for reasons unrelated to their
derivatives. And a headline could come to rest on a step where nothing was
actually compared. These tests pin all three.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "orchestrator" / "results" / "check_gradients.json"
SCRIPT = ROOT / "orchestrator" / "check_gradients.py"
README = ROOT / "README.md"

COMPONENTS = ("material_map", "stokes_brinkman", "thermal_advdiff", "thermal_fortran")

pytestmark = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason="check_gradients.json is produced by a slow sweep; "
           "absent in a source-only checkout",
)


@pytest.fixture(scope="module")
def report():
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _paths(report):
    for name, component in report["components"].items():
        for path, entry in component["input_paths"].items():
            yield name, path, entry


def test_every_component_was_checked(report):
    assert set(report["components"]) == set(COMPONENTS)


def test_the_artifact_is_the_audited_grid(report):
    audited = int(
        re.search(r"^AUDITED_N = (\d+)$", SCRIPT.read_text(encoding="utf-8"), re.M).group(1)
    )
    assert report["N"] == audited


def test_every_live_comparison_agrees(report):
    worst = report["worst_relative_disagreement"]
    assert math.isfinite(worst)
    assert worst < 1e-5, f"worst live relative disagreement is {worst}"


def test_the_headline_is_the_worst_component_not_the_best(report):
    assert report["worst_relative_disagreement"] == max(
        component["relative_disagreement"] for component in report["components"].values()
    )


def test_each_component_reports_the_worst_of_its_input_paths(report):
    for name, component in report["components"].items():
        assert component["relative_disagreement"] == max(
            entry["relative_disagreement"] for entry in component["input_paths"].values()
        ), name


def test_every_input_path_was_swept_across_a_ladder_of_steps(report):
    assert len(report["rel_eps_ladder"]) >= 4
    for name, path, entry in _paths(report):
        assert set(entry["by_rel_eps"]) == {
            repr(rel) for rel in report["rel_eps_ladder"]
        }, f"{name}/{path}"


def test_the_step_is_scaled_to_each_input_and_not_left_absolute(report):
    for name, path, entry in _paths(report):
        for rel in report["rel_eps_ladder"]:
            assert entry["by_rel_eps"][repr(rel)]["eps"] == pytest.approx(
                rel * entry["input_magnitude"], rel=1e-12
            ), f"{name}/{path} at rel {rel}"


def test_the_best_rung_had_something_live_to_compare(report):
    """A step where everything sat inside the checker's absolute floor carries
    no relative information, so it must not win the ranking."""
    for name, path, entry in _paths(report):
        rungs = entry["by_rel_eps"]
        live = {rel: rung for rel, rung in rungs.items() if rung["live_rows"] > 0}
        if not live:
            assert entry["all_within_atol"], f"{name}/{path}"
            continue
        best = min(live.values(), key=lambda r: r["relative_disagreement_live"])
        assert entry["relative_disagreement"] == best["relative_disagreement_live"], (
            f"{name}/{path}"
        )
        assert rungs[repr(entry["best_rel_eps"])] == best, f"{name}/{path}"


def test_phantom_sensitivities_are_counted_not_averaged_away(report):
    """A derivative w.r.t. an input the forward map never reads is a defect.

    It is reported as its own number rather than folded into the agreement
    figure, where a handful of them would either vanish into a maximum or
    swamp it.
    """
    assert report["phantom_rows_total"] == sum(
        component["phantom_rows"] for component in report["components"].values()
    )
    for name, component in report["components"].items():
        assert component["phantom_rows"] == sum(
            entry["phantom_rows"] for entry in component["input_paths"].values()
        ), name


def test_any_component_with_phantom_sensitivities_is_named_in_the_readme(report):
    """Having found this, we do not get to leave it out of the prose."""
    affected = [name for name, component in report["components"].items()
                if component["phantom_rows"]]
    if not affected:
        return
    prose = README.read_text(encoding="utf-8")
    for name in affected:
        assert name in prose, name
    assert "phantom" in prose.lower(), (
        "the README must describe the phantom sensitivities the checker found"
    )


def test_the_large_magnitude_input_is_actually_large(report):
    """The per-input step scaling only matters because one input is far from
    O(1). If that stopped being true, the workaround and the upstream report it
    points at would both be describing nothing."""
    alpha = report["components"]["stokes_brinkman"]["input_paths"]["alpha"]
    assert alpha["input_magnitude"] > 1e3


def test_something_was_actually_compared(report):
    for name, path, entry in _paths(report):
        for rel, rung in entry["by_rel_eps"].items():
            assert rung["comparisons"] > 0, f"{name}/{path} at rel {rel}"


def test_both_derivative_endpoints_were_exercised(report):
    assert set(report["endpoints"]) == {
        "jacobian_vector_product",
        "vector_jacobian_product",
    }


def test_the_record_explains_its_two_policies(report):
    assert "absolut" in report["step_policy"].lower()
    assert "rtol=0" in report["verdict_policy"]


def test_the_native_components_name_the_binary_they_were_checked_against(report):
    """A stale compiled object is the failure mode this record must foreclose.

    The C++ and Fortran components are checked through a `.so` that is built,
    not committed, so the artefact records which bytes it loaded. Both cached
    images were stale the first time this ran, which is exactly how a gradient
    check turns into a green tick for code nobody is running.
    """
    for name in ("stokes_brinkman", "thermal_fortran"):
        entry = report["native_libraries"][name]
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), name
        assert entry["bytes"] > 0, name


def test_a_smoke_grid_run_cannot_overwrite_the_evidence():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "if N == AUDITED_N:" in source, (
        "the artefact write must be gated on the audited grid"
    )
