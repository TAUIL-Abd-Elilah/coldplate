# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""The parity artefact must survive the judge's first command.

`scripts/judge_demo.sh` runs `compare_thermal_backends.py` at an 8x8 smoke grid
by default. When that script learned to record its measurements, it briefly
recorded them unconditionally -- so the reviewer's very first command would
have overwritten the audited 16x16 evidence with smoke-grid numbers, leaving
them a dirty clone and a claim audit that failed for reasons that were not
their fault.

These tests pin the rule that fixed it: a run reports at any grid, and records
only at the grid the README quotes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "orchestrator" / "results" / "thermal_backend_parity.json"
SCRIPT = ROOT / "orchestrator" / "compare_thermal_backends.py"


def test_artifact_records_a_converged_swap():
    parity = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert parity["converged"] is True
    assert parity["interchangeable"] is True
    # Both backends ran: same objective to picoscale, but not the same float.
    assert parity["J_jax"] != parity["J_fortran"]
    assert abs(parity["J_jax"] - parity["J_fortran"]) < 1e-9
    # Every level the README tabulates is present and genuinely small.
    for key in ("component_forward_T", "component_jvp", "component_vjp",
                "coupled_state_T", "end_to_end_gradient"):
        assert 0.0 < parity[key] < 1e-8, f"{key} = {parity[key]}"
    assert f"{parity['gradient_cosine']:.12f}" == "1.000000000000"


def test_the_artifact_is_the_audited_grid():
    parity = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    source = SCRIPT.read_text(encoding="utf-8")
    audited = int(re.search(r"^AUDITED_N = (\d+)$", source, re.M).group(1))
    assert parity["N"] == audited


def test_a_smoke_grid_run_cannot_overwrite_the_evidence():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "if N == AUDITED_N:" in source, (
        "the artefact write must be gated on the audited grid"
    )
    # And the judge path really does use a different grid, which is why it matters.
    demo = (ROOT / "scripts" / "judge_demo.sh").read_text(encoding="utf-8")
    default_grid = int(re.search(r"^N=(\d+)$", demo, re.M).group(1))
    parity = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert default_grid != parity["N"], (
        "this test only means something while the judge path runs a "
        "different grid from the audited one"
    )
