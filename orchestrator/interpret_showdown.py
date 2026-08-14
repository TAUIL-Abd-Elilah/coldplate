#!/usr/bin/env python3
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Derive a fail-closed interpretation of the frozen showdown artifact.

The raw artifact is immutable evidence.  Its original summary ranks only the
branches that reached eight decisions, which is structurally valid but easy to
misread as a three-way comparison.  This sidecar makes the primary endpoint
explicitly non-evaluable and permits only a labelled post-hoc common prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from strong_coupling_showdown import trajectory_metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_INPUT = HERE / "results" / "strong_coupling_showdown.json"
DEFAULT_OUTPUT = HERE / "results" / "strong_coupling_showdown_interpretation.json"


def _source_label(source: Path) -> str:
    """Name a source without leaking a local absolute path into the sidecar."""
    resolved = source.resolve()
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return resolved.name


def _same(left: object, right: object) -> bool:
    return (
        not isinstance(left, bool)
        and not isinstance(right, bool)
        and isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and math.isfinite(float(left))
        and math.isfinite(float(right))
        and math.isclose(float(left), float(right), rel_tol=2e-12, abs_tol=2e-12)
    )


def _validate_branch(branch: dict[str, Any], planned: int) -> None:
    completed = branch.get("completed_iterations")
    rows = branch.get("rows")
    proposals = branch.get("proposals")
    objectives = branch.get("objectives")
    metrics = branch.get("metrics")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or not 0 <= completed <= planned
        or branch.get("planned_iterations") != planned
        or not isinstance(rows, list)
        or len(rows) != completed
        or not isinstance(proposals, list)
        or not isinstance(objectives, list)
        or len(objectives) != completed + 1
        or not isinstance(metrics, dict)
        or not all(_same(value, value) for value in objectives)
    ):
        raise ValueError(f"invalid branch structure for {branch.get('method')!r}")
    if [row.get("iteration") for row in rows] != list(range(1, completed + 1)):
        raise ValueError("accepted branch rows are not a consecutive prefix")
    if rows != proposals[:completed] or any(row.get("status") != "accepted" for row in rows):
        raise ValueError("accepted rows do not match the durable proposal prefix")
    for index, row in enumerate(rows):
        if not (
            _same(row.get("J_before"), objectives[index])
            and _same(row.get("J_after"), objectives[index + 1])
            and _same(
                row.get("delta_J"),
                float(objectives[index + 1]) - float(objectives[index]),
            )
            and isinstance(row.get("raw_design_sha256"), str)
            and len(row["raw_design_sha256"]) == 64
        ):
            raise ValueError("an accepted proposal does not reproduce its objective chain")
    recomputed = trajectory_metrics([float(value) for value in objectives])
    if any(not _same(metrics.get(key), value) for key, value in recomputed.items()):
        raise ValueError("branch metrics do not reproduce from the objective chain")

    failure = branch.get("failure")
    if branch.get("complete") is True:
        if completed != planned or failure is not None or len(proposals) != planned:
            raise ValueError("a complete branch is not structurally complete")
        return
    if completed >= planned or not isinstance(failure, dict) or len(proposals) != completed + 1:
        raise ValueError("an incomplete branch lacks one terminal failure proposal")
    terminal = proposals[-1]
    if not (
        terminal.get("iteration") == completed + 1
        and terminal.get("status") == "candidate_not_converged"
        and failure.get("stage") == "candidate_forward"
        and failure.get("iteration") == completed + 1
        and terminal.get("raw_design_sha256")
        == failure.get("proposal_raw_design_sha256")
        and _same(terminal.get("candidate_residual"), failure.get("residual"))
        and isinstance(terminal.get("failed_raw_design"), list)
    ):
        raise ValueError("terminal proposal and branch failure do not match")


def build_interpretation(source: Path = DEFAULT_INPUT) -> dict[str, Any]:
    raw_bytes = source.read_bytes()
    data = json.loads(raw_bytes)
    protocol_path = ROOT / data["protocol_file"]
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if data.get("protocol_sha256") != protocol_hash:
        raise ValueError("showdown protocol hash does not match the committed bytes")
    if data.get("complete") is not False:
        raise ValueError("this interpretation is only for the frozen incomplete execution")

    planned = int(data["protocol"]["outer_steps"])
    branches = data.get("branches")
    if not isinstance(branches, list) or len(branches) != 3:
        raise ValueError("showdown must contain exactly three branches")
    branch_by_method = {branch.get("method"): branch for branch in branches}
    if set(branch_by_method) != {"composed", "one_way", "frozen"}:
        raise ValueError("showdown methods are missing or duplicated")
    for branch in branches:
        _validate_branch(branch, planned)

    composed = branch_by_method["composed"]
    if not (
        composed["completed_iterations"] == 5
        and composed["failure"]["iteration"] == 6
        and all(branch_by_method[name]["complete"] is True
                for name in ("one_way", "frozen"))
        and data["summary"]["frozen_success_condition_met"] is False
        and data["summary"]["final_objective_comparisons"] == []
    ):
        raise ValueError("artifact is not the recorded five-step terminal-failure state")
    initial = [float(branch["objectives"][0]) for branch in branches]
    targets = [float(branch["target_projected_volume"]) for branch in branches]
    if max(initial) - min(initial) > 1e-12 or max(targets) - min(targets) > 1e-12:
        raise ValueError("branches do not share their measured start and volume target")

    horizon = min(int(branch["completed_iterations"]) for branch in branches)
    descriptive: dict[str, Any] = {}
    for method, branch in branch_by_method.items():
        objectives = [float(value) for value in branch["objectives"][:horizon + 1]]
        metrics = trajectory_metrics(objectives)
        descriptive[method] = {
            "J_at_horizon": objectives[-1],
            "reduction_percent": metrics["reduction_percent"],
            "trajectory_auc": metrics["trajectory_auc"],
            "improving_steps": metrics["improving_steps"],
        }

    failure = composed["failure"]
    return {
        "schema_version": 1,
        "source_file": _source_label(source),
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "protocol_sha256": protocol_hash,
        "generated_by": "orchestrator/interpret_showdown.py",
        "execution_status": "incomplete_frozen_execution",
        "common_initial_objective_verified": True,
        "common_initial_objective": initial[0],
        "common_projected_volume_target_verified": True,
        "primary_endpoint": {
            "name": "true coupled objective after eight frozen-protocol decisions",
            "evaluable": False,
            "reason": "composed candidate_forward nonconvergence at iteration 6",
            "ranking": [],
            "comparisons": [],
            "frozen_success_condition_met": False,
        },
        "terminal_failure": {
            "method": "composed",
            "stage": failure["stage"],
            "iteration": failure["iteration"],
            "residual": failure["residual"],
            "proposal_raw_design_sha256": failure["proposal_raw_design_sha256"],
        },
        "completed_eight_step_branches_noncomparative": ["one_way", "frozen"],
        "descriptive_common_prefix": {
            "steps": horizon,
            "pre_specified": False,
            "status": "post_hoc_descriptive_only",
            "selection_reason": "minimum accepted-step count after the recorded failure",
            "methods": descriptive,
        },
    }


def main(source: Path = DEFAULT_INPUT, out: Path = DEFAULT_OUTPUT) -> int:
    interpretation = build_interpretation(source)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_bytes((json.dumps(interpretation, indent=2) + "\n").encode("utf-8"))
    temporary.replace(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    raise SystemExit(main(**vars(parser.parse_args())))
