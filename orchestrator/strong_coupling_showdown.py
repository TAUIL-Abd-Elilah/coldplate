# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Run the retrospectively frozen strong-coupling optimisation showdown.

This is deliberately a fixed-budget decision experiment rather than another
gradient norm comparison.  Three branches begin from the exact same raw design
and use the same update rule.  Only the sensitivity changes:

``composed``
    Implicit adjoint of the converged two-way fixed point.
``one_way``
    Differentiates every component, but cuts the temperature/flow feedback.
``frozen``
    Holds the flow fixed while differentiating the thermal/material path.

Every proposed action is judged by a fresh solve of the true coupled physics.
The JSON protocol records the frozen seed, coupling, action, budget, endpoint,
failure policy, actual Git provenance, and prior-observation disclosure.

Usage (from ``orchestrator``)::

    python strong_coupling_showdown.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from intervention_test import balanced_topk_direction
from optimize import volume_project
from pipeline import ColdPlate, Params

HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / "protocols" / "strong_coupling_showdown_v1.json"


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Load and validate the locked experiment definition."""
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    p = protocol.get("parameters", {})
    required = {
        "N", "Ra", "Pr", "seed", "methods", "iterations",
        "fraction_each_way", "amplitude", "beta",
    }
    missing = sorted(required - p.keys())
    if missing:
        raise ValueError(f"protocol is missing parameters: {', '.join(missing)}")
    if p["methods"] != ["composed", "one_way", "frozen"]:
        raise ValueError("v1 protocol methods must be composed, one_way, frozen")
    if type(p["N"]) is not int or p["N"] <= 0 or type(p["seed"]) is not int:
        raise ValueError("N must be positive and seed must be an integer")
    for key in ("Ra", "Pr", "beta"):
        value = p[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or value <= 0):
            raise ValueError(f"{key} must be finite and positive")
    if type(p["iterations"]) is not int or p["iterations"] != 8:
        raise ValueError("v1 protocol must retain exactly eight outer steps")
    fraction = p["fraction_each_way"]
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or not 0.0 < float(fraction) <= 0.5):
        raise ValueError("fraction_each_way must lie in (0, 0.5]")
    amplitude = p["amplitude"]
    if (isinstance(amplitude, bool) or not isinstance(amplitude, (int, float))
            or not math.isfinite(float(amplitude)) or amplitude <= 0.0):
        raise ValueError("amplitude must be finite and positive")
    if protocol.get("status") != "retrospectively_frozen_design":
        raise ValueError("protocol must truthfully identify its retrospective frozen status")
    if protocol.get("outer_steps") != 8:
        raise ValueError("outer_steps must remain fixed at eight")
    provenance = protocol.get("frozen_design_provenance", {})
    commit = provenance.get("commit", "")
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)):
        raise ValueError("protocol must record the full frozen-design commit")
    disclosure = protocol.get("prior_observation_disclosure", {})
    if not isinstance(disclosure.get("observed_before_freeze"), str):
        raise ValueError("protocol must disclose prior observations")
    equivalence = protocol.get("endpoints", {}).get("numerical_equivalence", {})
    for key in ("absolute_J", "relative_J"):
        value = equivalence.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0):
            raise ValueError(f"numerical equivalence {key} must be finite and non-negative")
    return protocol


def initial_design(parameters: dict[str, Any]) -> np.ndarray:
    """Recreate the common design using the RNG and distribution on record."""
    n = int(parameters["N"])
    rng = np.random.default_rng(int(parameters["seed"]))
    return rng.uniform(0.25, 0.75, size=(n, n))


def trajectory_metrics(objectives: list[float]) -> dict[str, float | int]:
    """Compute protocol endpoints from initial + post-action objectives."""
    if not objectives:
        raise ValueError("at least one objective is required")
    values = np.asarray(objectives, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("objective trajectory must be finite")
    first, last = float(values[0]), float(values[-1])
    if first == 0.0:
        raise ValueError("initial objective must be non-zero")
    improving = sum(b < a for a, b in zip(objectives, objectives[1:]))
    # Unit-spaced trapezoidal AUC; lower is better and all branches have the
    # same planned number of outer decisions.
    auc = float(np.trapezoid(values))
    return {
        "initial_J": first,
        "final_J": last,
        "reduction_percent": 100.0 * (first - last) / first,
        "trajectory_auc": auc,
        "improving_steps": improving,
    }


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_branch(method: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Run one sensitivity branch under the common fixed outer budget."""
    n = int(parameters["N"])
    iterations = int(parameters["iterations"])
    amplitude = float(parameters["amplitude"])
    fraction = float(parameters["fraction_each_way"])
    rho = initial_design(parameters)
    k_cells = max(1, int(round(fraction * rho.size)))
    params = Params(
        Nx=n,
        Ny=n,
        Ra=float(parameters["Ra"]),
        Pr=float(parameters["Pr"]),
        beta=float(parameters["beta"]),
    )

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    objectives: list[float] = []

    with ColdPlate(params=params) as cp:
        target_volume = float(np.mean(np.asarray(cp.material(rho)["rho_phys"])))
        for iteration in range(1, iterations + 1):
            gradient_started = time.perf_counter()
            try:
                result = cp.value_and_grad(rho, mode=method)
            except Exception as exc:  # solver/transport failures are experiment data
                failure = {
                    "stage": "gradient",
                    "iteration": iteration,
                    "reason": f"{type(exc).__name__}: {str(exc)[:240]}",
                }
                break
            if not result["info"]["ok"]:
                failure = {
                    "stage": "base_forward",
                    "iteration": iteration,
                    "reason": "coupled state did not converge",
                    "residual": float(result["info"]["residual"]),
                }
                break

            current_J = float(result["J"])
            if not math.isfinite(current_J):
                failure = {
                    "stage": "base_objective",
                    "iteration": iteration,
                    "reason": "coupled base objective is non-finite",
                }
                break
            if not objectives:
                objectives.append(current_J)
            elif not np.isclose(current_J, objectives[-1], rtol=2e-8, atol=2e-10):
                failure = {
                    "stage": "repeat_forward_consistency",
                    "iteration": iteration,
                    "reason": (
                        f"warm repeat gave J={current_J:.12g}, previous true "
                        f"candidate was {objectives[-1]:.12g}"
                    ),
                }
                break

            gradient_seconds = time.perf_counter() - gradient_started
            try:
                direction, add, remove = balanced_topk_direction(result["grad"], k_cells)
                proposal = np.clip(rho + amplitude * direction, 0.0, 1.0)
                proposal = volume_project(cp, proposal, target_volume)
                realised_volume = float(
                    np.mean(np.asarray(cp.material(proposal)["rho_phys"]))
                )
                predicted_delta = float(
                    np.sum(np.asarray(result["grad"]) * (proposal - rho))
                )
            except Exception as exc:  # noqa: BLE001 - proposal failure stays visible
                failure = {
                    "stage": "proposal_construction",
                    "iteration": iteration,
                    "reason": f"{type(exc).__name__}: {str(exc)[:240]}",
                }
                break
            proposal_bytes = np.asarray(proposal, dtype="<f8", order="C").tobytes()
            proposal_hash = hashlib.sha256(proposal_bytes).hexdigest()
            proposal_record: dict[str, Any] = {
                "iteration": iteration,
                "status": "proposed",
                "raw_design_sha256": proposal_hash,
                "raw_design_shape": list(proposal.shape),
                "raw_design_min": float(np.min(proposal)),
                "raw_design_max": float(np.max(proposal)),
                "J_before": current_J,
                "predicted_first_order_delta": predicted_delta,
                "projected_volume": realised_volume,
                "volume_error": realised_volume - target_volume,
                "add_cells": int(add.size),
                "remove_cells": int(remove.size),
                "gradient_seconds": gradient_seconds,
                "base_residual": float(result["info"]["residual"]),
            }

            candidate_started = time.perf_counter()
            cp._T_warm = np.asarray(result["T"])
            try:
                material = cp.material(proposal)
                candidate_T, info = cp.solve_coupled(material["alpha"], material["k"])
            except Exception as exc:  # noqa: BLE001 - a failed branch stays in the data
                proposal_record.update({
                    "status": "candidate_exception",
                    "candidate_forward_seconds": time.perf_counter() - candidate_started,
                    "failure_reason": f"{type(exc).__name__}: {str(exc)[:240]}",
                    "failed_raw_design": proposal.tolist(),
                })
                proposals.append(proposal_record)
                failure = {
                    "stage": "candidate_forward",
                    "iteration": iteration,
                    "reason": proposal_record["failure_reason"],
                    "proposal_raw_design_sha256": proposal_hash,
                }
                break
            candidate_seconds = time.perf_counter() - candidate_started
            if not info["ok"]:
                proposal_record.update({
                    "status": "candidate_not_converged",
                    "candidate_forward_seconds": candidate_seconds,
                    "candidate_residual": float(info["residual"]),
                    "failure_reason": "coupled candidate did not converge",
                    "failed_raw_design": proposal.tolist(),
                })
                proposals.append(proposal_record)
                failure = {
                    "stage": "candidate_forward",
                    "iteration": iteration,
                    "reason": "coupled candidate did not converge",
                    "residual": float(info["residual"]),
                    "proposal_raw_design_sha256": proposal_hash,
                }
                break

            try:
                candidate_J = float(cp.objective(candidate_T))
                if not math.isfinite(candidate_J):
                    raise ValueError("candidate objective is non-finite")
            except Exception as exc:  # noqa: BLE001 - objective failure stays visible
                proposal_record.update({
                    "status": "candidate_objective_failure",
                    "candidate_forward_seconds": candidate_seconds,
                    "candidate_residual": float(info["residual"]),
                    "failure_reason": f"{type(exc).__name__}: {str(exc)[:240]}",
                    "failed_raw_design": proposal.tolist(),
                })
                proposals.append(proposal_record)
                failure = {
                    "stage": "candidate_objective",
                    "iteration": iteration,
                    "reason": proposal_record["failure_reason"],
                    "proposal_raw_design_sha256": proposal_hash,
                }
                break
            proposal_record.update({
                "status": "accepted",
                "candidate_forward_seconds": candidate_seconds,
                "candidate_residual": float(info["residual"]),
                "J_after": candidate_J,
                "delta_J": candidate_J - current_J,
            })
            proposals.append(proposal_record)
            rows.append(dict(proposal_record))
            rho = proposal
            objectives.append(candidate_J)

        stats = {name: int(value) for name, value in cp.stats.items()}

    output: dict[str, Any] = {
        "method": method,
        "planned_iterations": iterations,
        "completed_iterations": len(rows),
        "complete": len(rows) == iterations and failure is None,
        "failure": failure,
        "k_cells_each_way": k_cells,
        "target_projected_volume": target_volume,
        "objectives": objectives,
        "rows": rows,
        "proposals": proposals,
        "component_stats": stats,
        "wall_seconds": time.perf_counter() - started,
        "final_rho_raw": rho.tolist(),
    }
    if objectives:
        output["metrics"] = trajectory_metrics(objectives)
    return output


def _same_number(left: object, right: object, *, rtol: float = 2.0e-12,
                 atol: float = 2.0e-12) -> bool:
    """Compare finite scalar evidence without letting booleans pass as numbers."""
    if (isinstance(left, bool) or isinstance(right, bool)
            or not isinstance(left, (int, float))
            or not isinstance(right, (int, float))):
        return False
    return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
        float(left), float(right), rel_tol=rtol, abs_tol=atol
    )


def _branch_fully_complete(branch: dict[str, Any], iterations: int) -> bool:
    """Require a self-consistent initial state plus every planned accepted step."""
    rows = branch.get("rows")
    proposals = branch.get("proposals")
    objectives = branch.get("objectives")
    metrics = branch.get("metrics")
    structural = (
        branch.get("complete") is True
        and branch.get("failure") is None
        and branch.get("planned_iterations") == iterations
        and branch.get("completed_iterations") == iterations
        and isinstance(rows, list)
        and len(rows) == iterations
        and [row.get("iteration") for row in rows] == list(range(1, iterations + 1))
        and all(row.get("status") == "accepted" for row in rows)
        and isinstance(proposals, list)
        and len(proposals) == iterations
        and all(proposal.get("status") == "accepted" for proposal in proposals)
        and isinstance(objectives, list)
        and len(objectives) == iterations + 1
        and isinstance(metrics, dict)
    )
    if not structural or rows != proposals:
        return False
    if not all(_same_number(value, value) for value in objectives):
        return False
    for index, row in enumerate(rows):
        if not (
            _same_number(row.get("J_before"), objectives[index])
            and _same_number(row.get("J_after"), objectives[index + 1])
            and _same_number(
                row.get("delta_J"), float(objectives[index + 1]) - float(objectives[index])
            )
            and isinstance(row.get("raw_design_sha256"), str)
            and len(row["raw_design_sha256"]) == 64
        ):
            return False
    try:
        recomputed = trajectory_metrics([float(value) for value in objectives])
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    return all(_same_number(metrics.get(key), value) for key, value in recomputed.items())


def summarize(branches: list[dict[str, Any]], iterations: int = 8,
              absolute_tolerance: float = 1.0e-10,
              relative_tolerance: float = 1.0e-8) -> dict[str, Any]:
    """Rank only fully completed branches and apply numerical equivalence."""
    expected_methods = {"composed", "one_way", "frozen"}
    methods = [branch.get("method") for branch in branches]
    structurally_complete = (
        len(branches) == len(expected_methods)
        and set(methods) == expected_methods
        and len(set(methods)) == len(methods)
    )
    completed = [
        branch for branch in branches if _branch_fully_complete(branch, iterations)
    ]
    ranking = sorted(
        (
            {
                "method": branch["method"],
                "final_J": branch["metrics"]["final_J"],
                "reduction_percent": branch["metrics"]["reduction_percent"],
            }
            for branch in completed
        ),
        key=lambda row: row["final_J"],
    )
    finals = {row["method"]: row["final_J"] for row in ranking}
    branch_steps_complete = structurally_complete and len(completed) == len(expected_methods)
    initial_values = [branch["metrics"]["initial_J"] for branch in completed]
    common_initial = branch_steps_complete and all(
        abs(float(value) - float(initial_values[0]))
        <= absolute_tolerance + relative_tolerance * max(
            abs(float(value)), abs(float(initial_values[0]))
        )
        for value in initial_values[1:]
    )
    all_complete = branch_steps_complete and common_initial
    comparisons = []
    if all_complete:
        for method in ("one_way", "frozen"):
            margin = finals[method] - finals["composed"]
            tolerance = absolute_tolerance + relative_tolerance * max(
                abs(finals[method]), abs(finals["composed"])
            )
            relation = (
                "composed_lower" if margin > tolerance
                else "numerically_equivalent" if abs(margin) <= tolerance
                else "composed_higher"
            )
            comparisons.append({
                "other_method": method,
                "other_minus_composed_J": margin,
                "equivalence_tolerance": tolerance,
                "relation": relation,
            })
    composed_wins = all_complete and all(
        row["relation"] == "composed_lower" for row in comparisons
    )
    return {
        "all_branches_complete": all_complete,
        "completed_branches": len(completed),
        "required_iterations_per_branch": iterations,
        "common_initial_objective_verified": common_initial,
        "ranking": ranking,
        "final_objective_comparisons": comparisons,
        "frozen_success_condition_met": composed_wins,
    }


def main(protocol_path: str = str(DEFAULT_PROTOCOL), out: str | None = None) -> int:
    protocol_file = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_file)
    parameters = protocol["parameters"]
    if out is None:
        # Result paths in protocols are repository-root relative.
        destination = HERE.parent / protocol["result_path"]
    else:
        destination = Path(out).resolve()

    payload: dict[str, Any] = {
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol_file.read_bytes()).hexdigest(),
        "protocol_file": str(protocol_file.relative_to(HERE.parent)).replace("\\", "/"),
        "complete": False,
        "branches": [],
    }
    _json_write(destination, payload)

    print("retrospectively frozen strong-coupling optimisation showdown")
    print(
        f"N={parameters['N']}, Ra={parameters['Ra']:.0f}, "
        f"steps={parameters['iterations']}, amplitude={parameters['amplitude']}"
    )
    for method in parameters["methods"]:
        print(f"\n=== {method} ===", flush=True)
        try:
            branch = run_branch(method, parameters)
        except Exception as exc:  # noqa: BLE001 - preserve terminal runner failure
            branch = {
                "method": method,
                "planned_iterations": int(parameters["iterations"]),
                "completed_iterations": 0,
                "complete": False,
                "failure": {
                    "stage": "branch_runner_exception",
                    "reason": f"{type(exc).__name__}: {str(exc)[:400]}",
                },
                "objectives": [],
                "rows": [],
                "proposals": [],
            }
        payload["branches"].append(branch)
        _json_write(destination, payload)
        if branch.get("metrics"):
            metrics = branch["metrics"]
            print(
                f"{branch['completed_iterations']}/{branch['planned_iterations']} steps; "
                f"J {metrics['initial_J']:.8f} -> {metrics['final_J']:.8f} "
                f"({metrics['reduction_percent']:.2f}% reduction)"
            )
        if branch["failure"]:
            print(f"recorded failure: {branch['failure']}")

    equivalence = protocol["endpoints"]["numerical_equivalence"]
    payload["summary"] = summarize(
        payload["branches"],
        iterations=int(parameters["iterations"]),
        absolute_tolerance=float(equivalence["absolute_J"]),
        relative_tolerance=float(equivalence["relative_J"]),
    )
    payload["complete"] = payload["summary"]["all_branches_complete"]
    _json_write(destination, payload)
    print(f"\nwrote {destination}")
    print(json.dumps(payload["summary"], indent=2))
    # A contrary scientific result is valid output, not a process error. Only
    # malformed/incomplete execution raises; the result audit decides claims.
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--out")
    raise SystemExit(main(**vars(parser.parse_args())))
