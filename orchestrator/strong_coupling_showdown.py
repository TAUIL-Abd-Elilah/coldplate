# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Run the preregistered strong-coupling optimisation showdown.

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
The JSON protocol fixes the seed, coupling, action, budget, endpoint, and
failure policy before the result is generated.

Usage (from ``orchestrator``)::

    python strong_coupling_showdown.py
"""

from __future__ import annotations

import argparse
import json
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
    if int(p["iterations"]) < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 < float(p["fraction_each_way"]) <= 0.5:
        raise ValueError("fraction_each_way must lie in (0, 0.5]")
    if float(p["amplitude"]) <= 0.0:
        raise ValueError("amplitude must be positive")
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
    first, last = float(objectives[0]), float(objectives[-1])
    improving = sum(b < a for a, b in zip(objectives, objectives[1:]))
    # Unit-spaced trapezoidal AUC; lower is better and all branches have the
    # same planned number of outer decisions.
    auc = float(np.trapezoid(np.asarray(objectives, dtype=np.float64)))
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
            direction, add, remove = balanced_topk_direction(result["grad"], k_cells)
            proposal = np.clip(rho + amplitude * direction, 0.0, 1.0)
            proposal = volume_project(cp, proposal, target_volume)
            realised_volume = float(
                np.mean(np.asarray(cp.material(proposal)["rho_phys"]))
            )

            candidate_started = time.perf_counter()
            cp._T_warm = np.asarray(result["T"])
            try:
                material = cp.material(proposal)
                candidate_T, info = cp.solve_coupled(material["alpha"], material["k"])
            except Exception as exc:  # noqa: BLE001 - a failed branch stays in the data
                failure = {
                    "stage": "candidate_forward",
                    "iteration": iteration,
                    "reason": f"{type(exc).__name__}: {str(exc)[:240]}",
                }
                break
            candidate_seconds = time.perf_counter() - candidate_started
            if not info["ok"]:
                failure = {
                    "stage": "candidate_forward",
                    "iteration": iteration,
                    "reason": "coupled candidate did not converge",
                    "residual": float(info["residual"]),
                }
                break

            candidate_J = float(cp.objective(candidate_T))
            rows.append(
                {
                    "iteration": iteration,
                    "J_before": current_J,
                    "J_after": candidate_J,
                    "delta_J": candidate_J - current_J,
                    "predicted_first_order_delta": float(
                        np.sum(np.asarray(result["grad"]) * (proposal - rho))
                    ),
                    "projected_volume": realised_volume,
                    "volume_error": realised_volume - target_volume,
                    "add_cells": int(add.size),
                    "remove_cells": int(remove.size),
                    "gradient_seconds": gradient_seconds,
                    "candidate_forward_seconds": candidate_seconds,
                    "base_residual": float(result["info"]["residual"]),
                    "candidate_residual": float(info["residual"]),
                }
            )
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
        "component_stats": stats,
        "wall_seconds": time.perf_counter() - started,
        "final_rho_raw": rho.tolist(),
    }
    if objectives:
        output["metrics"] = trajectory_metrics(objectives)
    return output


def summarize(branches: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank completed branches without suppressing failed outcomes."""
    completed = [branch for branch in branches if branch["complete"]]
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
    composed_wins = (
        len(completed) == len(branches)
        and "composed" in finals
        and all(finals["composed"] < value for method, value in finals.items()
                if method != "composed")
    )
    return {
        "all_branches_complete": len(completed) == len(branches),
        "completed_branches": len(completed),
        "ranking": ranking,
        "preregistered_success_condition_met": composed_wins,
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
        "protocol_file": str(protocol_file.relative_to(HERE.parent)).replace("\\", "/"),
        "complete": False,
        "branches": [],
    }
    _json_write(destination, payload)

    print("preregistered strong-coupling optimisation showdown")
    print(
        f"N={parameters['N']}, Ra={parameters['Ra']:.0f}, "
        f"steps={parameters['iterations']}, amplitude={parameters['amplitude']}"
    )
    for method in parameters["methods"]:
        print(f"\n=== {method} ===", flush=True)
        branch = run_branch(method, parameters)
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

    payload["summary"] = summarize(payload["branches"])
    payload["complete"] = True
    _json_write(destination, payload)
    print(f"\nwrote {destination}")
    print(json.dumps(payload["summary"], indent=2))
    # A contrary scientific result is valid output, not a process error. Only
    # malformed/incomplete execution raises; the result audit decides claims.
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--out")
    raise SystemExit(main(**vars(parser.parse_args())))
