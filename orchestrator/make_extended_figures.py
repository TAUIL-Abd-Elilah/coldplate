# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Render the frozen showdown, robustness, and physics evidence.

The three source JSON files are produced by ``extended submission evidence``.
This script has no fallback numbers: a missing result is an error, preventing a
video or README figure from silently using a stale hand-entered claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dimensional_coldplate import base_layout, finned_layout

TEAL = "#0f9d8a"
NAVY = "#12233f"
ORANGE = "#e48b32"
RED = "#c94a53"
GREY = "#8a94a6"


def _load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"run the extended evidence workflow first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def showdown_figure(data: dict, path: Path) -> None:
    expected_steps = int(data["protocol"]["outer_steps"])
    branches = data.get("branches", [])
    if len(branches) != 3 or {branch.get("method") for branch in branches} != {
        "composed", "one_way", "frozen"
    }:
        raise ValueError("showdown must contain the three frozen methods exactly once")
    completed = [int(branch.get("completed_iterations", -1)) for branch in branches]
    if any(value < 0 or len(branch.get("objectives", [])) != value + 1
           for value, branch in zip(completed, branches)):
        raise ValueError("showdown trajectories do not match their accepted-step counts")
    initial = [float(branch["objectives"][0]) for branch in branches]
    if max(initial) - min(initial) > 1e-12:
        raise ValueError("showdown branches do not share the measured initial objective")
    all_complete = (
        data.get("complete") is True
        and data.get("summary", {}).get("all_branches_complete") is True
        and all(value == expected_steps for value in completed)
    )
    common_horizon = expected_steps if all_complete else min(completed)
    if common_horizon <= 0:
        raise ValueError("showdown has no common accepted-decision horizon")
    colors = {"composed": TEAL, "one_way": ORANGE, "frozen": RED}
    labels = {"composed": "composed adjoint", "one_way": "loop cut", "frozen": "flow frozen"}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), gridspec_kw={"width_ratios": [1.65, 1]})
    ax, bars = axes
    reductions, names, bar_colors = [], [], []
    for branch in data["branches"]:
        objectives = np.asarray(branch["objectives"], dtype=float)
        improvement = 100.0 * (objectives[0] - objectives) / objectives[0]
        steps = np.arange(len(objectives))
        method = branch["method"]
        shared = slice(0, common_horizon + 1)
        ax.plot(steps[shared], improvement[shared], marker="o", ms=5, lw=2.5,
                color=colors[method], label=labels[method])
        if len(steps) > common_horizon + 1:
            ax.plot(
                steps[common_horizon:], improvement[common_horizon:],
                marker="o", ms=4, lw=1.7, ls="--", alpha=0.45,
                color=colors[method],
            )
        if branch.get("failure"):
            failed_step = int(branch["failure"].get("iteration", steps[-1]))
            ax.axvline(failed_step, ls=":", lw=1.4, color=colors[method])
            ax.annotate(
                f"candidate {failed_step}: no converged J",
                (failed_step, improvement[-1]), xytext=(8, -18),
                textcoords="offset points", fontsize=8, color=colors[method],
            )
        names.append(labels[method])
        reductions.append(float(improvement[common_horizon]))
        bar_colors.append(colors[method])
    ax.axhline(0, color="#c7ccd5", lw=1)
    ax.set_xlabel("common-rule design decision")
    ax.set_ylabel("true coupled objective reduction (%)")
    ax.set_title("Fresh forward solve after every decision", loc="left", weight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#e8ebf0", lw=0.8)

    order = np.arange(len(names))
    bars.barh(order, reductions, color=bar_colors, height=0.62)
    bars.set_yticks(order, names)
    bars.invert_yaxis()
    bars.set_xlabel(f"reduction after common first {common_horizon} decisions (%)")
    bars.set_title(
        "Frozen endpoint" if all_complete else "Post-hoc common-prefix description",
        loc="left", weight="bold",
    )
    bars.grid(axis="x", color="#e8ebf0", lw=0.8)
    for y, value in zip(order, reductions):
        bars.text(max(value, 0) + 0.15, y, f"{value:.2f}%", va="center", weight="bold")
    title = (
        "Strong-coupling optimisation showdown"
        if all_complete
        else "Frozen showdown stopped by solver failure · no eight-step verdict"
    )
    fig.suptitle(title, x=0.055, ha="left", fontsize=17, weight="bold", color=NAVY)
    if not all_complete:
        fig.text(
            0.055, 0.01,
            "Bars compare only the shared five accepted decisions; this descriptive "
            "prefix was selected after the recorded failure.",
            fontsize=8.5, color=RED,
        )
    fig.tight_layout(rect=(0, 0.04 if not all_complete else 0, 1, 0.95))
    _save(fig, path)


def robustness_figure(data: dict, path: Path) -> None:
    groups = data["by_rayleigh_number"]
    pooled = data["summary"]
    if not pooled.get("study_complete"):
        raise ValueError("robustness matrix is incomplete; refusing to render a finish")
    cluster = pooled["cluster_aware_seed_analysis"]
    if not cluster.get("all_planned_clusters_complete"):
        raise ValueError("seed-cluster analysis is incomplete")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax, stack = axes
    labels = [f"{row['Ra']:.0e}".replace("e+0", "e") for row in groups] + ["pooled\nclustered"]
    rows = groups
    rates, lower, upper = [], [], []
    for row in rows:
        interval = row["exact_win_rate_over_comparable"]
        estimate = interval["estimate"] if interval["estimate"] is not None else 0.0
        rates.append(100 * estimate)
        lower.append(100 * (estimate - (interval["lower"] or 0.0)))
        upper.append(100 * ((interval["upper"] or estimate) - estimate))
    cluster_interval = cluster["bootstrap"]
    cluster_estimate = cluster["pooled_exact_win_rate_over_comparable_in_complete_clusters"]
    if cluster_estimate is None or cluster_interval["lower"] is None or cluster_interval["upper"] is None:
        raise ValueError("cluster bootstrap has no comparable-case interval")
    rates.append(100 * cluster_estimate)
    lower.append(100 * max(0.0, cluster_estimate - cluster_interval["lower"]))
    upper.append(100 * max(0.0, cluster_interval["upper"] - cluster_estimate))
    x = np.arange(len(labels))
    ax.errorbar(x, rates, yerr=np.vstack([lower, upper]), fmt="o", ms=8,
                capsize=5, lw=2, color=TEAL)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("exact action wins among comparable cases (%)")
    ax.set_title("Per-Ra Wilson · pooled seed-cluster bootstrap (95%)", loc="left", weight="bold")
    ax.grid(axis="y", color="#e8ebf0", lw=0.8)
    display_rows = groups + [pooled]
    for xx, row, rate in zip(x, display_rows, rates):
        ax.text(xx, min(rate + 5, 100), f"{row['outcomes']['exact_wins']}/{row['comparable_cases']}",
                ha="center", fontsize=9, weight="bold")

    outcome_keys = ["exact_wins", "shortcut_wins", "tie", "inconclusive", "base_not_converged", "runner_failure"]
    outcome_labels = ["exact wins", "shortcut wins", "ties", "inconclusive", "base failed", "runner failed"]
    colors = [TEAL, ORANGE, GREY, "#c6a55b", RED, "#6d4c7d"]
    left = 0
    for key, label, color in zip(outcome_keys, outcome_labels, colors):
        count = pooled["outcomes"].get(key, 0)
        if count:
            stack.barh([0], [count], left=left, label=f"{label}: {count}", color=color, height=0.45)
            left += count
    stack.set_xlim(0, pooled["attempts_planned"])
    stack.set_yticks([])
    stack.set_xlabel("all frozen-protocol attempts")
    stack.set_title("Every outcome remains in the denominator audit", loc="left", weight="bold")
    stack.legend(
        frameon=True, facecolor="white", framealpha=0.92, edgecolor="none",
        loc="center left", bbox_to_anchor=(0.02, 0.38), ncol=1,
    )
    stack.grid(axis="x", color="#e8ebf0", lw=0.8)
    fig.suptitle("48-case robustness matrix · 16 fixed seeds at each coupling level",
                 x=0.055, ha="left", fontsize=17, weight="bold", color=NAVY)
    fig.tight_layout()
    _save(fig, path)


def physics_figure(cavity: list[dict], physical: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), gridspec_kw={"width_ratios": [1.25, 0.8, 1]})
    compare, layouts, audit = axes
    if not cavity or any(
        not row["solver"]["ok"] or not row["solver"]["fluid"]["converged"]
        for row in cavity
    ):
        raise ValueError("the nonlinear cavity benchmark is not converged")
    metric_names = ["Nu_mean", "u_max", "v_max"]
    markers = ["o", "s", "^"]
    for metric, marker in zip(metric_names, markers):
        reference = [row["reference"][metric] for row in cavity]
        measured = [row[metric] for row in cavity]
        compare.plot(reference, measured, marker=marker, ms=7, lw=1.8, label=metric.replace("_", " "))
    max_value = max(
        max(max(row["reference"].values()), *(row[name] for name in metric_names))
        for row in cavity
    ) * 1.12
    compare.plot([0, max_value], [0, max_value], "--", color=GREY, lw=1.2, label="perfect agreement")
    compare.set_xlim(0, max_value)
    compare.set_ylim(0, max_value)
    compare.set_xlabel("de Vahl Davis reference")
    compare.set_ylabel("composed solver")
    compare.set_title("Nonlinear cavity benchmark", loc="left", weight="bold")
    compare.legend(frameon=False, fontsize=8)
    compare.grid(color="#e8ebf0", lw=0.8)

    N = int(physical["grid"]["Nx"])
    canvas = np.concatenate([base_layout(N), np.ones((N, 2)), finned_layout(N)], axis=1)
    layouts.imshow(canvas, origin="lower", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    layouts.set_xticks([N / 2, N + 2 + N / 2], ["base", "+ four fins"])
    layouts.set_yticks([])
    layouts.set_title("5 × 5 × 2 mm\nwater / aluminium", weight="bold")

    mesh = physical["mesh_refinement"]
    grids = [int(value) for value in mesh["grids"]]
    rows = {int(row["N"]): row for row in mesh["rows"]}
    methods = ("baseline", "finned")
    status = np.asarray([
        [bool(rows[grid]["layouts"][method]["solver"]["ok"]) for grid in grids]
        for method in methods
    ], dtype=float)
    audit.imshow(status, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    audit.set_xticks(np.arange(len(grids)), [f"N={grid}" for grid in grids])
    audit.set_yticks(np.arange(2), ["base only", "+ four fins"])
    audit.set_title("Dimensional solve audit", loc="left", weight="bold")
    for row_index, method in enumerate(methods):
        for column_index, grid in enumerate(grids):
            solver = rows[grid]["layouts"][method]["solver"]
            label = "converged" if solver["ok"] else "not\nconverged"
            audit.text(column_index, row_index, label, ha="center", va="center",
                       color="white", fontsize=8,
                       weight="bold")
    valid_count = int(status.sum())
    total_count = int(status.size)
    audit.text(
        0.5, -0.20,
        f"exact 1 W load · {valid_count}/{total_count} solves converged\n"
        "comparison withheld · temperatures leave the liquid-water model regime",
        transform=audit.transAxes, ha="center", va="top", color=NAVY, fontsize=8,
    )
    fig.suptitle("Recognised nonlinear reference + explicit SI convergence audit",
                 x=0.045, ha="left", fontsize=17, weight="bold", color=NAVY)
    fig.tight_layout()
    _save(fig, path)


def main(results_dir="results") -> None:
    results = Path(results_dir)
    showdown_figure(
        _load(results / "strong_coupling_showdown.json"),
        results / "fig12_showdown.png",
    )
    robustness_figure(
        _load(results / "intervention_robustness_matrix_48.json"),
        results / "fig13_robustness_matrix.png",
    )
    physics_figure(
        _load(results / "de_vahl_davis.json"),
        _load(results / "dimensional_coldplate.json"),
        results / "fig14_physics_validation.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    main(**vars(parser.parse_args()))
