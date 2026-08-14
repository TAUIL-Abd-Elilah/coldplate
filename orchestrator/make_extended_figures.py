# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Render the preregistered showdown, robustness, and physics evidence.

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
        ax.plot(steps, improvement, marker="o", ms=5, lw=2.5,
                color=colors[method], label=labels[method])
        if branch.get("failure"):
            ax.scatter(steps[-1], improvement[-1], marker="x", s=90, color=colors[method])
        names.append(labels[method])
        reductions.append(float(improvement[-1]))
        bar_colors.append(colors[method])
    ax.axhline(0, color="#c7ccd5", lw=1)
    ax.set_xlabel("equal-budget design decision")
    ax.set_ylabel("true coupled objective reduction (%)")
    ax.set_title("Fresh forward solve after every decision", loc="left", weight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#e8ebf0", lw=0.8)

    order = np.arange(len(names))
    bars.barh(order, reductions, color=bar_colors, height=0.62)
    bars.set_yticks(order, names)
    bars.invert_yaxis()
    bars.set_xlabel("final reduction (%)")
    bars.set_title("Same start · same update · same outer budget", loc="left", weight="bold")
    bars.grid(axis="x", color="#e8ebf0", lw=0.8)
    for y, value in zip(order, reductions):
        bars.text(max(value, 0) + 0.15, y, f"{value:.2f}%", va="center", weight="bold")
    fig.suptitle("Strong-coupling optimisation showdown", x=0.055, ha="left",
                 fontsize=17, weight="bold", color=NAVY)
    fig.tight_layout()
    _save(fig, path)


def robustness_figure(data: dict, path: Path) -> None:
    groups = data["by_rayleigh_number"]
    pooled = data["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax, stack = axes
    labels = [f"{row['Ra']:.0e}".replace("e+0", "e") for row in groups] + ["pooled"]
    rows = groups + [pooled]
    rates, lower, upper = [], [], []
    for row in rows:
        interval = row["exact_win_rate_over_comparable"]
        estimate = interval["estimate"] if interval["estimate"] is not None else 0.0
        rates.append(100 * estimate)
        lower.append(100 * (estimate - (interval["lower"] or 0.0)))
        upper.append(100 * ((interval["upper"] or estimate) - estimate))
    x = np.arange(len(rows))
    ax.errorbar(x, rates, yerr=np.vstack([lower, upper]), fmt="o", ms=8,
                capsize=5, lw=2, color=TEAL)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("exact action wins among comparable cases (%)")
    ax.set_title("Wilson 95% intervals", loc="left", weight="bold")
    ax.grid(axis="y", color="#e8ebf0", lw=0.8)
    for xx, row, rate in zip(x, rows, rates):
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
    stack.set_xlabel("all preregistered attempts")
    stack.set_title("Every outcome remains in the denominator audit", loc="left", weight="bold")
    stack.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, -0.03), ncol=1)
    stack.grid(axis="x", color="#e8ebf0", lw=0.8)
    fig.suptitle("48-case robustness matrix · 16 fixed seeds at each coupling level",
                 x=0.055, ha="left", fontsize=17, weight="bold", color=NAVY)
    fig.tight_layout()
    _save(fig, path)


def physics_figure(cavity: list[dict], physical: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), gridspec_kw={"width_ratios": [1.25, 0.8, 1]})
    compare, layouts, bars = axes
    metric_names = ["Nu_mean", "u_max", "v_max"]
    markers = ["o", "s", "^"]
    for metric, marker in zip(metric_names, markers):
        reference = [row["reference"][metric] for row in cavity]
        measured = [row[metric] for row in cavity]
        compare.plot(reference, measured, marker=marker, ms=7, lw=1.8, label=metric.replace("_", " "))
    max_value = max(max(row["reference"].values()) for row in cavity) * 1.12
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

    values = [physical["layouts"][name]["thermal_resistance_K_W"] for name in ("baseline", "finned")]
    colors = [GREY, TEAL]
    rects = bars.bar(["base only", "+ four fins"], values, color=colors, width=0.62)
    bars.set_ylabel("chip-wall thermal resistance (K/W)")
    bars.set_title("Dimensional 1 W case", loc="left", weight="bold")
    bars.grid(axis="y", color="#e8ebf0", lw=0.8)
    for rect, value in zip(rects, values):
        bars.text(rect.get_x() + rect.get_width() / 2, value, f"{value:.2f}",
                  ha="center", va="bottom", weight="bold")
    change = physical["finned_thermal_resistance_reduction_percent"]
    bars.text(0.5, 0.92, f"{change:.1f}% lower", transform=bars.transAxes,
              ha="center", color=TEAL if change >= 0 else RED, weight="bold")
    fig.suptitle("Recognised physics reference + explicit SI engineering map",
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
