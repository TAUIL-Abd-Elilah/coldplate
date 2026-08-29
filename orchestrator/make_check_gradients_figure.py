# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Draw what Tesseract's own gradient checker found.

Left: the step-size ladder. For each differentiable input of each component,
`check_gradients.py` records the largest disagreement between the derivative
endpoint and a central difference through `apply`, divided by the largest entry
of the endpoint's own row. Plotted against the step, a correct derivative has to
show two regimes -- an O(eps^2) truncation leg where the difference scheme is
still converging, and a floor where the solver's own convergence noise takes
over. The leg is the part that carries the argument: a derivative that were
merely *close* would flatten out early at its own error instead of tracking the
difference scheme down.

Right: the two findings side by side. Three components agree on every live
comparison. The JAX thermal block also reports derivatives with respect to
wall-face velocities that its forward assembly never reads, and the
independently written Fortran block does not -- which is what localises that to
a defect rather than a modelling choice.

    usage:  python orchestrator/make_check_gradients_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1b1f24"
MUTED = "#6b7480"
GRID = "#dfe3e8"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "legend.frameon": False,
    "figure.dpi": 140,
})

ORDER = ("stokes_brinkman", "thermal_advdiff", "thermal_fortran", "material_map")
COLOUR = {
    "stokes_brinkman": "#c2410c",
    "thermal_advdiff": "#0369a1",
    "thermal_fortran": "#15803d",
    "material_map": "#7c3aed",
}
MARKERS = ("o", "s", "^", "D")
FLOOR = 1e-16  # so exact agreement is drawable on a log axis


def main() -> int:
    root = Path(__file__).resolve().parent
    report = json.loads((root / "results" / "check_gradients.json").read_text())

    fig, (ax, bar) = plt.subplots(
        1, 2, figsize=(11.4, 4.3), gridspec_kw={"width_ratios": [1.75, 1.0]}
    )

    for name in ORDER:
        component = report["components"][name]
        for marker, (path, entry) in zip(
            MARKERS, sorted(component["input_paths"].items()), strict=False
        ):
            rungs = entry["by_rel_eps"]
            steps, values = [], []
            for rel in sorted(float(r) for r in rungs):
                rung = rungs[repr(rel)]
                if rung["live_rows"] == 0:
                    continue  # nothing above the checker's floor to compare
                steps.append(rel)
                values.append(max(rung["relative_disagreement_live"], FLOOR))
            if not steps:
                continue
            ax.plot(
                steps, values, marker=marker, markersize=4.5, linewidth=1.4,
                color=COLOUR[name], alpha=0.9, label=f"{name}.{path}",
            )

    lo = min(float(r) for r in report["rel_eps_ladder"])
    hi = max(float(r) for r in report["rel_eps_ladder"])
    anchor = 1e-5
    guide = np.array([lo, hi])
    ax.plot(guide, anchor * (guide / hi) ** 2, color=MUTED, linestyle=":",
            linewidth=1.2, zorder=0)
    ax.annotate(r"slope 2: $\epsilon^2$ truncation",
                xy=(lo * 4, anchor * (lo * 4 / hi) ** 2),
                xytext=(10, 4), textcoords="offset points",
                ha="left", color=MUTED, fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("finite-difference step, relative to the input's own magnitude")
    ax.set_ylabel("relative disagreement (live comparisons)")
    ax.set_title("Truncation-limited, or noise-limited: both are the right shape")
    ax.legend(fontsize=7.5, ncol=2, loc="lower left")
    ax.grid(True, which="major", color=GRID, linewidth=0.6)

    names = list(ORDER)
    disagreement = [report["components"][n]["relative_disagreement"] for n in names]
    phantom = [report["components"][n]["phantom_rows"] for n in names]
    y = np.arange(len(names))

    bar.barh(y, [max(d, FLOOR) for d in disagreement],
             color=[COLOUR[n] for n in names], height=0.5)
    bar.set_yticks(y)
    bar.set_yticklabels(names, fontsize=8.5)
    bar.invert_yaxis()
    bar.set_xscale("log")
    bar.set_xlabel("worst live relative disagreement")
    bar.set_title("Agreement, and what it does not cover")
    for index, (value, count) in enumerate(zip(disagreement, phantom, strict=True)):
        label = f"{value:.1e}"
        if count:
            label += f"   +{count} phantom"
        bar.annotate(label, xy=(max(value, FLOOR), index), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=8.5,
                     color="#b91c1c" if count else INK,
                     fontweight="bold" if count else "normal")
    bar.grid(True, axis="x", which="major", color=GRID, linewidth=0.6)
    bar.set_xlim(right=bar.get_xlim()[1] * 60)

    fig.suptitle(
        f"Tesseract's own gradient checker, four components, "
        f"{report['N']}x{report['N']} at the converged coupled state — "
        f"{report['total_comparisons']} comparisons",
        fontsize=11.5, y=1.02,
    )
    out = root / "results" / "fig15_check_gradients.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
