# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Figures and animation for the cold-plate results.

  fig1  hero animation: material layout, temperature + streamlines, convergence
  fig2  gradient validation: composed vs finite differences vs frozen flow
  fig3  coupling strength: loop gain and naive-gradient error vs Rayleigh number
  fig4  optimisation driven by the composed gradient vs the naive one
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# --------------------------------------------------------------------------
# house style
# --------------------------------------------------------------------------

INK = "#1b1f24"
MUTED = "#6b7480"
ACCENT = "#c2410c"  # composed / correct
NAIVE = "#0369a1"  # frozen-flow / naive
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

# solid metal is dark, open coolant is pale
CMAP_RHO = LinearSegmentedColormap.from_list("rho", ["#f7f5f2", "#8a7a68", "#2b2622"])
CMAP_T = LinearSegmentedColormap.from_list(
    "T", ["#f8fafc", "#bae6fd", "#7dd3fc", "#fdba74", "#f97316", "#b91c1c"]
)


def centers(u, v):
    """Face velocities -> cell centres, for streamlines."""
    return 0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1, :] + v[1:, :])


# --------------------------------------------------------------------------


def fig1_animation(npz_path, history_path, out_gif, out_png):
    d = np.load(npz_path)
    hist = json.loads(Path(history_path).read_text())
    rho, T = d["snapshots_rho"], d["snapshots_T"]
    U, V = d["snapshots_u"], d["snapshots_v"]
    iters = d["snapshot_iters"]
    J = [h["J"] for h in hist]

    Ny, Nx = rho[0].shape
    y, x = np.mgrid[0:Ny, 0:Nx] / Nx
    Tmax = float(np.max(T))

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.86, bottom=0.12, wspace=0.28)

    def draw(fr, vmax=None):
        """vmax=None uses the run-wide maximum, which is right for the
        animation because it shows the temperature actually falling. For the
        still of the final design that washes everything out, so there we
        rescale to the frame itself."""
        for a in axes:
            a.clear()
        a0, a1, a2 = axes

        a0.imshow(rho[fr], origin="lower", cmap=CMAP_RHO, vmin=0, vmax=1,
                  extent=[0, 1, 0, 1], interpolation="nearest")
        a0.set_title("material layout   (dark = solid)")
        a0.set_xticks([]); a0.set_yticks([])

        im = a1.imshow(T[fr], origin="lower", cmap=CMAP_T, vmin=0,
                       vmax=Tmax if vmax is None else vmax,
                       extent=[0, 1, 0, 1], interpolation="bilinear")
        uc, vc = centers(U[fr], V[fr])
        speed = np.hypot(uc, vc)
        if speed.max() > 1e-12:
            a1.streamplot(x, y, uc, vc, color="#11182799", linewidth=0.7,
                          density=0.9, arrowsize=0.6)
        a1.set_title("temperature + coolant flow")
        a1.set_xlim(0, 1); a1.set_ylim(0, 1)
        a1.set_xticks([]); a1.set_yticks([])

        a2.plot(range(1, len(J) + 1), J, color=ACCENT, lw=1.8)
        k = min(int(iters[fr]), len(J)) - 1
        a2.plot([k + 1], [J[k]], "o", color=ACCENT, ms=6)
        a2.set_title("chip temperature")
        a2.set_xlabel("design iteration")
        a2.grid(True, color=GRID, lw=0.6)
        a2.set_xlim(0, len(J) + 1)

        fig.suptitle(
            f"Differentiable cold-plate topology optimisation   "
            f"iteration {int(iters[fr])}   J = {J[k]:.4f}",
            fontsize=12, fontweight="semibold", y=0.97,
        )
        return im

    last = len(rho) - 1
    im = draw(last, vmax=float(np.max(T[last])) * 1.02)
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    cb.set_label("T", rotation=0, labelpad=8)
    fig.savefig(out_png, bbox_inches="tight")
    cb.remove()

    writer = PillowWriter(fps=6)
    with writer.saving(fig, str(out_gif), dpi=110):
        for fr in range(len(rho)):
            draw(fr)
            writer.grab_frame()
        for _ in range(10):  # hold on the final design
            writer.grab_frame()
    plt.close(fig)
    print(f"wrote {out_gif} and {out_png}")


def fig2_gradient_validation(data, out_png):
    """data: dict with arrays 'fd', 'composed', 'frozen'."""
    fd = np.asarray(data["fd"])
    comp = np.asarray(data["composed"])
    frz = np.asarray(data["frozen"])

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.1))
    lim = 1.15 * max(np.abs(fd).max(), np.abs(frz).max())

    for a, g, c, name in ((ax[0], comp, ACCENT, "composed adjoint"),
                          (ax[1], frz, NAIVE, "frozen flow (naive)")):
        a.axhline(0, color=GRID, lw=0.8)
        a.axvline(0, color=GRID, lw=0.8)
        a.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.9, ls="--", label="exact")
        a.scatter(fd, g, s=46, color=c, zorder=3, label=name)
        a.set_xlabel("finite-difference gradient")
        a.set_ylabel("computed gradient")
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
        a.set_aspect("equal")
        a.legend(loc="upper left")
        err = np.max(np.abs(g - fd)) / max(np.max(np.abs(fd)), 1e-30)
        a.set_title(f"{name}\nmax rel err {err:.1e}")

    # shade the quadrants where the naive gradient has the wrong sign
    for xv, yv in zip(fd, frz):
        if xv * yv < 0:
            ax[1].scatter([xv], [yv], s=180, facecolors="none",
                          edgecolors="#dc2626", lw=1.6, zorder=4)
    ax[1].text(0.03, 0.03, "circled = wrong sign", transform=ax[1].transAxes,
               color="#dc2626", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def fig3_coupling(rows, out_png):
    """Naive-gradient error against coupling loop gain.

    Plotted against rho(Phi_T) rather than Rayleigh number on purpose: the
    error is *not* monotonic in Ra (Ra = 3e5 has a lower loop gain than
    Ra = 3e4, and correspondingly less error), but it is monotonic in the loop
    gain. The gain is the controlling variable, which is what makes the finding
    transferable to other coupled systems.
    """
    rows = sorted(rows, key=lambda r: r["rho_phi"])
    Ra = np.array([r["Ra"] for r in rows])
    gain = np.array([r["rho_phi"] for r in rows])
    err = np.array([r["rel_err"] for r in rows])  # plotted as measured, unclamped
    flip = np.array([100 * r.get("sign_flip_frac", 0.0) for r in rows])

    fig, ax = plt.subplots(figsize=(7.9, 5.0))
    ax.set_ylim(err.min() * 0.35, err.max() * 6.0)

    # Picard is only guaranteed to converge left of this line.
    ax.axvspan(1.0, max(1.35, gain.max() * 1.12), color="#fee2e2", alpha=0.6, zorder=0)
    ax.axvline(1.0, color="#b91c1c", lw=1.2, ls="--", zorder=1)
    ax.text(1.03, err.min() * 0.5, "  fixed point repelling:\n  Picard cannot converge",
            color="#b91c1c", fontsize=8.8, va="bottom")

    ax.semilogy(gain, err, "o-", color=NAIVE, lw=2.0, ms=7, zorder=3,
                label="naive gradient relative error")
    for g, e, r, fl in zip(gain, err, Ra, flip):
        ax.annotate(f"Ra={r:.0e}".replace("e+0", "e"), (g, e),
                    textcoords="offset points", xytext=(7, -11),
                    fontsize=8.2, color=MUTED)
        if fl > 0:
            # right-align the rightmost point's label so it stays in the axes
            right = g > 0.9 * gain.max()
            ax.annotate(f"{fl:.0f}% wrong sign", (g, e),
                        textcoords="offset points",
                        xytext=(-7 if right else 7, 9),
                        ha="right" if right else "left",
                        fontsize=8.4, color="#b91c1c", fontweight="bold")

    ax.set_xlabel(r"coupling loop gain  $\rho(\Phi_T)$")
    ax.set_ylabel("relative error of naive gradient")
    ax.grid(True, color=GRID, lw=0.6, which="both")
    ax.set_xlim(0, max(1.35, gain.max() * 1.12))
    ax.legend(loc="upper left")  # lower right is occupied by the repelling note
    ax.set_title("Component-wise differentiation fails as the coupling loop gain rises")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def fig4_opt_comparison(composed_hist, naive_hist, out_png):
    """Optimisation driven by each gradient.

    Reported as measured: at this operating point both reach essentially the
    same design, and the naive one ends a hair lower. That is a real result, not
    a disappointing one -- it says the failure of component-wise
    differentiation shows up in the gradient as a *quantity*, not necessarily in
    whether a normalised optimiser can still descend with it.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    finals = {}
    for path, c, name in ((composed_hist, ACCENT, "composed gradient (exact)"),
                          (naive_hist, NAIVE, "one-way gradient (naive)")):
        if not Path(path).exists():
            continue
        h = json.loads(Path(path).read_text())
        it = [r["iter"] for r in h]
        J = [r["J"] for r in h]
        ax.plot(it, J, color=c, lw=2.0, label=name)
        finals[name] = J[-1]
    ax.set_xlabel("design iteration")
    ax.set_ylabel("mean chip temperature  J")
    ax.set_yscale("log")
    ax.grid(True, color=GRID, lw=0.6, which="both")
    ax.legend()
    if len(finals) == 2:
        a, b = finals.values()
        ax.set_title("Both gradients optimise successfully at this operating point\n"
                     f"final J: {a:.4f} vs {b:.4f}", fontsize=10.5)
    else:
        ax.set_title("Topology optimisation convergence")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def fig6_trajectory(hist_path, out_png):
    """How wrong the naive gradient is along the optimisation trajectory.

    This is the resolution of an apparent paradox: the naive gradient carries
    40-150% error the whole way, yet the optimisation driven by it still
    succeeds. The reason is that after the first few iterations its *direction*
    is still roughly right (cosine 0.8-0.96), and a per-coordinate-normalised
    optimiser like Adam only consumes direction. It is a usable search
    direction and a useless sensitivity.
    """
    rows = [r for r in json.loads(Path(hist_path).read_text()) if "naive_rel_err" in r]
    if not rows:
        print(f"no diagnostics in {hist_path}")
        return
    it = np.array([r["iter"] for r in rows])
    err = np.array([r["naive_rel_err"] for r in rows])
    flip = np.array([100 * r["naive_sign_flip"] for r in rows])
    cos = np.array([r["naive_cos"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.plot(it, 100 * err, "o-", color=NAIVE, lw=2.0, ms=5,
            label="naive gradient relative error (%)")
    ax.plot(it, flip, "s--", color="#b91c1c", lw=1.8, ms=5,
            label="design variables with the wrong sign (%)")
    ax.set_xlabel("design iteration")
    ax.set_ylabel("percent")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_ylim(0, max(160, flip.max() * 1.15))

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    (lcos,) = ax2.plot(it, cos, color=MUTED, lw=1.4, ls=":",
                       label="cosine with the exact gradient (right axis)")
    ax2.set_ylabel("cosine with the exact gradient", color=MUTED)
    ax2.tick_params(axis="y", colors=MUTED)
    ax2.set_ylim(0, 1.05)

    ax.annotate(
        "uniform design:\n74% wrong sign",
        xy=(it[0], flip[0]), xytext=(11, 104),
        fontsize=8.8, color="#b91c1c", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.1),
    )
    ax.text(it[len(it) // 3], 17,
            "design solidifies → solid blocks the flow\n→ coupling weakens, direction recovers",
            fontsize=8.6, color=MUTED)

    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1 + [lcos], l1 + [lcos.get_label()], loc="upper center",
              bbox_to_anchor=(0.62, 1.0), fontsize=8.8)
    ax.set_title("The naive gradient stays wrong -- it just stays pointed downhill")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def fig5_architecture(out_png):
    """Diagram: three components, and the adjoint conversation between them."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    def box(x, y, w, h, title, sub, strategy, fc):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.12",
            linewidth=1.4, edgecolor=INK, facecolor=fc, zorder=2))
        ax.text(x + w / 2, y + h - 0.30, title, ha="center", fontsize=10.5,
                fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h - 0.62, sub, ha="center", fontsize=8.6,
                color=MUTED, zorder=3)
        ax.text(x + w / 2, y + 0.22, strategy, ha="center", fontsize=8.4,
                style="italic", color=ACCENT, zorder=3)

    box(0.30, 4.55, 2.85, 1.45, "material_map", "PyTorch", "torch.autograd", "#fdf6ec")
    box(3.75, 4.55, 2.85, 1.45, "thermal_advdiff", "JAX", "JAX autodiff", "#f0f7f1")
    box(3.75, 2.00, 2.85, 1.45, "stokes_brinkman", "C++ / Eigen",
        "hand-derived adjoint", "#eef4fb")

    def arrow(p0, p1, color, style="-", rad=0.0, lw=1.5, z=4):
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
            color=color, linestyle=style,
            connectionstyle=f"arc3,rad={rad}", zorder=z))

    # design -> properties -> both solvers
    arrow((3.15, 5.45), (3.75, 5.45), INK)
    ax.text(3.10, 5.62, "k", fontsize=8.8, color=INK, ha="center")
    arrow((1.72, 4.55), (3.75, 2.90), INK, rad=-0.18)
    ax.text(1.95, 3.55, "alpha", fontsize=8.8, color=INK)

    # the two-way coupling
    arrow((4.75, 3.45), (4.75, 4.55), NAIVE, lw=2.0)
    ax.text(4.28, 3.92, "u, v", fontsize=9.2, color=NAIVE, fontweight="bold")
    arrow((5.75, 4.55), (5.75, 3.45), ACCENT, lw=2.0)
    ax.text(5.88, 3.92, "T", fontsize=9.2, color=ACCENT, fontweight="bold")
    # below the solver rather than in the gap: anything placed between the
    # boxes masks the coupling arrowheads
    ax.text(5.18, 1.72, "two-way coupled fixed point", fontsize=8.6,
            color=MUTED, ha="center", va="top")

    # the krylov conversation
    ax.text(7.05, 5.62, "forward: Newton–Krylov", fontsize=9.6, fontweight="bold")
    arrow((7.05, 5.34), (9.55, 5.34), NAIVE, lw=1.7)
    ax.text(7.05, 4.72, "each GMRES matvec = one JVP,\nforward through C++ then JAX",
            fontsize=8.5, color=MUTED)

    ax.text(7.05, 3.92, "adjoint: GMRES", fontsize=9.6, fontweight="bold")
    arrow((9.55, 3.64), (7.05, 3.64), ACCENT, lw=1.7)
    ax.text(7.05, 3.02, "each matvec = one VJP,\nback through JAX then C++",
            fontsize=8.5, color=MUTED)

    ax.text(0.30, 1.32,
            "Three languages, three differentiation strategies, one differentiable function.",
            fontsize=9.8, color=INK, fontweight="bold", va="top")
    ax.text(0.30, 0.92,
            "The coupled adjoint exists only as a conversation between the solvers. It cannot\n"
            "be assembled component by component — and if you try, the resulting gradient\n"
            "points uphill.",
            fontsize=9.0, color=MUTED, va="top", linespacing=1.5)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--N", type=int, default=48)
    a = ap.parse_args()
    R = Path(a.results)
    R.mkdir(parents=True, exist_ok=True)

    npz = R / f"run_composed_N{a.N}.npz"
    hist = R / f"history_composed_N{a.N}.json"
    if npz.exists():
        fig1_animation(npz, hist, R / "fig1_optimisation.gif", R / "fig1_final.png")
    if (R / "gradient_validation.json").exists():
        fig2_gradient_validation(
            json.loads((R / "gradient_validation.json").read_text()),
            R / "fig2_gradient_validation.png",
        )
    if (R / "coupling_sweep.json").exists():
        fig3_coupling(
            json.loads((R / "coupling_sweep.json").read_text()),
            R / "fig3_coupling_strength.png",
        )
    fig4_opt_comparison(
        hist, R / f"history_one_way_N{a.N}.json", R / "fig4_opt_comparison.png"
    )
    fig5_architecture(R / "fig5_architecture.png")
    if (R / f"history_diag_N{a.N}.json").exists():
        fig6_trajectory(R / f"history_diag_N{a.N}.json", R / "fig6_trajectory_error.png")
