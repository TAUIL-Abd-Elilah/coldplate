# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Figures and animation for the cold-plate results.

  fig1  hero animation: material layout, temperature + streamlines, convergence
  fig2  gradient validation: composed vs finite differences vs frozen flow
  fig3  coupling strength: loop gain and naive-gradient error vs Rayleigh number
  fig4  optimisation driven by the composed gradient vs the naive one
  fig5  architecture: the four components and the adjoint between them
  fig6  naive-gradient error along the optimisation trajectory
  fig7  one design, rising coupling: where the two gradients disagree in space
  fig8  which statistic predicts that error -- directional gain, not loop gain
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


def fig8_predictor(json_path, out_png):
    """Which statistic predicts the error of a component-wise gradient?

    Left: the coupling loop gain rho(Phi_T), the obvious candidate. Right: the
    directional gain gamma = ||Phi_T^T g|| / ||g||, which is what the implicit
    function theorem actually puts in the leading error term. rho is a worst
    case over all directions; gamma asks about the one direction the objective
    cares about, and costs a single VJP.
    """
    rows = json.loads(Path(json_path).read_text())
    if len(rows) < 3:
        print(f"not enough rows in {json_path}")
        return

    err = np.array([r["rel_err"] for r in rows])
    sr = np.array([r["rho_phi"] for r in rows])
    gm = np.array([r["gamma"] for r in rows])
    names = [r["design"] for r in rows]
    uniq = sorted(set(names))
    palette = {n: c for n, c in zip(uniq, [ACCENT, NAIVE, "#7c3aed", "#059669"])}

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.7))
    floor = max(err.min() * 0.5, 1e-5)

    for a, x, lab in ((ax[0], sr, r"coupling loop gain  $\rho(\Phi_T)$"),
                      (ax[1], gm, r"directional gain  $\gamma=\|\Phi_T^{T}g\|/\|g\|$")):
        for n in uniq:
            m = [i for i, v in enumerate(names) if v == n]
            a.scatter(x[m], np.maximum(err[m], floor), s=62, label=n,
                      color=palette[n], edgecolors="white", linewidths=0.8, zorder=3)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel(lab)
        a.grid(True, color=GRID, lw=0.6, which="both")

    ax[0].set_ylabel("relative error of the naive gradient")
    lo = min(gm.min(), err.min()) * 0.5
    hi = max(gm.max(), err.max()) * 2
    ax[1].plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1.1, zorder=1,
               label=r"error $=\gamma$")
    ax[1].legend(loc="upper left", fontsize=9)
    ax[0].legend(loc="upper left", fontsize=9, title="design")

    ax[0].set_title("loop gain does not order the error", fontsize=11)
    ax[1].set_title(r"directional gain does, and error $\approx\gamma$ when small",
                    fontsize=11)

    # highlight the pair that the loop gain gets backwards
    try:
        i_r = next(i for i, r in enumerate(rows)
                   if r["design"] == "rough" and abs(r["Ra"] - 1e4) < 1)
        i_s = next(i for i, r in enumerate(rows)
                   if r["design"] == "smooth" and abs(r["Ra"] - 1e4) < 1)
        for i in (i_r, i_s):
            ax[0].scatter([sr[i]], [err[i]], s=210, facecolors="none",
                          edgecolors="#b91c1c", lw=1.8, zorder=4)
        ax[0].annotate("same Ra: lower loop gain,\nhigher error",
                       xy=(sr[i_s], err[i_s]), xytext=(0.06, 0.12),
                       textcoords="axes fraction", fontsize=8.8, color="#b91c1c",
                       arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.1))
    except StopIteration:
        pass

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def fig7_regime_maps(npz_path, out_png):
    """Same design, rising coupling: where the two gradients start to disagree.

    One row per Rayleigh number. The disagreement is not uniform noise -- it
    appears in the open fluid where a design change actually moves the flow,
    and is absent where nothing moves. That spatial structure is the coupling
    term, made visible.
    """
    d = np.load(npz_path)
    Ra, ge, gn = d["Ra"], d["g_exact"], d["g_naive"]
    rho, gain, flip = d["rho_phys"], d["loop_gain"], d["flip"]
    rel, cos = d["rel"], d["cos"]
    n = len(Ra)

    lim = float(np.percentile(np.abs(np.concatenate([ge.ravel(), gn.ravel()])), 99.0))
    cmap_g = LinearSegmentedColormap.from_list(
        "grad", ["#1e3a8a", "#60a5fa", "#f8fafc", "#fca5a5", "#991b1b"]
    )

    fig, axes = plt.subplots(n, 3, figsize=(9.6, 3.15 * n))
    axes = np.atleast_2d(axes)
    kw = dict(origin="lower", extent=[0, 1, 0, 1], interpolation="nearest")

    for r in range(n):
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])

        axes[r][0].imshow(ge[r], cmap=cmap_g, vmin=-lim, vmax=lim, **kw)
        axes[r][1].imshow(gn[r], cmap=cmap_g, vmin=-lim, vmax=lim, **kw)

        fl = np.sign(gn[r]) != np.sign(ge[r])
        axes[r][2].imshow(rho[r], cmap="Greys", vmin=0, vmax=3.2, **kw)
        ov = np.zeros((*fl.shape, 4))
        ov[fl] = [0.86, 0.15, 0.15, 0.92]
        axes[r][2].imshow(ov, **kw)

        if r == 0:
            axes[r][0].set_title("exact gradient\n(composed adjoint)",
                                 fontsize=10.5, color=ACCENT)
            axes[r][1].set_title("naive gradient\n(feedback loop cut)",
                                 fontsize=10.5, color=NAIVE)
            axes[r][2].set_title("sign disagreement", fontsize=10.5, color="#b91c1c")

        axes[r][0].set_ylabel(
            f"Ra = {Ra[r]:.0e}".replace("e+0", "e") + f"\nloop gain {gain[r]:.2f}",
            fontsize=10, labelpad=10,
        )
        axes[r][2].text(
            0.5, -0.07,
            f"{100*flip[r]:.0f}% wrong sign   ·   {100*rel[r]:.0f}% error   ·   cos {cos[r]:.3f}",
            transform=axes[r][2].transAxes, ha="center", va="top",
            fontsize=9.2, color="#b91c1c" if flip[r] > 0.02 else MUTED,
            fontweight="bold" if flip[r] > 0.02 else "normal",
        )

    fig.suptitle(
        "One design, rising coupling: component-wise differentiation degrades in place",
        fontsize=12.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def fig5_architecture(out_png):
    """Diagram: three components, and the adjoint conversation between them."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10.2, 7.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.6)
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

    box(0.30, 5.55, 2.85, 1.40, "material_map", "PyTorch", "torch.autograd", "#fdf6ec")
    box(3.75, 5.75, 2.85, 1.20, "thermal_advdiff", "JAX", "JAX autodiff", "#f0f7f1")
    box(3.75, 4.05, 2.85, 1.20, "thermal_fortran", "Fortran",
        "Enzyme, compiler AD", "#f3f0fa")
    box(3.75, 1.55, 2.85, 1.30, "stokes_brinkman", "C++ / Eigen",
        "hand-derived adjoint", "#eef4fb")

    def arrow(p0, p1, color, style="-", rad=0.0, lw=1.5, z=4, both=False):
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="<|-|>" if both else "-|>", mutation_scale=13,
            linewidth=lw, color=color, linestyle=style,
            connectionstyle=f"arc3,rad={rad}", zorder=z))

    # the two thermal implementations are drop-in replacements for each other
    arrow((6.75, 5.90), (6.75, 5.10), "#7c3aed", lw=1.6, both=True)
    ax.text(6.90, 5.50, "interchangeable\nto 5×10⁻¹²", fontsize=8.4,
            color="#7c3aed", va="center", fontweight="bold")

    # design -> properties -> solvers
    arrow((3.15, 6.35), (3.75, 6.35), INK)
    ax.text(3.05, 6.52, "k", fontsize=8.8, color=INK, ha="center")
    arrow((1.72, 5.55), (3.75, 2.45), INK, rad=-0.18)
    ax.text(1.90, 3.90, "alpha", fontsize=8.8, color=INK)

    # the two-way coupling, drawn to whichever thermal block is active
    arrow((4.60, 2.85), (4.60, 4.05), NAIVE, lw=2.0)
    ax.text(4.05, 3.40, "u, v", fontsize=9.2, color=NAIVE, fontweight="bold")
    arrow((5.75, 4.05), (5.75, 2.85), ACCENT, lw=2.0)
    ax.text(5.88, 3.40, "T", fontsize=9.2, color=ACCENT, fontweight="bold")
    ax.text(5.18, 1.30, "two-way coupled fixed point", fontsize=8.6,
            color=MUTED, ha="center", va="top")

    # the krylov conversation
    ax.text(7.05, 3.20, "forward: Newton–Krylov", fontsize=9.6, fontweight="bold")
    arrow((7.05, 2.92), (9.65, 2.92), NAIVE, lw=1.7)
    ax.text(7.05, 2.60, "each GMRES matvec = one JVP,\nforward through C++ then thermal",
            fontsize=8.5, color=MUTED, va="top")

    ax.text(7.05, 1.85, "adjoint: GMRES", fontsize=9.6, fontweight="bold")
    arrow((9.65, 1.57), (7.05, 1.57), ACCENT, lw=1.7)
    ax.text(7.05, 1.25, "each matvec = one VJP,\nback through thermal then C++",
            fontsize=8.5, color=MUTED, va="top")

    ax.text(0.30, 0.80,
            "Four languages, four differentiation strategies, one differentiable function.",
            fontsize=9.8, color=INK, fontweight="bold", va="top")
    ax.text(0.30, 0.44,
            "The coupled adjoint exists only as a conversation between the solvers — it cannot be\n"
            "assembled component by component. And the two thermal blocks are interchangeable:\n"
            "the end-to-end gradient does not depend on which derivative technology produced it.",
            fontsize=8.8, color=MUTED, va="top", linespacing=1.5)

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
    if (R / "gradient_maps.npz").exists():
        fig7_regime_maps(R / "gradient_maps.npz", R / "fig7_regime_maps.png")
    if (R / "predict_error.json").exists():
        fig8_predictor(R / "predict_error.json", R / "fig8_predictor.png")
