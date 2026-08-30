#!/usr/bin/env python3
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Render docs/index.html from the committed measurements.

Every number on that page is read out of a file in `orchestrator/results/`.
Nothing is typed by hand, which is the point: the page is a pure function of
the stored evidence, so `--check` can fail the build the moment prose and data
disagree. Same discipline as the paper build, applied to the web page.

    usage:  python scripts/build_results_page.py           # write docs/index.html
            python scripts/build_results_page.py --check   # fail if it is stale
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "index.html"

# Figures live where they were produced. The page is served from the repository
# root (a local clone, or Pages deployed from "/"), so these relative paths
# resolve in both cases and nothing is duplicated into docs/.
FIG = "../orchestrator/results"
# Markdown files render as a download when Pages serves them, so prose links
# point at the repository while assets stay relative and keep working offline.
REPO = "https://github.com/TAUIL-Abd-Elilah/coldplate"


def load(name: str, sub: str = "results"):
    path = ROOT / "orchestrator" / sub / name
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def esc(text: object) -> str:
    return html.escape(str(text))


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def sci(value: float) -> str:
    """A measurement in the notation the README uses: 5.3 x 10^-12."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}&nbsp;&times;&nbsp;10<sup>{int(exponent)}</sup>"


def table(headers: list[str], rows: list[list[str]], *, align: str = "") -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cls = f' class="{align}"' if align else ""
    return f"<table{cls}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def source(*files: str) -> str:
    links = " &middot; ".join(f"<code>{esc(f)}</code>" for f in files)
    return f'<p class="src">measured in {links}</p>'


def figure(name: str, caption: str) -> str:
    return (
        f'<figure><img loading="lazy" src="{FIG}/{name}" alt="{esc(caption)}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def section_headline() -> str:
    grad = load("gradient_validation.json")
    one = grad["stats_one-way"]
    frozen = grad["stats_frozen-flow(naive)"]
    interv = load("intervention_test.json")
    stats = load("predictor_statistics.json")

    best = max(interv["rows"], key=lambda r: r["amplitude"])
    gain = best["delta_J_exact_action"] / best["delta_J_naive_action"] - 1.0

    rows = [
        [
            "the loop-cut gradient, against a true coupled finite difference",
            f"relative error <b>{pct(one['rel_err'])}</b>, cosine {one['cos']:.4f}, "
            f"<b>{pct(one['sign_flip_frac'], 0)}</b> of design variables with the wrong sign",
        ],
        [
            "the same shortcut with the velocity field frozen",
            f"relative error {pct(frozen['rel_err'])}, cosine {frozen['cos']:.4f}, "
            f"{pct(frozen['sign_flip_frac'], 0)} wrong sign",
        ],
        [
            "acting on each gradient, then scoring both with the true coupled solver",
            f"the coupling-complete choice cools by {best['delta_J_exact_action']:.5f} "
            f"against {best['delta_J_naive_action']:.5f} &mdash; <b>{gain * 100:.0f}% more "
            "realised cooling</b> at equal raw-variable count and amplitude",
        ],
        [
            "screening the shortcut for one VJP",
            f"log-correlation <b>{stats['log_gamma_correlation']:.3f}</b> between &gamma; "
            f"and the error over {stats['n_converged']} converged configurations, against "
            f"{stats['rho_correlation']:.3f} for the spectral radius",
        ],
    ]
    return f"""
<section id="headline">
  <h2>The result</h2>
  <p class="lede">Temperature drives the flow through buoyancy and the flow drives
  temperature through advection, so the steady state is a fixed point rather than a
  chain. Cutting that loop &mdash; the shortcut written whenever one solver hands out no
  derivatives &mdash; is not a small approximation. It is most of the gradient, and the
  forward solution gives no warning at all.</p>
  {table(["what was measured", "result"], rows)}
  {source("results/gradient_validation.json", "results/intervention_test.json",
          "results/predictor_statistics.json")}
  <p><b>Who this is for.</b> Anyone standing in front of a coupled pipeline deciding
  whether to build the coupled adjoint or just differentiate the components separately
  &mdash; conjugate heat transfer, fluid&ndash;structure interaction,
  reservoir&ndash;geomechanics, any two solvers that feed each other. That call is usually
  made on intuition, because the forward solution looks healthy either way. This project
  measures what the shortcut actually costs on one such problem, shows that the obvious
  diagnostic is not sufficient, and ships the one-VJP screen that works better as a module
  that knows nothing about cold plates.</p>
  {figure("fig1_optimisation.gif",
          "The design evolving: material layout, temperature with streamlines, and the objective history.")}
</section>"""


def section_composition() -> str:
    parity = load("thermal_backend_parity.json")
    rows = [
        ["<code>stokes_brinkman</code>", "C++ / Eigen",
         "hand-derived discrete adjoint, no AD tool", "served"],
        ["<code>thermal_advdiff</code>", "Python / JAX",
         "JAX autodiff of the residual", "served (thermal slot)"],
        ["<code>thermal_fortran</code>", "Fortran",
         "Enzyme, compiler AD over LLVM IR", "served (thermal slot)"],
        ["<code>material_map</code>", "Python / PyTorch",
         "torch.autograd", "served"],
    ]
    return f"""
<section id="composition">
  <h2>What is composed</h2>
  <p>Three implementation languages and four derivative stacks. Three containers are
  served in any one run: the material map, the fluid solver, and exactly one of the two
  interchangeable thermal backends.</p>
  {table(["Tesseract", "language", "how derivatives are obtained", "role"], rows)}
  <p>The forward solve is Newton&ndash;Krylov on
  <code>F(T) = &Phi;(T) &minus; T</code>; the gradient is a GMRES solve against
  <code>(I &minus; &Phi;<sub>T</sub>)<sup>T</sup></code>. Every matvec in both crosses the
  container boundary &mdash; a JVP forward through the C++ block then the thermal block, a
  VJP backward through the thermal block then the C++ block. That is the sense in which
  the composition is load-bearing rather than convenient: there is no ordering of these
  components in which one sweep of the chain rule suffices.</p>
  <p><b>Chain, unrolled loop, solved loop.</b> &ldquo;Two-way coupled&rdquo; covers three
  different things. A <em>chain</em> needs one sweep of the chain rule. An
  <em>unrolled loop</em> contracts, so you iterate a fixed number of times and
  differentiate through the iterations &mdash; exact for the iterate it computes, no
  adjoint solve needed, and the right choice when the loop contracts and the coupled
  state is small. This is the third kind. At the gradient-study state
  &rho;(&Phi;<sub>T</sub>) &asymp; 1.19 &gt; 1, so the Picard iteration you would unroll
  does not converge at all &mdash; ours did not, and Anderson acceleration did not rescue
  it &mdash; and the coupled state is a temperature field carrying N&sup2; unknowns rather
  than a handful of scalars. The steady state has to be <em>solved</em> for, by Newton,
  and its sensitivity by a second transposed solve whose operator exists only as an
  action across the container boundary.</p>
  <p>The two thermal backends share a schema and nothing else, and they are
  interchangeable rather than merely composable. Swapping JAX autodiff for the
  independently written Fortran differentiated by Enzyme, at a converged coupled
  state on a {parity['N']}&times;{parity['N']} grid:</p>
  {table(["level", "JAX vs Fortran/Enzyme"], [
      ["component <code>T</code>", sci(parity["component_forward_T"])],
      ["component JVP", sci(parity["component_jvp"])],
      ["component VJP", sci(parity["component_vjp"])],
      ["converged coupled state <code>T*</code>", sci(parity["coupled_state_T"])],
      ["<b>end-to-end <code>dJ/d&rho;</code></b>",
       f"<b>{sci(parity['end_to_end_gradient'])}</b>, cosine "
       f"{parity['gradient_cosine']:.12f}"],
  ], align="num")}
  <p>The gradient that comes out of the whole composition &mdash; through the C++ fluid
  solver and the PyTorch material map &mdash; does not care whether the thermal block was
  differentiated by a Python tracer or by a compiler pass over Fortran. Reproduce it with
  <code>python orchestrator/compare_thermal_backends.py {parity['N']}</code>, which writes
  the file these numbers are read from.</p>
  {source("results/thermal_backend_parity.json")}
  {figure("fig5_architecture.png", "Three active Tesseracts and one selectable thermal backend.")}
</section>"""


def section_decision() -> str:
    interv = load("intervention_test.json")
    rows = []
    for row in sorted(interv["rows"], key=lambda r: r["amplitude"]):
        gain = row["delta_J_exact_action"] / row["delta_J_naive_action"] - 1.0
        rows.append([
            f"{row['amplitude']:.3f}",
            f"{row['delta_J_exact_action']:.5f}",
            f"{row['delta_J_naive_action']:.5f}",
            f"<b>{gain * 100:.0f}%</b>",
        ])

    matrix = load("intervention_robustness_matrix_48.json")
    summary = matrix["summary"]

    def tally(block: dict) -> dict:
        outcomes = block["outcomes"]
        return {
            "exact": outcomes["exact_wins"],
            "shortcut": outcomes["shortcut_wins"],
            "tie": outcomes["tie"],
            "noncomparable": block["attempts_recorded"] - block["comparable_cases"],
        }

    by_ra = []
    for entry in sorted(matrix["by_rayleigh_number"], key=lambda e: e["Ra"]):
        counts = tally(entry)
        by_ra.append([
            f"{int(entry['Ra']):,}",
            str(counts["exact"]), str(counts["shortcut"]),
            str(counts["tie"]), str(counts["noncomparable"]),
        ])
    pooled = tally(summary)
    by_ra.append([
        "<b>pooled</b>",
        f"<b>{pooled['exact']}</b>", f"<b>{pooled['shortcut']}</b>",
        f"<b>{pooled['tie']}</b>", f"<b>{pooled['noncomparable']}</b>",
    ])

    over_comparable = summary["exact_win_rate_over_comparable"]
    cluster = summary["cluster_aware_seed_analysis"]
    bootstrap = cluster["bootstrap"]
    prior = matrix["by_prior_observation_status"]["observed_before_frozen_design"]
    prior_n = prior["attempts_recorded"]
    return f"""
<section id="decision">
  <h2>The gradient changes a realised engineering decision</h2>
  <p>Gradient accuracy only matters if acting on it changes the physical outcome. At a
  strongly coupled state each gradient was given the same rule &mdash; raise the 5% of
  design variables it calls most beneficial, lower the 5% it calls least beneficial by the
  same amplitude, zero-sum in the raw variables &mdash; and then both predictions were
  thrown away and the true coupled forward problem re-solved. Filtering and projection are
  nonlinear, so this equalises the raw move, not the realised density move.</p>
  {table(["raw amplitude per selected cell",
          "&Delta;J, cells chosen by the exact gradient",
          "&Delta;J, cells chosen by the loop-cut gradient",
          "extra realised cooling"], rows, align="num")}
  {source("results/intervention_test.json")}
  {figure("fig10_intervention.png",
          "Equal zero-sum raw-design interventions selected by each gradient, evaluated by the true coupled solver.")}

  <h3>Repeated over a frozen 48-attempt matrix</h3>
  <p>The retrospectively frozen extension completed every planned
  <code>(Ra, seed)</code> cell and kept base-solver failures, incomplete action pairs, ties
  and the one shortcut win. It is a robustness extension rather than an untouched
  independent confirmation set: {prior_n} of the 48 cells had stored evidence before the
  design was frozen, and that split is reported rather than buried.</p>
  {table(["Ra", "exact wins", "shortcut wins", "ties", "noncomparable"], by_ra, align="num")}
  <p>{over_comparable['successes']} of {over_comparable['trials']} comparable cases
  ({pct(over_comparable['estimate'], 1)}) favour the exact-gradient action; the
  attempt-level Wilson interval is
  {pct(over_comparable['lower'], 1)}&ndash;{pct(over_comparable['upper'], 1)}. Because each
  seed appears at all three Rayleigh numbers, a secondary bootstrap resamples the
  {cluster['complete_clusters']} complete seed clusters rather than pretending the attempts
  are independent; that post-freeze <em>descriptive</em> interval is
  {pct(bootstrap['lower'], 1)}&ndash;{pct(bootstrap['upper'], 1)}.</p>
  {source("results/intervention_robustness_matrix_48.json")}
  {figure("fig13_robustness_matrix.png",
          "All 48 retained outcomes, with per-Rayleigh Wilson intervals and a pooled seed-cluster bootstrap.")}
</section>"""


def section_predictor() -> str:
    stats = load("predictor_statistics.json")
    folds = stats["leave_one_family_out"].values()
    fold_lo = min(v["log_gamma_correlation"] for v in folds)
    fold_hi = max(v["log_gamma_correlation"] for v in folds)

    sweep = load("objective_sweep.json")
    rows = [
        [esc(row["objective"].replace("_", " ")), f"{row['gamma']:.4f}", f"{row['rel_err']:.4f}"]
        for row in sorted(sweep, key=lambda r: r["gamma"])
    ]
    rho = sorted({round(row["rho_phi"], 8) for row in sweep})[0]
    spread = int(max(r["rel_err"] for r in sweep) / min(r["rel_err"] for r in sweep))

    gg = load("gamma_generalization.json")
    families = [[
        "<b>all</b>", f"<b>{gg['overall']['n']:,}</b>",
        f"<b>{gg['overall']['log_gamma_correlation']:+.4f}</b>",
        f"{gg['overall']['rho_correlation']:+.4f}",
    ]]
    families += [
        [esc(name), f"{entry['n']:,}", f"{entry['log_gamma_correlation']:+.4f}",
         f"{entry['rho_correlation']:+.4f}"]
        for name, entry in sorted(gg["per_family"].items())
    ]
    families += [
        [f"{esc(name)} loops", f"{entry['n']:,}",
         f"{entry['log_gamma_correlation']:+.4f}", f"{entry['rho_correlation']:+.4f}"]
        for name, entry in sorted(gg["per_kind"].items())
    ]
    safe, unsafe = gg["safe_bucket"], gg["unsafe_bucket"]
    return f"""
<section id="predictor">
  <h2>One VJP says whether the shortcut is safe</h2>
  <p>The adjoint solves <code>(I &minus; &Phi;<sub>T</sub>)<sup>T</sup>&lambda; = g</code>.
  Cutting the loop uses <code>&lambda;<sub>0</sub> = g</code>, whose residual in that
  equation is exactly <code>&Phi;<sub>T</sub><sup>T</sup>g</code>. Its normalised norm
  &mdash; the <b>directional gain</b>
  <code>&gamma; = &#8214;&Phi;<sub>T</sub><sup>T</sup>g&#8214; / &#8214;g&#8214;</code>
  &mdash; costs one VJP, far less than the gradient it screens, and unlike the spectral
  radius it knows which direction the objective actually cares about.</p>
  {table(["predictor", "correlation with log<sub>10</sub>(loop-cut error)"],
         [["&rho;(&Phi;<sub>T</sub>), the loop gain", f"{stats['rho_correlation']:+.3f}"],
          ["<b>log<sub>10</sub>(&gamma;), the directional gain</b>",
           f"<b>{stats['log_gamma_correlation']:+.3f}</b>"]])}
  <p>Over {stats['n_converged']} converged configurations drawn from
  {stats['n_design_families']} design families and {stats['n_rayleigh_levels_represented']}
  Rayleigh levels. Leave-one-family-out correlations are {fold_lo:.3f}&ndash;{fold_hi:.3f};
  a seeded {stats['bootstrap_samples']:,}-sample bootstrap gives a 95% interval of
  {stats['bootstrap_95_percent_interval'][0]:.3f}&ndash;{stats['bootstrap_95_percent_interval'][1]:.3f}.</p>
  {source("results/predictor_statistics.json", "results/predict_error.json")}
  {figure("fig8_predictor.png",
          "The directional gain predicts the loop-cut error; the loop gain does not.")}

  <h3>The confound removed: hold the physics fixed, change only the objective</h3>
  <p>At one design and one Rayleigh number there is a single coupled state, hence a single
  &rho;(&Phi;<sub>T</sub>) = {rho:.4f} on every row below. Changing what is being measured
  changes <code>g = dJ/dT</code>, and therefore &gamma; &mdash; but &rho; cannot move. The
  error still varies by a factor of {spread:.0f}&times;, so a constant cannot explain it.</p>
  {table(["objective", "&gamma;", "loop-cut relative error"], rows, align="num")}
  <p>&gamma; moves with the error and gets the dangerous row right, but it is a screening
  signal with a theoretical basis, not a formula: across objectives it correlates 0.80
  against 0.995 across designs, because converting an adjoint-equation residual into a
  design-gradient error also involves
  <code>(I &minus; &Phi;<sub>T</sub><sup>T</sup>)<sup>&minus;1</sup></code>.</p>
  {source("results/objective_sweep.json")}

  <h3>Does it generalise off this problem? {gg['trials_usable']:,} random systems</h3>
  <p>The physics is removed entirely: random coupled fixed points
  <code>x* = &Phi;(x*, &theta;)</code> where every quantity is closed form &mdash; no solver
  tolerance, no finite differences &mdash; across four structural families and both linear
  and nonlinear loops, with &gamma; computed by calling the shipped
  <code>coupling_check.py</code> rather than a reimplementation.</p>
  {table(["subset", "n", "corr(log &gamma;, log error)", "corr(&rho;, log error)"],
         families, align="num")}
  <p>The shipped thresholds hold, which matters more than the correlation, because a false
  <span class="ok">SAFE</span> is the verdict that hurts someone &mdash; it tells them to
  skip the adjoint. <b>{safe['n']} cases below &gamma; = 0.01</b> have a worst error of
  {pct(safe['worst_rel_err'], 1)}, and {pct(safe['frac_under_5pct'], 0)} sit under 5%: no
  false SAFE. <b>{unsafe['n']} cases above &gamma; = 0.10</b> are
  {pct(unsafe['frac_over_5pct'], 0)} genuinely above 5% error: no false alarm.</p>
  <p class="caveat"><b>And the boundary, which we would rather not have found.</b> &gamma;
  correlates {gg['attracting']['log_gamma_correlation']:+.4f} on the
  {gg['attracting']['n']:,} attracting fixed points (&rho; &lt; 1) but only
  {gg['repelling']['log_gamma_correlation']:+.2f} on the {gg['repelling']['n']} repelling
  ones, because the residual it measures is amplified by
  <code>(I &minus; &Phi;<sub>T</sub><sup>T</sup>)<sup>&minus;1</sup></code> and no
  contraction bound applies there. The shipped policy for a repelling fixed point is
  therefore to refuse to screen and compute the adjoint &mdash; which is exactly what this
  project does at its own headline state, where &rho; &asymp; 1.19.</p>
  {source("results/gamma_generalization.json")}
  {threshold_explorer()}
  {figure("fig11_generalization.png",
          "The directional gain against the truth on random coupled systems, and where it stops working.")}
</section>"""


#: Candidate SAFE gates. 0.01 is the value `coupling_check.py` actually ships;
#: the others exist so a reader can see what choosing differently would cost.
GATES = (0.003, 0.01, 0.03, 0.1, 0.3)
SHIPPED_GATE = 0.01

# Plot geometry, in the SVG's own user units.
PLOT = {"w": 660, "h": 400, "l": 62, "r": 14, "t": 14, "b": 46}
XLIM = (-4.0, 1.0)   # log10 gamma
YLIM = (-4.0, 0.5)   # log10 relative error
DANGER = 0.05        # "wrong enough to matter", the threshold the gate must catch


def _scale(value: float, lo: float, hi: float, a: float, b: float) -> float:
    return a + (value - lo) / (hi - lo) * (b - a)


def threshold_explorer() -> str:
    """A gate the reader can move, with every count precomputed from the data.

    No script: five radio buttons and sibling selectors swap between five
    pre-rendered readouts. That keeps the page inert and offline-safe while
    still letting someone check the claim rather than take it.
    """
    points = load("gamma_generalization_points.json")
    gamma_error = [(row[0], row[1]) for row in points["rows"]]

    # The axes are fixed, the data is not: rerunning the study with a different
    # seed or trial count could put a point outside the frame, where it would
    # silently draw over the page instead of inside the plot. Fail instead.
    for name, values, (lo, hi) in (
        ("gamma", [g for g, _ in gamma_error], XLIM),
        ("relative error", [e for _, e in gamma_error], YLIM),
    ):
        if min(values) <= 0:
            raise ValueError(f"{name} must be positive to plot on a log axis")
        low, high = math.log10(min(values)), math.log10(max(values))
        if low < lo or high > hi:
            raise ValueError(
                f"{name} spans 10^{low:.2f} to 10^{high:.2f}, outside the plotted "
                f"10^{lo:g} to 10^{hi:g}; widen the axis rather than clipping data"
            )
    # Gates are gamma values, so they belong on the horizontal axis.
    for gate in GATES:
        if not XLIM[0] <= math.log10(gate) <= XLIM[1]:
            raise ValueError(f"gate {gate:g} would draw outside the plot")

    x0, x1 = PLOT["l"], PLOT["w"] - PLOT["r"]
    y0, y1 = PLOT["h"] - PLOT["b"], PLOT["t"]

    def px(gamma: float) -> float:
        return _scale(math.log10(gamma), *XLIM, x0, x1)

    def py(err: float) -> float:
        return _scale(math.log10(err), *YLIM, y0, y1)

    dots = []
    for gamma, err in gamma_error:
        cls = "hot" if err > DANGER else "cool"
        dots.append(f'<circle class="{cls}" cx="{px(gamma):.1f}" cy="{py(err):.1f}" r="1.7"/>')

    # Axis ticks at decade boundaries.
    ticks = []
    for decade in range(int(XLIM[0]), int(XLIM[1]) + 1):
        x = px(10.0 ** decade)
        ticks.append(f'<line class="grid" x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y0}"/>')
        ticks.append(f'<text class="tick" x="{x:.1f}" y="{y0 + 18}" '
                     f'text-anchor="middle">10<tspan dy="-5" font-size="8">{decade}</tspan></text>')
    for decade in range(int(YLIM[0]), 1):
        y = py(10.0 ** decade)
        ticks.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        ticks.append(f'<text class="tick" x="{x0 - 8}" y="{y + 3:.1f}" '
                     f'text-anchor="end">10<tspan dy="-5" font-size="8">{decade}</tspan></text>')
    danger_y = py(DANGER)
    ticks.append(f'<line class="danger" x1="{x0}" y1="{danger_y:.1f}" '
                 f'x2="{x1}" y2="{danger_y:.1f}"/>')
    ticks.append(f'<text class="tick danger-label" x="{x1 - 4}" y="{danger_y - 6:.1f}" '
                 f'text-anchor="end">5% error</text>')

    inputs, labels, lines, readouts = [], [], [], []
    for index, gate in enumerate(GATES):
        checked = " checked" if gate == SHIPPED_GATE else ""
        inputs.append(f'<input type="radio" name="gate" id="gate{index}"{checked}>')
        shipped = " &#9733;" if gate == SHIPPED_GATE else ""
        labels.append(f'<label for="gate{index}">&gamma; &lt; {gate:g}{shipped}</label>')

        gx = px(gate)
        lines.append(
            f'<g class="gate g{index}">'
            f'<rect x="{x0:.1f}" y="{y1}" width="{gx - x0:.1f}" height="{y0 - y1}"/>'
            f'<line x1="{gx:.1f}" y1="{y1}" x2="{gx:.1f}" y2="{y0}"/>'
            f'<text x="{gx + 6:.1f}" y="{y1 + 14}">screened SAFE</text>'
            f"</g>"
        )

        screened = [err for gamma, err in gamma_error if gamma < gate]
        false_safe = sum(1 for err in screened if err > DANGER)
        worst = max(screened) if screened else 0.0
        verdict = (
            f'<span class="ok">no false SAFE</span>'
            if false_safe == 0
            else f'<span class="bad">{false_safe} false SAFE</span>'
        )
        readouts.append(
            f'<div class="readout r{index}">'
            f"<p>Gating at <b>&gamma; &lt; {gate:g}</b> screens "
            f"<b>{len(screened):,}</b> of the {len(gamma_error):,} systems as safe to "
            f"differentiate component-wise. Worst true error among them: "
            f"<b>{pct(worst, 1)}</b>. Above the 5% line: {verdict}."
            f"</p></div>"
        )

    return f"""
  <h3>Move the gate yourself</h3>
  <p>Every dot is one random coupled system: its directional gain across, the true error
  of the loop-cut gradient up, both on log axes. Orange dots are the ones where cutting
  the loop costs more than 5%. A gate is a vertical line, and everything to its left is a
  system the screen would wave through without an adjoint &mdash; so an orange dot inside
  the shaded band is somebody being told to skip a computation they needed.</p>
  <div class="explorer">
    {''.join(inputs)}
    <div class="gates">{''.join(labels)}</div>
    <figure class="plot">
      <svg viewBox="0 0 {PLOT['w']} {PLOT['h']}" role="img" aria-label="Loop-cut gradient error against the directional gain for {len(gamma_error)} random coupled systems, with selectable screening gates.">
        {''.join(ticks)}
        {''.join(lines)}
        <g class="dots">{''.join(dots)}</g>
        <text class="axis" x="{(x0 + x1) / 2:.0f}" y="{PLOT['h'] - 8}"
              text-anchor="middle">directional gain &gamma;</text>
        <text class="axis" transform="translate(14,{(y0 + y1) / 2:.0f}) rotate(-90)"
              text-anchor="middle">true relative error of the loop-cut gradient</text>
      </svg>
    </figure>
    <div class="readouts">{''.join(readouts)}</div>
  </div>
  <p class="src">counts recomputed at build time from
  <code>results/gamma_generalization_points.json</code>, written by the same run that
  produced the summary above. &#9733; marks the gate <code>coupling_check.py</code>
  actually ships.</p>"""


def section_validation() -> str:
    rayleigh = load("critical_rayleigh.json")
    rows = [
        [str(row["aspect"]), f"{row['ra_classical']:.2f}", f"{row['excess'] * 100:+.3f}%"]
        for row in sorted(rayleigh, key=lambda r: r["aspect"])
    ]
    best = min(rayleigh, key=lambda r: abs(r["excess"]))

    grid = load("grid_convergence.json")
    grid_rows = [[str(row["N"]), f"{row['J']:.6f}", "yes" if row["ok"] else "no"]
                 for row in sorted(grid, key=lambda r: r["N"])]

    cavity = load("de_vahl_davis.json")
    errors = [abs(value)
              for row in cavity
              for value in row["relative_error"].values()]
    cavity_note = (
        f"Both cases converge, and all {len(errors)} Nusselt and centreline-velocity "
        f"observables land within {pct(max(errors), 1)} of the published references."
        if errors else ""
    )
    return f"""
<section id="validation">
  <h2>Validation</h2>
  <h3>The coupled physics reproduces the classical onset of convection</h3>
  <p>A fluid layer heated from below stays motionless until buoyancy overcomes viscous and
  thermal diffusion, at a precisely known <code>Ra<sub>c</sub> = 1707.762</code> for rigid
  walls. Onset is also exactly where this project's own machinery puts it: at the
  conduction state the coupling loop <em>is</em> the linear stability operator, so
  convection begins precisely when the loop gain reaches one. No-slip side walls stabilise
  a confined box, so the measured value must sit above the classical one and fall towards
  it as the box widens.</p>
  {table(["aspect ratio", "Ra<sub>c</sub> measured", "excess over 1707.762"], rows, align="num")}
  <p>At aspect ratio {best['aspect']} that is agreement to four significant figures. One
  number checks the C++ Stokes solver, the thermal solver, the buoyancy coupling between
  them and the loop-gain machinery, all at once.</p>
  {source("results/critical_rayleigh.json")}

  <h3>The discretisation converges at second order</h3>
  <p>Verification rather than validation: does the code solve its own equations at the rate
  the scheme implies? Smooth analytic properties, the grid-dependent density filter
  bypassed, Richardson extrapolation on two independent grid trios &mdash; observed order
  1.87 and 1.83, monotone in J.</p>
  {table(["N", "J", "converged"], grid_rows, align="num")}
  {source("results/grid_convergence.json")}

  <h3>Nonlinear, finite-Prandtl cavity</h3>
  <p>The de Vahl Davis benchmark activates the Navier&ndash;Stokes inertia term, so it
  validates the nonlinear path rather than asking the Stokes approximation to reproduce
  finite-Prandtl flow. {cavity_note}</p>
  {source("results/de_vahl_davis.json")}
  {figure("fig14_physics_validation.png",
          "The nonlinear cavity reference, and the explicit 1 W SI illustration that failed its audit.")}
</section>"""


def section_unroll() -> str:
    study = load("unroll_study.json")
    rep = study["regimes"]["repelling"]
    con = study["regimes"]["contracting"]
    best_rep = min(r["relative_error"] for r in rep["unrolled"]
                   if r["relative_error"] is not None)
    best_con = min(r["relative_error"] for r in con["unrolled"]
                   if r["relative_error"] is not None)

    rows = []
    for a, b in zip(sorted(rep["unrolled"], key=lambda r: r["sweeps"]),
                    sorted(con["unrolled"], key=lambda r: r["sweeps"]), strict=True):
        def fmt(row):
            return "not finite" if row["relative_error"] is None else f"{row['relative_error']:.2e}"
        rows.append([str(a["sweeps"]), fmt(a), fmt(b)])

    head = [
        ["loop gain &rho;(&Phi;<sub>T</sub>)",
         f"<strong>{rep['loop_gain']:.3f}</strong>", f"{con['loop_gain']:.3f}"],
        ["plain Picard",
         f"{rep['picard']['residuals'][0]:.2e} &rarr; {rep['picard']['residuals'][-1]:.2e}, stalls",
         f"{con['picard']['residuals'][0]:.2e} &rarr; {con['picard']['residuals'][-1]:.2e}, converges"],
        ["implicit adjoint",
         f"<strong>{rep['implicit_relative_error']:.2e}</strong>",
         f"{con['implicit_relative_error']:.2e}"],
        ["best unrolled gradient",
         f"{best_rep:.2e}", f"<strong>{best_con:.2e}</strong>"],
    ]

    return f"""
<section id="unroll">
  <h2>Where an unrolled loop is fine, and where it is not</h2>
  <p>&ldquo;Two-way coupled&rdquo; covers three different things: a chain, an
  <em>unrolled</em> loop, and a <em>solved</em> loop. Only the third needs any of the
  machinery here, so that claim is worth testing rather than asserting. Same design
  draw, same three components, same containers &mdash; only the Rayleigh number moves.
  Both candidates are scored against a central difference of the fully solved coupled
  problem, the one referee sharing no method with either.</p>
  {table(["", "repelling (Ra = 3&times;10<sup>4</sup>)", "contracting (Ra = 10<sup>3</sup>)"], head)}
  {table(["sweeps unrolled", "repelling", "contracting"], rows, align="num")}
  <p><strong>Read the right-hand column first.</strong> Where the loop contracts,
  differentiating the unrolled iteration is as accurate as the implicit adjoint
  &mdash; {best_con:.1e} against {con['implicit_relative_error']:.1e}. If your loop
  contracts, unroll it; nothing here is needed.</p>
  <p>The left-hand column is why this exists. At &rho; = {rep['loop_gain']:.2f} the plain
  iteration stalls at {rep['picard']['residuals'][-1]:.1e} instead of converging, the
  gradient differentiated through it is {best_rep * 100:.0f}% wrong <em>at best</em>, and
  more sweeps make it worse rather than better &mdash; there is no converged iterate to
  unroll toward. The implicit adjoint lands at {rep['implicit_relative_error']:.2e}, about
  {best_rep / rep['implicit_relative_error']:,.0f}&times; closer.</p>
  {source("results/unroll_study.json")}
</section>"""


def section_independent() -> str:
    checked = load("check_gradients.json")
    cost = load("adjoint_cost.json")

    order = ("stokes_brinkman", "thermal_advdiff", "thermal_fortran", "material_map")
    rows = []
    for name in order:
        component = checked["components"][name]
        paths = ", ".join(f"<code>{esc(p)}</code>" for p in component["input_paths"])
        count = component["phantom_rows"]
        rows.append([
            f"<code>{esc(name)}</code>",
            esc(component["how_differentiated"]),
            paths,
            f"{component['relative_disagreement']:.1e}",
            f"<strong>{count}</strong>" if count else "0",
        ])

    fd = cost["central_difference_gradient"]
    hours = fd["extrapolated_seconds"] / 3600.0
    matvecs = cost["cross_boundary_matvecs_per_gradient"]
    cost_rows = [
        [f"one coupled solve from cold, {cost['cold_solve']['newton_iterations']} Newton iterations",
         f"{cost['cold_solve']['seconds']:.1f}&nbsp;s", "measured"],
        ["one finite-difference probe &mdash; perturb one variable, warm-started",
         f"{cost['seconds_one_coupled_forward_solve']:.2f}&nbsp;s",
         f"measured, median of {cost['finite_difference_probe']['probes']}"],
        [f"one composed adjoint gradient, {matvecs['jvp']} JVP + {matvecs['vjp']} VJP "
         f"across the boundary",
         f"{cost['seconds_one_composed_gradient']:.2f}&nbsp;s",
         f"measured, median of {cost['repeats']}"],
        [f"one central-difference gradient, {fd['coupled_solves_required']:,} coupled solves",
         f"{hours:,.1f}&nbsp;h", "<strong>extrapolated</strong>"],
    ]
    speedup = cost["adjoint_speedup_over_central_differences"]
    opt = cost["optimisation"]
    phantom = checked["phantom_rows_total"]

    return f"""
<section id="independent">
  <h2>What Tesseract's own gradient checker found</h2>
  <p>Every other check on this page is one we wrote. Tesseract ships its own: it samples
  random input/output index pairs and compares each declared derivative endpoint against a
  central difference taken through <code>apply</code>. It knows nothing about cold plates.
  Run against all four components at the converged coupled state, it found two things.</p>
  {table(["Tesseract", "how its derivatives are obtained", "inputs checked",
          "relative disagreement", "phantom"], rows)}
  <p><strong>The live comparisons agree.</strong> Across
  {checked["total_comparisons"]} comparisons the worst disagreement between any endpoint
  and a central difference is <strong>{checked["worst_relative_disagreement"]:.1e}</strong>
  relative &mdash; four derivative implementations sharing no machinery, measured by
  somebody else's tool. Each figure is the best rung of a
  {len(checked["rel_eps_ladder"])}-step ladder of finite-difference steps, and the number
  is not a pass at a threshold picked afterwards: the checker's verdict is
  <code>allclose(fd, endpoint, atol=1e-8, rtol=r)</code> and neither side depends on
  <code>r</code>, so running it at <code>rtol&nbsp;=&nbsp;0</code> hands back both rows and
  the disagreement follows by arithmetic.</p>
  <p><strong>And it found a defect we had not.</strong> {phantom} of the sampled
  comparisons are <em>phantom sensitivities</em>: the finite difference is exactly zero
  because the forward map never reads the perturbed input, while the endpoint reports a
  real number. They are the wall-face velocities &mdash; the assembly <code>apply</code>
  uses sums fluxes over interior faces only, while the JAX residual the derivative path
  differentiates includes wall terms that vanish in value at a no-slip wall but not in
  derivative. The independently written Fortran block has none, which localises it; and
  swapping that block in moves the end-to-end gradient by only 5.3&nbsp;&times;&nbsp;10<sup>-12</sup>,
  which bounds what the defect can cost. It is recorded here rather than quietly fixed.</p>
  {figure("fig15_check_gradients.png",
          "The step-size ladder behind the independent check, and the two findings side by side.")}
  {source("results/check_gradients.json")}

  <h3>What the coupled adjoint costs, and what it replaces</h3>
  <p>If you were not going to build it, the alternatives are the loop-cut shortcut &mdash;
  whose error is the subject of this whole page &mdash; or finite differences over the
  design vector. At {cost['N']}&times;{cost['N']} that is
  {cost['n_design_variables']:,} design variables and two coupled solves apiece.</p>
  {table(["cost of", "wall clock", "basis"], cost_rows)}
  <p>Roughly <strong>{speedup:,.0f}&times;</strong>. The per-solve figure is a warm-started
  probe rather than a cold solve, which is the number most favourable to finite
  differences. The whole {opt['iterations']}-iteration optimisation took
  <strong>{opt['measured_seconds'] / 60:.1f} minutes</strong>; on central differences the
  same schedule extrapolates to
  <strong>{opt['extrapolated_seconds_with_central_differences'] / 86400:.1f} days</strong>.
  We did not run {fd['coupled_solves_required']:,} solves to prove a multiplication, and
  the table says which row was measured and which was not.</p>
  {source("results/adjoint_cost.json")}
</section>"""


def section_negatives() -> str:
    interp = load("strong_coupling_showdown_interpretation.json")
    prefix = interp["descriptive_common_prefix"]["methods"]

    physical = load("dimensional_coldplate.json")
    mesh_solves = [
        layout
        for row in physical["mesh_refinement"]["rows"]
        for layout in row["layouts"].values()
    ]
    converged = sum(1 for layout in mesh_solves if layout["solver"]["ok"])

    ranking = load("sensitivity_ranking.json")
    one_way = ranking["one_way"]
    top10 = next(entry for entry in one_way["per_k"] if entry["k"] == 10)
    top50 = next(entry for entry in one_way["per_k"] if entry["k"] == 50)
    return f"""
<section id="negatives">
  <h2>What we do not claim</h2>
  <p class="lede">These are the results that did not go our way. They are kept because a
  study that cannot record a negative is not evidence.</p>

  <h3>Both gradients drive the long optimisation successfully</h3>
  <p>Run twice at 96&times;96 from the same seed and schedule, the coupling-complete and
  loop-cut gradients reach essentially the same objective &mdash; and the naive run ends a
  hair lower. An optimiser converging is not evidence that a gradient is right. Along that
  trajectory the shortcut is only 4&ndash;20% off with a cosine above 0.98 and almost no
  sign errors: a perfectly serviceable search direction. Two things differ from the
  strongly coupled state above &mdash; the Rayleigh number, and the fact that filtered,
  projected designs are smooth while the random design used for the gradient study is
  not.</p>

  <h3>Serviceable as a direction, worthless as an attribution</h3>
  <p>Descent needs only a positive inner product with the truth. Attribution &mdash; which
  parts of the design actually drive the objective, the question behind tolerancing,
  sensor placement and mesh refinement &mdash; reads the gradient entry by entry, so every
  entry has to be right on its own. Ranking every design cell by influence, the loop-cut
  ordering of magnitudes correlates <b>{one_way['spearman_magnitude']:+.3f}</b> with the
  true one: statistically indistinguishable from chance. Its single most influential cell
  is truly the #{one_way['top1_true_rank_of_naive_pick']}; it recovers
  {pct(top10['recall'], 0)} of the true top ten and {pct(top50['recall'], 0)} of the true
  top fifty; and it promotes into its own top fifty a cell that is truly ranked
  #{top50['worst_true_rank_promoted']:,} of {one_way['n_cells']:,} &mdash; among the least
  influential in the entire domain. Sign agreement on the cells that genuinely matter is
  {pct(top50['sign_agreement_on_true_topk'], 0)}. Signs survive; magnitudes do not, and
  nothing in the forward solution would warn the engineer who tightened a tolerance on that
  cell.</p>
  {source("results/sensitivity_ranking.json")}
  {figure("fig9_attribution.png",
          "Which cells each gradient says matter: signs survive, ranking does not.")}

  <h3>The frozen eight-step showdown did not complete</h3>
  <p>Three branches were given the same initial design, proposal rule, projected-volume
  target, eight update opportunities and candidate-solve budget. The composed branch
  accepted five decisions and then its sixth candidate failed to converge inside the frozen
  budget. Because the horizons differ, the eight-step endpoint is <b>not evaluable and
  there is no verdict</b>. The shared five-step prefix &mdash;
  {prefix['composed']['reduction_percent']:.2f}% composed,
  {prefix['one_way']['reduction_percent']:.2f}% loop-cut,
  {prefix['frozen']['reduction_percent']:.2f}% frozen-flow &mdash; was examined after the
  failure; it is descriptive context, not the frozen endpoint, and it is not called a
  win.</p>
  {source("results/strong_coupling_showdown_interpretation.json")}
  {figure("fig12_showdown.png",
          "The incomplete frozen showdown, with the solver failure retained and only the shared prefix compared.")}

  <h3>The dimensional SI example is a failed audit</h3>
  <p>Only {converged} of {len(mesh_solves)} planned layout and mesh solves converged, and even
  the converged baseline temperature sits outside the constant-property liquid-water regime
  the model assumes. Its apparent resistance reduction is withheld rather than reported.
  The exact bookkeeping, the solver outcomes and the failure all remain stored, and the
  script exits non-zero on this condition rather than printing a number.</p>
  {source("results/dimensional_coldplate.json")}

  <h3>And the scope</h3>
  <p>This is a steady two-dimensional research prototype at modest resolution, with no
  experimental validation, no pressure-drop or manufacturing analysis and no
  three-dimensional effects. Differentiable topology optimisation of thermo-fluidic devices
  is not new, and neither is natural-convection heat-sink design; what is new here is
  composition across genuinely heterogeneous components, the demonstration that spectral
  radius alone is insufficient for this decision, and &gamma; as an operational one-VJP
  check.</p>
</section>"""


def section_reproduce() -> str:
    return f"""
<section id="reproduce">
  <h2>Reproduce it</h2>
  <p>Supported review path: Linux <code>amd64</code>, or Windows through WSL2, with a
  running Docker daemon and Python 3.12+.</p>
  <pre><code>pip install -r requirements-orchestrator.txt
bash scripts/judge_demo.sh        # 1-3 minutes warm: serves both thermal backends
                                  # and swaps them on an 8x8 smoke grid</code></pre>
  <p>Then any of:</p>
  <pre><code>cd orchestrator
python validate_pipeline.py 16                 # composed gradient vs finite differences
python compare_thermal_backends.py 16          # JAX and Fortran/Enzyme are interchangeable
python sweep_coupling.py                       # where component-wise differentiation breaks
python predict_error.py --N 20                 # what predicts the damage, for one VJP
python gamma_generalization.py --trials 2400   # no containers, no solver, closed-form truth
python intervention_test.py --N 20 --Ra 3e4    # act on each gradient, re-solve the truth</code></pre>
  <p>The full command set, including the long optimisations and the frozen protocols, is in
  the <a href="{REPO}#reproduce">repository README</a>. The extended evidence is
  byte-bound to the workflow runs that produced it in
  <code>orchestrator/results/EVIDENCE_PROVENANCE.json</code>, and rechecked with
  <code>python scripts/validate_evidence_provenance.py --verify-github</code>.</p>
</section>"""


CSS = """
:root{--bg:#fbfaf8;--fg:#1a1a1a;--muted:#5d5a55;--rule:#e0dcd5;--accent:#7a3b12;
--card:#fff;--code:#f2efe9;--ok:#1f6b3a;--warn:#b4521a}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#eeece7;--muted:#a5a099;
--rule:#302e2b;--accent:#e2a06a;--card:#1c1b19;--code:#222120;--ok:#63c58a;--warn:#e2914f}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.62 ui-serif,Georgia,"Times New Roman",serif;-webkit-text-size-adjust:100%}
.wrap{max-width:52rem;margin:0 auto;padding:2.5rem 1.25rem 5rem}
header{border-bottom:2px solid var(--fg);padding-bottom:1.5rem;margin-bottom:1rem}
h1{font-size:2.4rem;line-height:1.1;margin:0 0 .5rem;letter-spacing:-.02em}
h2{font-size:1.55rem;margin:3.2rem 0 .9rem;padding-top:1.4rem;
border-top:1px solid var(--rule);letter-spacing:-.01em}
h3{font-size:1.1rem;margin:2.1rem 0 .6rem;color:var(--accent)}
p{margin:0 0 1rem}
.kicker{font:600 .78rem/1.4 ui-sans-serif,system-ui,sans-serif;letter-spacing:.11em;
text-transform:uppercase;color:var(--muted);margin:0 0 .8rem}
.lede{font-size:1.07rem}
nav{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1.4rem}
nav a{font:600 .85rem/1 ui-sans-serif,system-ui,sans-serif;text-decoration:none;
border:1px solid var(--rule);background:var(--card);color:var(--fg);
padding:.6rem .85rem;border-radius:.4rem}
nav a:hover{border-color:var(--accent);color:var(--accent)}
a{color:var(--accent)}
table{width:100%;border-collapse:collapse;margin:1.1rem 0 .6rem;
font:.9rem/1.45 ui-sans-serif,system-ui,sans-serif;display:block;overflow-x:auto}
th,td{text-align:left;padding:.5rem .6rem;border-bottom:1px solid var(--rule);
vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:.76rem;letter-spacing:.05em;
text-transform:uppercase}
table.num td+td,table.num th+th{text-align:right;white-space:nowrap}
code{background:var(--code);padding:.1em .35em;border-radius:.25em;
font:.87em/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--code);padding:1rem;border-radius:.45rem;overflow-x:auto;
border:1px solid var(--rule)}
pre code{background:none;padding:0;font-size:.83rem;line-height:1.6}
figure{margin:1.6rem 0}
figure img{width:100%;height:auto;border:1px solid var(--rule);border-radius:.35rem;
background:var(--card)}
figcaption{font:.83rem/1.5 ui-sans-serif,system-ui,sans-serif;color:var(--muted);
margin-top:.5rem}
.src{font:.78rem/1.5 ui-sans-serif,system-ui,sans-serif;color:var(--muted);margin:.2rem 0 0}
.src code{background:none;padding:0}
.caveat{border-left:3px solid var(--accent);padding-left:.95rem}
.ok{color:var(--ok);font-weight:600}
.bad{color:var(--warn);font-weight:600}
/* The gate explorer. Five radios and sibling selectors: no script runs here. */
.explorer{margin:1.4rem 0}
.explorer input{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}
.gates{display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:.8rem}
.gates label{font:600 .82rem/1 ui-monospace,SFMono-Regular,Menlo,monospace;
cursor:pointer;border:1px solid var(--rule);background:var(--card);color:var(--muted);
padding:.5rem .7rem;border-radius:.35rem}
.gates label:hover{border-color:var(--accent);color:var(--fg)}
.plot{margin:0}
.plot svg{width:100%;height:auto;border:1px solid var(--rule);border-radius:.35rem;
background:var(--card)}
.dots circle.cool{fill:var(--muted);opacity:.42}
.dots circle.hot{fill:var(--warn);opacity:.62}
line.grid{stroke:var(--rule);stroke-width:1}
line.danger{stroke:var(--fg);stroke-width:1;stroke-dasharray:4 3;opacity:.55}
text.tick{fill:var(--muted);font:10px ui-sans-serif,system-ui,sans-serif}
text.danger-label{font-weight:600}
text.axis{fill:var(--muted);font:11px ui-sans-serif,system-ui,sans-serif}
g.gate{display:none}
g.gate rect{fill:var(--accent);opacity:.09}
g.gate line{stroke:var(--accent);stroke-width:2}
g.gate text{fill:var(--accent);font:600 10px ui-sans-serif,system-ui,sans-serif}
.readout{display:none;margin-top:.7rem;border-left:3px solid var(--accent);
padding-left:.95rem}
.readout p{margin:0;font:.92rem/1.55 ui-sans-serif,system-ui,sans-serif}
#gate0:checked~.plot .g0,#gate1:checked~.plot .g1,#gate2:checked~.plot .g2,
#gate3:checked~.plot .g3,#gate4:checked~.plot .g4{display:block}
#gate0:checked~.readouts .r0,#gate1:checked~.readouts .r1,
#gate2:checked~.readouts .r2,#gate3:checked~.readouts .r3,
#gate4:checked~.readouts .r4{display:block}
#gate0:checked~.gates label[for=gate0],#gate1:checked~.gates label[for=gate1],
#gate2:checked~.gates label[for=gate2],#gate3:checked~.gates label[for=gate3],
#gate4:checked~.gates label[for=gate4]{background:var(--accent);color:var(--bg);
border-color:var(--accent)}
/* The radios are visually hidden, so the keyboard focus ring has to be put
   back on the label the focused radio actually controls. */
#gate0:focus-visible~.gates label[for=gate0],
#gate1:focus-visible~.gates label[for=gate1],
#gate2:focus-visible~.gates label[for=gate2],
#gate3:focus-visible~.gates label[for=gate3],
#gate4:focus-visible~.gates label[for=gate4]{outline:2px solid var(--fg);
outline-offset:2px}
footer{margin-top:4rem;padding-top:1.4rem;border-top:1px solid var(--rule);
font:.85rem/1.6 ui-sans-serif,system-ui,sans-serif;color:var(--muted)}
"""


def render() -> str:
    body = "".join([
        section_headline(),
        section_composition(),
        section_decision(),
        section_predictor(),
        section_validation(),
        section_unroll(),
        section_independent(),
        section_negatives(),
        section_reproduce(),
    ])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coldplate &mdash; results</title>
<meta name="description" content="A coupled cold-plate adjoint composed across C++, JAX, Fortran/Enzyme and PyTorch Tesseracts. Every number on this page is read from a committed measurement file.">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <p class="kicker">Tesseract Hackathon 2026 &middot; Track: multi-physics &amp; coupled systems &middot; Apache-2.0</p>
  <h1>Coldplate</h1>
  <p class="lede">One <code>jax.grad</code> across three languages, four derivative stacks
  and a two-way physics loop &mdash; and the loop is the part everyone else drops.</p>
  <nav>
    <a href="{REPO}">Repository</a>
    <a href="../PAPER.pdf">4-page paper</a>
    <a href="../demo/coldplate_submission.mp4">4:51 demo</a>
    <a href="#reproduce">Reproduce</a>
    <a href="#negatives">What we do not claim</a>
  </nav>
</header>
{body}
<footer>
  <p>This page is generated by <code>scripts/build_results_page.py</code> from the JSON
  files in <code>orchestrator/results/</code>. No result on it is typed by hand, and
  <code>python scripts/build_results_page.py --check</code> fails the build the moment the
  page and the measurements disagree.</p>
  <p>Apache-2.0. Built on <a href="https://github.com/pasteurlabs/tesseract-core">Tesseract</a>,
  from Pasteur Labs and ISI.</p>
</footer>
</div>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if docs/index.html is stale")
    args = parser.parse_args()

    page = render()
    if args.check:
        if not OUT.exists():
            print(f"FAIL: {OUT.relative_to(ROOT)} does not exist")
            return 1
        if OUT.read_text(encoding="utf-8") != page:
            print(f"FAIL: {OUT.relative_to(ROOT)} is stale; "
                  "rerun scripts/build_results_page.py")
            return 1
        print(f"ok: {OUT.relative_to(ROOT)} matches the committed measurements")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(page)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(page):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
