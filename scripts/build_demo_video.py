#!/usr/bin/env python3
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Build the data-backed four-minute Coldplate submission video.

The narration is assembled from committed result JSON, synthesized one caption
at a time, normalized to broadcast-style loudness, and muxed with generated
1920x1080 evidence slides. No headline measurement is hand-entered here.

Requirements: ``pip install -r requirements-video.txt`` plus ``ffmpeg`` and
``ffprobe`` on PATH. Run after ``orchestrator/make_extended_figures.py`` can see
the three extended-evidence JSON files.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
from functools import lru_cache
import math
import os
from pathlib import Path
import shutil
import subprocess
import textwrap
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from validate_video import (
    probe_video,
    sha256_file,
    validate_probe,
    validate_release_video,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "orchestrator" / "results"
DEMO = ROOT / "demo"
BUILD = DEMO / "build"
WIDTH, HEIGHT = 1920, 1080
VOICE = "en-US-AndrewMultilingualNeural"
RATE = "+18%"

# Two narration engines. The difference between them is a licence position
# rather than a preference about voices.
#
#   edge   - the Microsoft Edge read-aloud service. Good voice, but the audio
#            is produced by a remote service whose terms are about reading
#            pages aloud in a browser, and we hold no written confirmation that
#            its output may be redistributed in a prize submission. The client
#            package's LGPL is not the issue; the service's terms are.
#   piper  - synthesis on this machine. Nothing is sent anywhere, so no
#            service terms govern the result. The voice is
#            en_US-ljspeech-high, trained from scratch on LJ Speech, which is
#            public domain; onnxruntime, which executes it, is MIT.
#
# Be precise about the rest: piper-tts 1.7.0 is itself GPL-3.0-or-later,
# because it links espeak-ng for phonemization. That governs redistributing
# *the program*, which this repository does not do -- it is an optional build
# dependency in requirements-video.txt, exactly like ffmpeg. A GPL program's
# output is not a derivative work of it, the same reason a binary built by GCC
# is not GPL. What we redistribute is the rendered MP4.
#
# `piper` is therefore the engine behind the rights-clean deliverable. Both are
# kept, because the honest comparison is part of the point.
ENGINES = ("edge", "piper")
PIPER_VOICE = "en_US-ljspeech-high"
# Below 1.0 speaks faster. Calibrated, not guessed: at 0.86 this script's own
# sentences run 303.8 s, past the guardrail, and speech time scales with the
# knob while the lead, section gaps and tail do not. 0.82 lands near 290 s,
# alongside the Edge render and comfortably inside the five-minute rule.
PIPER_LENGTH_SCALE = 0.82
PIPER_MODEL_ENV = "COLDPLATE_PIPER_MODEL"
# Pinned so a swapped or truncated download fails loudly instead of quietly
# changing the voice in a published artefact.
PIPER_MODEL_SHA256 = "5d4f08ba6a2a48c44592eed3ce56bf85e9de3dd4e20df90541ae68a8310c029a"
PIPER_MODEL_BYTES = 114199011
PIPER_MODEL_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "en/en_US/ljspeech/high/en_US-ljspeech-high.onnx"
)
LEAD_SECONDS = 0.5
SECTION_GAP_SECONDS = 0.25
TAIL_SECONDS = 1.0
CAPTION_WRAP_WIDTH = 58
MAX_CAPTION_LINES = 3


def _write_utf8_lf(path: Path, text: str) -> None:
    """Write byte-stable UTF-8 text without platform newline translation."""
    path.write_bytes(text.encode("utf-8"))


@dataclass(frozen=True)
class Section:
    title: str
    kicker: str
    claim: str
    asset: Path | None
    sentences: tuple[str, ...]
    dark: bool = False


def _load(name: str) -> Any:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"missing evidence {path}; run the extended evidence workflow")
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f} percent"


def make_story() -> list[Section]:
    gradient = _load("gradient_validation.json")
    intervention = _load("intervention_test.json")
    general = _load("gamma_generalization.json")
    predictor = _load("predictor_statistics.json")
    history = _load("history_composed_N96.json")
    showdown = _load("strong_coupling_showdown.json")
    robustness = _load("intervention_robustness_matrix_48.json")
    cavity = _load("de_vahl_davis.json")
    physical = _load("dimensional_coldplate.json")

    if len(showdown.get("branches", [])) != 3:
        raise ValueError("the showdown result must contain exactly three branches")
    if not robustness["summary"].get("study_complete"):
        raise ValueError("the 48-case robustness result is incomplete")
    if any(not row["solver"]["ok"] for row in cavity):
        raise ValueError("a de Vahl Davis nonlinear solve did not converge")
    if any(not row["solver"].get("fluid", {}).get("converged") for row in cavity):
        raise ValueError("the cavity result lacks inner nonlinear convergence evidence")
    if any(not row["solver"].get("fluid", {}).get("converged")
           for row in physical["layouts"].values()):
        raise ValueError("the dimensional result lacks inner nonlinear convergence evidence")
    if physical.get("comparison", {}).get("equal_material_budget") is not False:
        raise ValueError("the dimensional geometry comparison must disclose unequal material")
    if abs(float(physical["grid"]["represented_heat_load_W"]) - 1.0) > 1.0e-12:
        raise ValueError("the dimensional result does not preserve the stated one-watt load")
    mesh = physical.get("mesh_refinement")
    if not isinstance(mesh, dict) or mesh.get("grids") != [16, 24, 32]:
        raise ValueError("the dimensional case lacks its planned three-grid audit")
    baseline_solver = physical["layouts"]["baseline"]["solver"]
    finned_solver = physical["layouts"]["finned"]["solver"]
    if not (
        physical.get("evidence_valid") is False
        and mesh.get("all_solves_converged") is False
        and baseline_solver.get("ok") is True
        and finned_solver.get("ok") is False
    ):
        raise ValueError("the dimensional result does not match the retained partial-validity outcome")

    shortcut = gradient["stats_one-way"]
    fd_samples = gradient.get("fd")
    composed_samples = gradient.get("composed")
    if (not isinstance(fd_samples, list) or not isinstance(composed_samples, list)
            or not fd_samples or len(fd_samples) != len(composed_samples)):
        raise ValueError("the gradient result lacks aligned finite-difference samples")
    fd_values = [float(value) for value in fd_samples]
    composed_values = [float(value) for value in composed_samples]
    if not all(math.isfinite(value) for value in [*fd_values, *composed_values]):
        raise ValueError("the gradient validation samples must be finite")
    fd_norm = math.sqrt(sum(value * value for value in fd_values))
    if fd_norm == 0.0:
        raise ValueError("the finite-difference validation vector has zero norm")
    composed_fd_ppm = 1.0e6 * math.sqrt(sum(
        (actual - reference) ** 2
        for actual, reference in zip(composed_values, fd_values)
    )) / fd_norm
    intervention_rows = intervention.get("rows")
    intervention_trials = intervention.get("n_amplitudes")
    intervention_wins = intervention.get("exact_wins")
    if (not isinstance(intervention_rows, list) or not intervention_rows
            or isinstance(intervention_trials, bool)
            or not isinstance(intervention_trials, int)
            or intervention_trials != len(intervention_rows)
            or isinstance(intervention_wins, bool)
            or not isinstance(intervention_wins, int)):
        raise ValueError("the intervention result has inconsistent trial counts")
    intervention_pairs = [
        (float(row["J_exact_action"]), float(row["J_naive_action"]))
        for row in intervention_rows
    ]
    if not all(math.isfinite(value) for pair in intervention_pairs for value in pair):
        raise ValueError("the intervention objectives must be finite")
    computed_wins = sum(exact < naive for exact, naive in intervention_pairs)
    if intervention_wins != computed_wins:
        raise ValueError("the intervention win count does not match its recorded rows")
    largest = max(intervention_rows, key=lambda row: row["amplitude"])
    exact_delta = float(largest["delta_J_exact_action"])
    shortcut_delta = float(largest["delta_J_naive_action"])
    if not (math.isfinite(exact_delta) and math.isfinite(shortcut_delta)
            and exact_delta < shortcut_delta < 0.0):
        raise ValueError("the largest intervention does not support a more-cooling claim")
    realised_more = 100.0 * (abs(exact_delta) / abs(shortcut_delta) - 1.0)
    intervention_outcome = (
        f"The composed choice wins all {intervention_trials} tested action sizes."
        if intervention_wins == intervention_trials
        else (
            f"The composed choice wins {intervention_wins} of "
            f"{intervention_trials} tested action sizes."
        )
    )
    long_reduction = 100.0 * (history[0]["J"] - history[-1]["J"]) / history[0]["J"]

    branches = {row["method"]: row for row in showdown["branches"]}
    if set(branches) != {"composed", "one_way", "frozen"}:
        raise ValueError("the showdown result must contain each frozen method once")
    initial = [float(row["objectives"][0]) for row in branches.values()]
    if max(initial) - min(initial) > 1e-12:
        raise ValueError("the showdown branches do not share their initial objective")
    summary = showdown["summary"]
    showdown_complete = (
        showdown.get("complete") is True
        and summary.get("all_branches_complete") is True
        and all(row.get("complete") is True for row in branches.values())
    )
    if showdown_complete:
        reductions = {
            name: row["metrics"]["reduction_percent"]
            for name, row in branches.items()
        }
        condition = summary["frozen_success_condition_met"]
        if condition:
            showdown_outcome = (
                f"The composed branch cut the objective by {reductions['composed']:.2f} percent, "
                f"versus {reductions['one_way']:.2f} for the loop-cut branch and "
                f"{reductions['frozen']:.2f} with frozen flow."
            )
            showdown_claim = "COMPOSED FINISHES WITH THE LOWEST TRUE OBJECTIVE"
        else:
            showdown_outcome = (
                f"The frozen-protocol finish was {reductions['composed']:.2f}, "
                f"{reductions['one_way']:.2f}, and {reductions['frozen']:.2f} percent reduction; "
                "the stored result does not meet the frozen composed-win condition."
            )
            showdown_claim = "FROZEN-PROTOCOL OUTCOME REPORTED WITHOUT RETUNING"
        showdown_title = "Eight decisions at the strong setting"
    else:
        composed = branches["composed"]
        failure = composed.get("failure", {})
        proposals = composed.get("proposals", [])
        if not (
            summary.get("frozen_success_condition_met") is False
            and summary.get("final_objective_comparisons") == []
            and composed.get("completed_iterations") == 5
            and len(proposals) == 6
            and proposals[-1].get("status") == "candidate_not_converged"
            and failure.get("stage") == "candidate_forward"
            and failure.get("iteration") == 6
            and all(branches[name].get("complete") is True
                    and branches[name].get("completed_iterations") == 8
                    for name in ("one_way", "frozen"))
        ):
            raise ValueError("the incomplete showdown lacks its expected durable failure")
        common_horizon = min(
            int(row["completed_iterations"]) for row in branches.values()
        )
        prefix_reductions = {
            name: 100.0 * (
                float(row["objectives"][0])
                - float(row["objectives"][common_horizon])
            ) / float(row["objectives"][0])
            for name, row in branches.items()
        }
        showdown_outcome = (
            "The frozen eight-step endpoint has no winner because the composed "
            f"step-six candidate did not converge. Over the shared first {common_horizon} "
            f"accepted decisions, the descriptive reductions are "
            f"{prefix_reductions['composed']:.2f}, {prefix_reductions['one_way']:.2f}, "
            f"and {prefix_reductions['frozen']:.2f} percent; that common prefix was "
            "examined after the failure and is not the frozen endpoint."
        )
        showdown_claim = "NO EIGHT-STEP WINNER CLAIMED · SOLVER FAILURE RETAINED"
        showdown_title = "Frozen eight-step showdown: incomplete"

    pooled = robustness["summary"]
    wins = pooled["outcomes"]["exact_wins"]
    losses = pooled["outcomes"]["shortcut_wins"]
    ties = pooled["outcomes"]["tie"]
    comparable = pooled["comparable_cases"]
    noncomparable = pooled["attempts_recorded"] - comparable
    cluster = pooled["cluster_aware_seed_analysis"]
    cluster_lower_raw = cluster["bootstrap"]["lower"]
    if not cluster.get("all_planned_clusters_complete") or cluster_lower_raw is None:
        raise ValueError("the robustness result lacks a complete seed-cluster analysis")
    cluster_lower = 100.0 * cluster_lower_raw
    cluster_confidence = 100.0 * float(cluster["bootstrap"]["confidence_level"])
    seed_count = int(cluster["clusters_planned"])
    rayleigh_count = len(robustness["by_rayleigh_number"])
    attempts_recorded = int(pooled["attempts_recorded"])
    shortcut_label = "SHORTCUT WIN" if losses == 1 else "SHORTCUT WINS"
    tie_label = "TIE" if ties == 1 else "TIES"
    observation_strata = robustness.get("by_prior_observation_status")
    if not isinstance(observation_strata, dict):
        raise ValueError("the robustness result lacks the prior-observation disclosure")
    observed_attempts = observation_strata["observed_before_frozen_design"]["attempts_planned"]
    new_attempts = observation_strata["not_stored_before_frozen_design"]["attempts_planned"]

    max_cavity_error = 100.0 * max(
        error for row in cavity for error in row["relative_error"].values()
    )
    cavity_rayleigh = sorted(float(row["Ra"]) for row in cavity)
    cavity_metric_count = sum(len(row["relative_error"]) for row in cavity)
    cavity_rayleigh_label = " and ".join(f"{value:g}" for value in cavity_rayleigh)
    all_cavity_within = all(row["within_coarse_grid_tolerance"] for row in cavity)
    cavity_sentence = (
        f"At Rayleigh {cavity_rayleigh_label}, all {cavity_metric_count} Nusselt and centerline-velocity "
        f"metrics are within {max_cavity_error:.1f} percent of the published reference."
        if all_cavity_within
        else
        f"The two cavity cases converged; their largest coarse-grid reference error is "
        f"{max_cavity_error:.1f} percent, reported without hiding any metric."
    )
    physical_inputs = physical["physical_inputs"]
    case_width_mm = 1000.0 * float(physical_inputs["width_m"])
    case_height_mm = 1000.0 * float(physical_inputs["height_m"])
    case_depth_mm = 1000.0 * float(physical_inputs["depth_m"])
    case_heat_w = float(physical_inputs["heat_load_W"])
    mesh_solver_statuses = [
        bool(layout["solver"]["ok"])
        for row in mesh["rows"]
        for layout in row["layouts"].values()
    ]
    mesh_converged = sum(mesh_solver_statuses)
    mesh_attempted = len(mesh_solver_statuses)
    predictor_cases = int(predictor["n_converged"])
    synthetic_cases = int(general["overall"]["n"])
    optimization_iterations = len(history)

    return [
        Section(
            "The decision, not just the derivative",
            "COLDPLATE · DIFFERENTIABLE MULTI-PHYSICS",
            "WHEN EVERY COMPONENT IS RIGHT, THE COMPOSITION CAN STILL BE WRONG",
            RESULTS / "fig10_intervention.png",
            (
                "What if every component derivative is correct, yet the engineering decision is wrong?",
                "Coldplate asks where a limited amount of metal should go in a buoyancy-cooled chip.",
                "At strong coupling, cutting one feedback loop changes which cells we choose and how much cooling we realize.",
            ),
            dark=True,
        ),
        Section(
            "A real heterogeneous fixed point",
            "PYTORCH + C++/EIGEN + JAX OR FORTRAN/ENZYME",
            "NEWTON–KRYLOV USES JVPs · THE IMPLICIT ADJOINT USES VJPs",
            RESULTS / "fig5_architecture.png",
            (
                "The pipeline serves a PyTorch material map, a C plus plus Eigen flow solver, and a thermal solver.",
                "Temperature drives buoyancy; velocity advects heat, so the converged state is a two-way fixed point.",
                "Newton Krylov crosses the component boundary with J V P's; the implicit adjoint crosses it again with V J P's.",
                "The thermal slot swaps JAX autodiff for independent Fortran differentiated by Enzyme at LLVM I R, and the full coupled pipeline is checked again after the swap.",
            ),
        ),
        Section(
            "Cutting one loop corrupts the sensitivity",
            "FINITE DIFFERENCE VALIDATION AT STRONG COUPLING",
            f"LOOP-CUT ERROR {100*shortcut['rel_err']:.0f}% · WRONG SIGN IN {100*shortcut['sign_flip_frac']:.0f}% OF CELLS",
            RESULTS / "fig2_gradient_validation.png",
            (
                f"Finite-difference component samples confirm the composed gradient to {composed_fd_ppm:.1f} parts per million.",
                "The strongest shortcut still differentiates every component, but freezes temperature inside buoyancy.",
                f"At this strong setting it has {100*shortcut['rel_err']:.0f} percent relative error and wrong signs in {100*shortcut['sign_flip_frac']:.0f} percent of the design field, while the forward temperature gives no warning.",
            ),
        ),
        Section(
            "A fresh forward solve decides",
            "SAME RAW-DESIGN CELL COUNT · NO GRADIENT GETS TO GRADE ITSELF",
            (
                f"{intervention_wins} / {intervention_trials} EXACT WINS · "
                f"{realised_more:.0f}% MORE COOLING AT THE LARGEST STEP"
            ),
            RESULTS / "fig10_intervention.png",
            (
                "A norm is not an outcome, so each gradient proposes the same count and amplitude of positive and negative raw-design changes and is judged by a fresh coupled solve.",
                intervention_outcome,
                f"At the largest step it delivers {realised_more:.0f} percent more realized cooling under the same zero-sum raw-design rule.",
            ),
        ),
        Section(
            showdown_title,
            "RETROSPECTIVE FROZEN PROTOCOL · PRIOR OVERLAP DISCLOSED",
            showdown_claim,
            RESULTS / "fig12_showdown.png",
            (
                "We froze the repeated procedure before storing these trajectories, but the same operating point already had favorable one-step evidence, so this is follow-up rather than an untouched independent confirmation.",
                "The protocol gives every branch the same start, projected-volume target, eight update opportunities, and true candidate-solve budget.",
                showdown_outcome,
                "The failure is retained without parameter tuning or selective rerun.",
            ),
        ),
        Section(
            f"{attempts_recorded} attempts, with overlap disclosed",
            f"{seed_count} FIXED SEEDS × {rayleigh_count} RAYLEIGH LEVELS",
            f"{wins} EXACT WINS · {losses} {shortcut_label} · {ties} {tie_label} · {noncomparable} NONCOMPARABLE",
            RESULTS / "fig13_robustness_matrix.png",
            (
                f"This retrospective frozen extension retains every failure; {observed_attempts} attempts overlap the earlier pilot and {new_attempts} cells had no stored result when the design was frozen.",
                f"Among {comparable} comparable cases, the exact action wins {wins}, the shortcut wins {losses}, and {ties} are ties.",
                f"The post-freeze descriptive {cluster_confidence:.0f} percent seed-cluster bootstrap interval has a lower endpoint of {cluster_lower:.1f} percent; the other {noncomparable} attempts remain visible as noncomparable, not deleted.",
            ),
        ),
        Section(
            "Physics outside the original design point",
            "DE VAHL DAVIS 1983 + AN EXPLICIT S I CONVERGENCE AUDIT",
            f"MAX CAVITY ERROR {max_cavity_error:.1f}% · FIN COMPARISON WITHHELD",
            RESULTS / "fig14_physics_validation.png",
            (
                "The de Vahl Davis reference activates full nonlinear Navier Stokes inertia, hot and cold side walls, and insulated horizontal walls.",
                cavity_sentence,
                f"A separate {case_width_mm:g} by {case_height_mm:g} by {case_depth_mm:g} millimeter sealed-water example maps every nondimensional group back to S I units and preserves exactly {case_heat_w:g} watt on the discretized chip.",
                f"Only {mesh_converged} of {mesh_attempted} planned layout and mesh solves converged; the N equals 32 finned solve stalled, so its apparent reduction is withheld rather than promoted as evidence.",
                "Even the converged baseline predicts a temperature above water's liquid range, outside the constant-property model used for this scaling exercise.",
                "The retained failure is a boundary on the dimensional illustration, not a performance or equal-material optimization claim.",
            ),
        ),
        Section(
            "One VJP tells us when to worry",
            "OBJECTIVE-AWARE ADJOINT RESIDUAL",
            f"PHYSICAL CORRELATION {predictor['log_gamma_correlation']:.3f} · {synthetic_cases:,}-SYSTEM CORRELATION {general['overall']['log_gamma_correlation']:.3f}",
            RESULTS / "fig8_predictor.png",
            (
                "The loop-cut adjoint's exact equation residual is Phi transpose g; normalizing it costs one V J P and retains the objective direction spectral radius discards.",
                f"Across {predictor_cases} converged physical configurations, its log correlation with measured error is {predictor['log_gamma_correlation']:.3f}.",
                f"Across {synthetic_cases:,} synthetic fixed points, it is {general['overall']['log_gamma_correlation']:.3f}, versus {general['overall']['rho_correlation']:.3f} for spectral radius.",
            ),
        ),
        Section(
            "A diagnostic with an honest boundary",
            "ATTRACTING AND REPELLING LOOPS ARE NOT THE SAME",
            f"REPELLING-SUBSET CORRELATION FALLS TO {general['repelling']['log_gamma_correlation']:.2f}",
            RESULTS / "fig11_generalization.png",
            (
                f"The limit is explicit: correlation falls to {general['repelling']['log_gamma_correlation']:.2f} when the loop repels, so the reusable PyTree utility provides no universal threshold and never calls that regime safe.",
                "An upstream-ready Tesseract JAX issue and test plan are prepared, but nothing will be submitted before publication review.",
            ),
        ),
        Section(
            "The optimized artefact",
            "A LIMITATION SHOWN, NOT HIDDEN",
            f"FULL RUN · {optimization_iterations} ITERATIONS · {long_reduction:.1f}% LOWER CHIP OBJECTIVE",
            RESULTS / "fig1_final.png",
            (
                "At the weaker-coupling topology-optimization start both gradients can descend, which is precisely why the strong-setting decision studies matter.",
                f"The full composed run over {optimization_iterations} iterations lowers the chip objective by {long_reduction:.1f} percent.",
                "It forms a branching conductor toward the cold sink while preserving channels for buoyant coolant flow.",
            ),
        ),
        Section(
            "Auditable in one judge path",
            "SOURCE BUILD OR DIGEST-PINNED RELEASE",
            "TESTS · CLAIM AUDIT · FOUR COMPONENT IMAGES · PAPER · CHECKSUMS",
            RESULTS / "fig5_architecture.png",
            (
                "Linux C I runs the tests and claim audit, while a separate job rebuilds all four component images and serves three at a time across the real derivative boundary.",
                "The August twenty-ninth release workflow records exact O C I digests, checksums the paper and video, and refuses to publish until anonymous pulls succeed.",
                "Coldplate is a two-way equilibrium whose composition changes a measured engineering decision, with the evidence and the failure modes attached.",
            ),
            dark=True,
        ),
    ]


@lru_cache(maxsize=16)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    path = next((candidate for candidate in names if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("Segoe UI or DejaVu Sans is required to render slides")
    return ImageFont.truetype(str(path), size=size)


def _wrapped(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def render_slide(section: Section, number: int, total: int, path: Path) -> None:
    bg = "#081827" if section.dark else "#f4f6f9"
    image = Image.new("RGB", (WIDTH, HEIGHT), bg)
    draw = ImageDraw.Draw(image)
    accent, navy, muted = "#26c6b2", "#10223d", "#647084"
    title_color = "white" if section.dark else navy
    draw.rectangle((0, 0, 18, HEIGHT), fill=accent)
    draw.text((78, 48), section.kicker, font=_font(24, True), fill=accent)
    draw.text((78, 88), section.title, font=_font(55, True), fill=title_color)
    draw.text((1730, 58), f"{number:02d} / {total:02d}", font=_font(22, True),
              fill="#9aa7b8" if section.dark else muted)

    # Keep the evidence and headline above a dedicated caption-safe band. The
    # rendered SRT never needs to cover a plot, claim, or footer to stay legible.
    content_box = (76, 176, 1844, 780)
    if section.asset is not None:
        asset = Image.open(section.asset).convert("RGB")
        max_w = content_box[2] - content_box[0] - 40
        max_h = content_box[3] - content_box[1] - 30
        scale = min(max_w / asset.width, max_h / asset.height)
        resized = asset.resize(
            (max(1, round(asset.width * scale)), max(1, round(asset.height * scale))),
            Image.Resampling.LANCZOS,
        )
        x = content_box[0] + (content_box[2] - content_box[0] - resized.width) // 2
        y = content_box[1] + (content_box[3] - content_box[1] - resized.height) // 2
        if section.dark:
            panel = Image.new("RGB", (content_box[2] - content_box[0], content_box[3] - content_box[1]), "white")
            image.paste(panel, (content_box[0], content_box[1]))
        image.paste(resized, (x, y))

    draw.rounded_rectangle((76, 805, 1844, 895), radius=18,
                           fill="#12334b" if section.dark else "#e5f5f2")
    claim_font = _font(28, True)
    lines = _wrapped(draw, section.claim, claim_font, 1690)
    start_y = 822 if len(lines) == 1 else 806
    for index, line in enumerate(lines[:2]):
        draw.text((112, start_y + index * 38), line, font=claim_font,
                  fill="#e9fffb" if section.dark else "#087b70")
    draw.rounded_rectangle(
        (76, 920, 1844, 1060), radius=18,
        fill="#0d2233" if section.dark else "#e7ebf1",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95)


def _resolve_piper_model(explicit: Path | None) -> Path:
    """Locate the ONNX voice and refuse to use one we did not pin."""
    candidates = [explicit] if explicit else []
    env = os.environ.get(PIPER_MODEL_ENV)
    if env:
        candidates.append(Path(env))
    candidates.extend([
        DEMO / "voices" / f"{PIPER_VOICE}.onnx",
        BUILD / "voices" / f"{PIPER_VOICE}.onnx",
    ])
    for candidate in candidates:
        if candidate and candidate.is_file():
            # Size first: a truncated or interrupted download is the common
            # failure and this reports it without hashing 109 MB to find out.
            size = candidate.stat().st_size
            if size != PIPER_MODEL_BYTES:
                raise ValueError(
                    f"{candidate} is {size} bytes, not the pinned "
                    f"{PIPER_MODEL_BYTES}; the download is incomplete or is a "
                    "different voice"
                )
            digest = sha256_file(candidate)
            if digest != PIPER_MODEL_SHA256:
                raise ValueError(
                    f"{candidate} has sha256 {digest}, not the pinned "
                    f"{PIPER_MODEL_SHA256}; refusing to narrate with an unverified voice"
                )
            return candidate
    raise FileNotFoundError(
        f"the {PIPER_VOICE} voice was not found. Download it once:\n"
        f"  curl -L -o demo/voices/{PIPER_VOICE}.onnx {PIPER_MODEL_URL}\n"
        f"  curl -L -o demo/voices/{PIPER_VOICE}.onnx.json {PIPER_MODEL_URL}.json\n"
        f"or set {PIPER_MODEL_ENV} to its path. Expected sha256 {PIPER_MODEL_SHA256}."
    )


@lru_cache(maxsize=2)
def _piper_voice(model: Path):
    from piper import PiperVoice

    return PiperVoice.load(str(model))


def _synthesize_piper(text: str, path: Path, model: Path, length_scale: float) -> None:
    """Synthesize one sentence locally. No network, no service terms."""
    import wave

    from piper import SynthesisConfig

    voice = _piper_voice(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        voice.synthesize_wav(
            text, handle, syn_config=SynthesisConfig(length_scale=length_scale)
        )
    if path.stat().st_size < 1000:
        raise RuntimeError(f"synthesized audio for {text!r} is unexpectedly small")


async def _synthesize_one(text: str, path: Path, voice: str, rate: str) -> None:
    import edge_tts

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            await edge_tts.Communicate(text, voice, rate=rate).save(str(path))
            if path.stat().st_size < 1000:
                raise RuntimeError("synthesized audio is unexpectedly small")
            return
        except Exception as exc:  # noqa: BLE001 - remote synthesis gets retries
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"narration synthesis failed: {last_error}")


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd or ROOT, check=True)


def _duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _sentence_timings(
    flattened: list[tuple[int, str]], durations: list[float]
) -> tuple[list[tuple[float, float, str]], float]:
    """Place speech on one clock with deliberate section-level breathing room."""
    if not flattened or len(flattened) != len(durations):
        raise ValueError("sentences and decoded-audio durations must be nonempty and aligned")
    timings: list[tuple[float, float, str]] = []
    cursor = LEAD_SECONDS
    for index, ((section_index, sentence), duration) in enumerate(
        zip(flattened, durations)
    ):
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("every decoded narration clip must have a positive duration")
        timings.append((cursor, cursor + duration, sentence))
        cursor += duration
        if (index + 1 < len(flattened)
                and flattened[index + 1][0] != section_index):
            cursor += SECTION_GAP_SECONDS
    return timings, cursor + TAIL_SECONDS


def _caption_chunks(
    sentence: str,
    *,
    width: int = CAPTION_WRAP_WIDTH,
    max_lines: int = MAX_CAPTION_LINES,
) -> list[str]:
    """Wrap and balance one spoken sentence into at most three-line cues."""
    if not sentence.strip() or width <= 0 or max_lines <= 0:
        raise ValueError("caption text and wrapping limits must be positive")
    lines = textwrap.wrap(
        sentence,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not lines:
        raise ValueError("caption sentence produced no visible text")
    groups = math.ceil(len(lines) / max_lines)
    base, extra = divmod(len(lines), groups)
    sizes = [base + (1 if index < extra else 0) for index in range(groups)]
    chunks, offset = [], 0
    for size in sizes:
        chunks.append("\n".join(lines[offset:offset + size]))
        offset += size
    if any(len(chunk.splitlines()) > max_lines for chunk in chunks):
        raise AssertionError("caption chunk exceeded its line limit")
    return chunks


def _caption_cues(
    timings: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Split long sentences and apportion their speech time across SRT cues."""
    cues: list[tuple[float, float, str]] = []
    for start, end, sentence in timings:
        chunks = _caption_chunks(sentence)
        weights = [max(1, len("".join(chunk.split()))) for chunk in chunks]
        total_weight = sum(weights)
        consumed = 0
        for index, (chunk, weight) in enumerate(zip(chunks, weights)):
            cue_start = start + (end - start) * consumed / total_weight
            consumed += weight
            cue_end = end if index == len(chunks) - 1 else (
                start + (end - start) * consumed / total_weight
            )
            cues.append((cue_start, cue_end, chunk))
    return cues


def _ffconcat(paths_and_durations: list[tuple[Path, float | None]], destination: Path) -> None:
    lines = ["ffconcat version 1.0"]
    for path, duration in paths_and_durations:
        escaped = path.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
        if duration is not None:
            lines.append(f"duration {duration:.6f}")
    _write_utf8_lf(destination, "\n".join(lines) + "\n")


def _decode_narration_clip(source: Path, destination: Path) -> None:
    """Decode a remote TTS MP3 to the common PCM format used by the timeline."""
    if destination.exists() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(destination),
    ])


def _make_silence(destination: Path, duration: float) -> None:
    """Create exact-format PCM silence for a lead, section gap, or tail."""
    if destination.exists():
        return
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
        "-t", f"{duration:.6f}", "-c:a", "pcm_s16le", str(destination),
    ])


def _write_script(sections: list[Section], timings: list[tuple[float, float, str]], duration: float) -> None:
    lines = [
        f"# Demo video script (rendered {_timestamp(duration).replace(',', '.')})",
        "",
        "This narration is generated from committed result JSON by",
        "`scripts/build_demo_video.py`; the timestamps below match the rendered audio.",
        "",
    ]
    timing_index = 0
    for section in sections:
        start = timings[timing_index][0]
        end = timings[timing_index + len(section.sentences) - 1][1]
        lines.extend([
            f"## {_timestamp(start)[:-4]}–{_timestamp(end)[:-4]} — {section.title}",
            "",
            "*On screen: `"
            + (section.asset.relative_to(ROOT).as_posix() if section.asset else "generated title")
            + "`.*",
            "",
        ])
        for sentence in section.sentences:
            lines.append(f"> {sentence}")
            lines.append(">")
        lines.append("")
        timing_index += len(section.sentences)
    _write_utf8_lf(ROOT / "DEMO_SCRIPT.md", "\n".join(lines))


def variant_paths(variant: str | None) -> dict[str, Path]:
    """Where one narration variant's deliverables live.

    The canonical (Edge-narrated) render keeps the names the release manifest,
    the README and the validator already refer to. Any other variant gets a
    parallel set, so a second narration can ship beside the first without
    either one silently overwriting the other.
    """
    if variant is None:
        stem = "coldplate_submission"
        manifest = "video_manifest.json"
    else:
        if not variant.replace("_", "").isalnum():
            raise ValueError("a variant name must be alphanumeric with underscores")
        stem = f"coldplate_submission_{variant}"
        manifest = f"video_manifest_{variant}.json"
    return {
        "video": DEMO / f"{stem}.mp4",
        "captions": DEMO / f"{stem}.en.srt",
        "manifest": DEMO / manifest,
        # The slides do not depend on the voice, so every variant shares one
        # poster -- and the build asserts that rather than assuming it.
        "poster": DEMO / "poster.png",
    }


def build(
    voice: str = VOICE,
    rate: str = RATE,
    output: Path | None = None,
    *,
    engine: str = "edge",
    variant: str | None = None,
    piper_model: Path | None = None,
    length_scale: float = PIPER_LENGTH_SCALE,
) -> dict[str, Any]:
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            raise FileNotFoundError(f"{command} is required")
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, not {engine!r}")
    paths = variant_paths(variant)
    canonical_output = paths["video"].resolve()
    if output is not None and output.resolve() != canonical_output:
        raise ValueError(
            f"--output must be {paths['video'].relative_to(ROOT).as_posix()} for this "
            "variant, because the release manifest and validator require a named "
            "deliverable rather than an arbitrary path"
        )
    if engine == "piper":
        piper_model = _resolve_piper_model(piper_model)
        voice, rate = PIPER_VOICE, f"length_scale={length_scale:g}"
    sections = make_story()
    DEMO.mkdir(exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)

    slides: list[Path] = []
    for index, section in enumerate(sections, 1):
        slide = BUILD / f"slide-{index:02d}.png"
        render_slide(section, index, len(sections), slide)
        slides.append(slide)
    poster = paths["poster"]
    if variant is not None and poster.exists():
        # The visual track is voice-independent. Prove it instead of trusting it:
        # a variant that would change the shared poster is a bug, not a build.
        if sha256_file(slides[0]) != sha256_file(poster):
            raise ValueError(
                "this variant renders a different first slide from the canonical "
                "build; the poster is shared, so the slides must be identical"
            )
    else:
        shutil.copyfile(slides[0], poster)

    flattened: list[tuple[int, str]] = [
        (section_index, sentence)
        for section_index, section in enumerate(sections)
        for sentence in section.sentences
    ]
    pcm_paths: list[Path] = []
    suffix = "mp3" if engine == "edge" else "wav"
    for index, (_, sentence) in enumerate(flattened):
        digest = hashlib.sha256(
            f"{engine}\0{voice}\0{rate}\0{sentence}".encode()
        ).hexdigest()[:12]
        path = BUILD / f"voice-{index:03d}-{digest}.{suffix}"
        if not path.exists():
            print(f"synthesizing {index + 1}/{len(flattened)} [{engine}]", flush=True)
            if engine == "edge":
                asyncio.run(_synthesize_one(sentence, path, voice, rate))
            else:
                _synthesize_piper(sentence, path, piper_model, length_scale)
        pcm = BUILD / f"pcm-{index:03d}-{digest}.wav"
        _decode_narration_clip(path, pcm)
        pcm_paths.append(pcm)

    durations = [_duration(path) for path in pcm_paths]
    timings, planned_duration = _sentence_timings(flattened, durations)

    lead = BUILD / "silence-lead-0500ms.wav"
    gap = BUILD / "silence-section-0250ms.wav"
    tail = BUILD / "silence-tail-1000ms.wav"
    for silence, duration in (
        (lead, LEAD_SECONDS),
        (gap, SECTION_GAP_SECONDS),
        (tail, TAIL_SECONDS),
    ):
        _make_silence(silence, duration)
        if abs(_duration(silence) - duration) > 0.005:
            raise ValueError(f"silence clip {silence.name} has the wrong duration")

    audio_timeline: list[Path] = [lead]
    slide_entries: list[tuple[Path, float | None]] = [(slides[0], LEAD_SECONDS)]
    for index, (((section_index, _), pcm), duration) in enumerate(
        zip(zip(flattened, pcm_paths), durations)
    ):
        audio_timeline.append(pcm)
        slide_entries.append((slides[section_index], duration))
        if (index + 1 < len(flattened)
                and flattened[index + 1][0] != section_index):
            audio_timeline.append(gap)
            slide_entries.append((slides[section_index], SECTION_GAP_SECONDS))
    audio_timeline.append(tail)
    slide_entries.append((slides[-1], TAIL_SECONDS))

    if not 180.0 <= planned_duration <= 295.0:
        raise ValueError(
            f"narration duration {planned_duration:.1f}s is outside the 3:00–4:55 guardrail"
        )

    srt_lines = []
    caption_cues = _caption_cues(timings)
    for index, (start, end, caption) in enumerate(caption_cues, 1):
        srt_lines.extend([
            str(index), f"{_timestamp(start)} --> {_timestamp(end)}", caption, "",
        ])
    captions = paths["captions"]
    _write_utf8_lf(captions, "\n".join(srt_lines))

    audio_concat = BUILD / "audio.ffconcat"
    _ffconcat([(path, None) for path in audio_timeline], audio_concat)
    narration = BUILD / "narration.wav"
    _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat",
          "-safe", "0", "-i", str(audio_concat), "-ar", "48000", "-ac", "1",
          "-c:a", "pcm_s16le", str(narration)])
    narration_duration = _duration(narration)
    if abs(narration_duration - planned_duration) > 0.05:
        raise ValueError(
            "assembled narration duration differs from its caption timeline: "
            f"{narration_duration:.3f}s versus {planned_duration:.3f}s"
        )

    # The concat demuxer needs the final still repeated for its last duration.
    slide_entries.append((slides[flattened[-1][0]], None))
    slides_concat = BUILD / "slides.ffconcat"
    _ffconcat(slide_entries, slides_concat)
    silent = BUILD / "silent.mp4"
    _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat",
          "-safe", "0", "-i", str(slides_concat), "-vf", "fps=30,format=yuv420p",
          "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-an", str(silent)])

    final = canonical_output
    caption_filter = (
        "subtitles=filename='" + captions.resolve().as_posix().replace(":", "\\:") + "':"
        "force_style='FontName=DejaVu Sans,FontSize=18,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00101A24,BorderStyle=3,BackColour=&H80081420,Outline=1,"
        "Shadow=0,MarginV=32,Alignment=2'"
    )
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(silent), "-i", str(narration),
        "-vf", caption_filter,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "1",
        "-metadata", "title=Coldplate — differentiating a multi-physics fixed point",
        "-metadata", "comment=Tesseract Hackathon 2026 submission",
        "-movflags", "+faststart", "-shortest", str(final),
    ])
    media = validate_probe(probe_video(final))
    rendered_duration = media["duration_seconds"]
    if variant is None:
        # DEMO_SCRIPT.md documents the canonical render's timings; a variant
        # must not silently rewrite them with its own.
        _write_script(sections, timings, rendered_duration)
    report = {
        "output": final.relative_to(ROOT).as_posix(),
        "duration_seconds": rendered_duration,
        "width": media["width"],
        "height": media["height"],
        "video_codec": media["video_codec"],
        "pixel_format": media["pixel_format"],
        "audio_codec": media["audio_codec"],
        "audio_sample_rate_hz": media["audio_sample_rate_hz"],
        "audio_channels": media["audio_channels"],
        "engine": engine,
        "variant": variant,
        "voice": voice,
        "rate": rate,
        "sections": len(sections),
        "captions": captions.relative_to(ROOT).as_posix(),
        "caption_cues": len(caption_cues),
        "captions_sha256": sha256_file(captions),
        "captions_bytes": captions.stat().st_size,
        "poster": poster.relative_to(ROOT).as_posix(),
        "poster_sha256": sha256_file(poster),
        "poster_bytes": poster.stat().st_size,
        "poster_width": WIDTH,
        "poster_height": HEIGHT,
        "poster_format": "PNG",
        "lead_seconds": LEAD_SECONDS,
        "section_gap_seconds": SECTION_GAP_SECONDS,
        "tail_seconds": TAIL_SECONDS,
        "sha256": sha256_file(final),
        "bytes": final.stat().st_size,
    }
    manifest = paths["manifest"]
    _write_utf8_lf(manifest, json.dumps(report, indent=2) + "\n")
    # Re-read the committed-deliverable shape through the same validator used
    # on release day.  This catches muxing surprises and stale manifests.
    validate_release_video(final, manifest, captions, poster)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default=VOICE,
                        help="Edge voice name; ignored when --engine piper")
    parser.add_argument("--rate", default=RATE,
                        help="Edge speaking rate; ignored when --engine piper")
    parser.add_argument("--engine", default="edge", choices=ENGINES,
                        help="narration engine (default: edge)")
    parser.add_argument("--variant", default=None,
                        help="name a parallel deliverable set, e.g. --variant local_voice")
    parser.add_argument("--piper-model", type=Path, default=None,
                        help=f"path to the pinned {PIPER_VOICE}.onnx voice")
    parser.add_argument("--length-scale", type=float, default=PIPER_LENGTH_SCALE,
                        help="Piper pace; below 1.0 speaks faster")
    parser.add_argument("--output", type=Path)
    build(**vars(parser.parse_args()))
