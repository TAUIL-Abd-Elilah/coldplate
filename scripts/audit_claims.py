#!/usr/bin/env python3
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Check that the headline numbers quoted in README/PAPER are measured.

Written after several claims in this repository turned out to be stale or wrong
-- a gradient comparison that measured a mean-removal instead of the coupling
and a convergence order extrapolated through a stalled solve. Prose drifts from
data silently, so this re-derives the canonical values from the stored result
files and asserts that the documents agree.

    usage:  python scripts/audit_claims.py
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "orchestrator" / "results"

FAIL: list[str] = []
NOTE: list[str] = []


def load(name, sub="results"):
    p = ROOT / "orchestrator" / sub / name
    if not p.exists():
        NOTE.append(f"missing result file: {sub}/{name}")
        return None
    return json.loads(p.read_text())


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(label)


def docs_contain(needle: str, *, files=("README.md", "PAPER.md")) -> list[str]:
    """Which documents mention this exact string."""
    out = []
    for f in files:
        if needle in (ROOT / f).read_text(encoding="utf-8"):
            out.append(f)
    return out


def main() -> int:
    print("=== canonical values, re-derived from stored results ===\n")

    # ---- critical Rayleigh number -------------------------------------
    rows = load("critical_rayleigh.json")
    if rows:
        best = min(rows, key=lambda r: abs(r["excess"]))
        ra = best["ra_classical"]
        print(f"critical Rayleigh: finest aspect={best['aspect']} "
              f"Ra_c={ra:.2f} excess={100*best['excess']:+.3f}%")
        check("Ra_c within 0.1% of 1707.762 at the widest box",
              abs(ra / 1707.762 - 1) < 1e-3, f"measured {ra:.2f}")
        check("README/PAPER quote 1707.97", bool(docs_contain("1707.97")))
        # monotone approach from above
        srt = sorted(rows, key=lambda r: r["aspect"])
        check("Ra_c decreases monotonically with aspect ratio",
              all(a["ra_classical"] > b["ra_classical"]
                  for a, b in zip(srt, srt[1:])))
        check("every measured Ra_c exceeds the unbounded value",
              all(r["ra_classical"] > 1707.762 for r in rows))

    # ---- grid convergence ---------------------------------------------
    rows = load("grid_convergence.json")
    if rows:
        Js = {r["N"]: r["J"] for r in rows}
        print(f"grid convergence: J(16)={Js.get(16):.6f} .. J(96)={Js.get(96):.6f}")
        check("all grids converged", all(r["ok"] for r in rows))
        seq = [Js[n] for n in sorted(Js)]
        check("J is monotone in N (asymptotic range)",
              all(a < b for a, b in zip(seq, seq[1:]))
              or all(a > b for a, b in zip(seq, seq[1:])))
        import math
        for trio in ((16, 32, 64), (24, 48, 96)):
            if all(n in Js for n in trio):
                d1, d2 = Js[trio[0]] - Js[trio[1]], Js[trio[1]] - Js[trio[2]]
                p = math.log(d1 / d2) / math.log(2.0)
                print(f"  observed order on {trio}: {p:.3f}")
                check(f"order on {trio} is between 1.5 and 2.5 (2nd order)",
                      1.5 < p < 2.5, f"p={p:.3f}")

    # ---- predictor comparison -----------------------------------------
    rows = load("predict_error.json")
    if rows:
        import numpy as np
        g = np.array([r["gamma"] for r in rows])
        e = np.array([r["rel_err"] for r in rows])
        s = np.array([r["rho_phi"] for r in rows])
        lg = np.corrcoef(np.log10(np.maximum(g, 1e-12)), np.log10(np.maximum(e, 1e-12)))[0, 1]
        ls = np.corrcoef(s, np.log10(np.maximum(e, 1e-12)))[0, 1]
        print(f"predictors: corr(log gamma)={lg:.4f}  corr(rho)={ls:.4f}  n={len(rows)}")
        check("README/PAPER quote 0.995 for log-gamma correlation",
              bool(docs_contain("0.995")) and abs(lg - 0.995) < 0.005,
              f"measured {lg:.4f}")
        check("README/PAPER quote 0.825 for rho correlation",
              bool(docs_contain("0.825")) and abs(ls - 0.825) < 0.005,
              f"measured {ls:.4f}")
        # the ordering-reversal pair
        pair = {r["design"]: r for r in rows
                if r["design"] in ("rough", "smooth") and abs(r["Ra"] - 1e4) < 1}
        if len(pair) == 2:
            ro, sm = pair["rough"], pair["smooth"]
            check("rho orders the Ra=1e4 pair backwards",
                  sm["rho_phi"] < ro["rho_phi"] and sm["rel_err"] > ro["rel_err"],
                  f"rough rho={ro['rho_phi']:.3f} err={ro['rel_err']:.3f} | "
                  f"smooth rho={sm['rho_phi']:.3f} err={sm['rel_err']:.3f}")
            check("gamma orders that pair correctly",
                  sm["gamma"] > ro["gamma"])
            for v in (f"{ro['rho_phi']:.3f}", f"{sm['rho_phi']:.3f}"):
                check(f"docs quote rho={v}", bool(docs_contain(v)))

    stats = load("predictor_statistics.json")
    if stats:
        check("predictor report states 14 converged of 20 attempted",
              stats["n_converged"] == 14
              and stats["attempted_configurations"] == 20)
        holdouts = stats["leave_one_family_out"]
        check("log-gamma correlation survives every family holdout",
              len(holdouts) == 4
              and min(r["log_gamma_correlation"] for r in holdouts.values()) > 0.98)
        lo, hi = stats["bootstrap_95_percent_interval"]
        check("bootstrap interval supports the predictor correlation",
              lo > 0.98 and hi <= 1.0, f"95% interval [{lo:.4f}, {hi:.4f}]")

    # ---- objective sweep ----------------------------------------------
    rows = load("objective_sweep.json")
    if rows:
        import numpy as np
        e = np.array([r["rel_err"] for r in rows])
        rp = {round(r["rho_phi"], 6) for r in rows}
        spread = e.max() / e.min()
        print(f"objective sweep: rho values seen={rp}  error spread={spread:.1f}x")
        check("rho is identical across every objective", len(rp) == 1)
        check("error spread is the ~136x quoted",
              abs(spread - 136) < 8, f"measured {spread:.1f}x")
        check("docs quote 0.5481 for the constant rho",
              bool(docs_contain("0.5481")))

    # ---- optimisation --------------------------------------------------
    for tag, quoted in (("composed", "1.2588"), ("one_way", "1.2576")):
        h = load(f"history_{tag}_N96.json")
        if h:
            J0, J1 = h[0]["J"], h[-1]["J"]
            red = 100 * (J0 - J1) / J0
            print(f"optimisation {tag}: J {J0:.4f} -> {J1:.4f}  ({red:.1f}% reduction)")
            check(f"{tag} final J matches the quoted {quoted}",
                  f"{J1:.4f}" == quoted, f"actual {J1:.4f}")
            check(f"{tag} reduction is the quoted 84.6%", abs(red - 84.6) < 0.15,
                  f"{red:.2f}%")

    # ---- sensitivity attribution ---------------------------------------
    rep = load("sensitivity_ranking.json")
    if rep:
        ow = rep["one_way"]
        by_k = {r["k"]: r for r in ow["per_k"]}
        sp = ow["spearman_magnitude"]
        n = ow["n_cells"]
        print(f"attribution: Spearman(|g|)={sp:+.4f} over {n} cells, "
              f"recall@50={by_k[50]['recall']:.0%}")
        check("naive influence ranking is no better than chance",
              abs(sp) < 0.05, f"Spearman {sp:+.4f}")
        check("README quotes the measured Spearman", bool(docs_contain("0.011")))
        check("naive misses the single most influential cell",
              not ow["top1_correct"],
              f"its pick is truly #{ow['top1_true_rank_of_naive_pick']+1}")
        check("recall of the true top 50 is the quoted 56%",
              abs(by_k[50]["recall"] - 0.56) < 0.01,
              f"{by_k[50]['recall']:.0%}")
        check("signs on the true top 50 are all correct",
              by_k[50]["sign_agreement_on_true_topk"] == 1.0,
              "this is why descent still works")
        worst = by_k[50]["worst_true_rank_promoted"] + 1
        check(f"docs quote the worst promoted cell as #{worst} of {n}",
              bool(docs_contain(str(worst))) and bool(docs_contain(str(n))),
              f"measured #{worst} of {n}")

    # ---- gamma-gated adjoint -------------------------------------------
    # Kept in its own directory: this is an 80-iteration cost comparison, and
    # writing it into results/ would silently overwrite the 120-iteration
    # N=48 runs that back the replication claim in the README.
    gated = load("history_gamma_gated_N48.json", sub="results_gate")
    base = load("history_composed_N48.json", sub="results_gate")
    if gated and base:
        cheap = sum(1 for r in gated if r.get("gate") == "cheap")
        spent = sum(r.get("adjoint_matvecs", 0) for r in gated)
        exact_cost = sum(r.get("adjoint_matvecs", 0) for r in base)
        Jg, Jb = gated[-1]["J"], base[-1]["J"]
        gap = abs(Jg - Jb) / abs(Jb)
        print(f"gamma gate: cheap on {cheap}/{len(gated)} iterations, "
              f"{spent}+{len(gated)} VJPs vs {exact_cost} always-exact, "
              f"final J {Jg:.4f} vs {Jb:.4f} ({100*gap:.2f}%)")
        check("the gated final objective stays within 2% of the exact run",
              gap < 0.02 and bool(docs_contain("0.51%")),
              f"final J differs by {100*gap:.2f}%")
        if exact_cost:
            saved = 1 - (spent + len(gated)) / exact_cost
            check("the gate costs less than always paying for the adjoint",
                  saved > 0.5, f"{100*saved:.0f}% fewer VJPs "
                  f"({spent + len(gated)} vs {exact_cost})")
        check("every gamma recorded is below the gate that was set",
              all(r["gamma"] < 0.10 for r in gated if r.get("gate") == "cheap"))
        check("gamma stays in the MARGINAL band throughout",
              all(0.001 < r["gamma"] < 0.10 for r in gated),
              f"range {min(r['gamma'] for r in gated):.4f}"
              f"-{max(r['gamma'] for r in gated):.4f}")

    # ---- forward-validated intervention -------------------------------
    intervention = load("intervention_test.json")
    if intervention:
        rows = intervention["rows"]
        check("exact-gradient intervention wins every equal raw-design re-solve",
              intervention["exact_wins"] == intervention["n_amplitudes"]
              and all(r["exact_advantage"] > 0 for r in rows))
        check("every exact-gradient intervention reduces the true objective",
              all(r["delta_J_exact_action"] < 0 for r in rows))
        last = max(rows, key=lambda r: r["amplitude"])
        ratio = last["delta_J_exact_action"] / last["delta_J_naive_action"]
        check("largest action realizes the quoted ~58% extra cooling",
              1.55 < ratio < 1.61 and bool(docs_contain("58%")),
              f"measured {100*(ratio-1):.1f}%")

    robustness = load("intervention_robustness.json")
    if robustness and "seeds_comparable" not in robustness:
        # Superseded schema: the first version of that sweep hand-picked its
        # seeds and raised on a loss, so it could not report one. Refuse to
        # audit it rather than quietly validate numbers it could not have
        # falsified.
        check("intervention robustness uses the independently evaluated sweep", False,
              "results predate the independent-action rewrite; re-run "
              "orchestrator/intervention_robustness.py")
        robustness = None
    if robustness:
        comparable = robustness["seeds_comparable"]
        wins = robustness["exact_wins"]
        cases = [
            c for c in robustness["cases"]
            if c["outcome"] in {"exact_wins", "shortcut_wins", "tie"}
        ]
        print(f"intervention robustness: {wins} wins, "
              f"{robustness['shortcut_wins']} observed losses over "
              f"{comparable} comparable designs; "
              f"{robustness['seeds_inconclusive']} inconclusive action pair, "
              f"{robustness['base_state_failures']} failed base solve")
        check("the seed range is fixed and not selected after individual results",
              "Fixed contiguous seed range" in robustness.get("selection_note", ""))
        check("the sweep is a contiguous range with no gaps",
              [c["seed"] for c in robustness["cases"]]
              == list(range(robustness["cases"][0]["seed"],
                            robustness["cases"][0]["seed"]
                            + robustness["seeds_attempted"])))
        check("every attempted seed is accounted for",
              comparable + robustness["seeds_inconclusive"]
              + robustness["base_state_failures"]
              == robustness["seeds_attempted"])
        check("losses were recordable, and are counted if they occurred",
              wins + robustness["shortcut_wins"] + robustness["ties"]
              == comparable)
        check("the composed choice won every comparable design",
              wins == comparable and comparable > 0, f"{wins}/{comparable}")
        check("README/PAPER disclose wins, observed losses, and inconclusive cases",
              bool(docs_contain("10 wins"))
              and bool(docs_contain("0 observed losses"))
              and bool(docs_contain("2 inconclusive")))
        check("every comparable design realizes more cooling",
              all(c["extra_cooling_fraction"] > 0 for c in cases))
        med = robustness["median_extra_cooling_when_winning"]
        check("docs quote the median extra cooling as 36%",
              abs(100 * med - 36) < 1.5 and bool(docs_contain("36%")),
              f"measured {100*med:.0f}%")
        best = robustness["max_extra_cooling_when_winning"]
        check("largest robustness advantage is the quoted ~276%",
              2.7 < best < 2.8 and bool(docs_contain("276%")),
              f"measured {100*best:.1f}%")

    # ---- retrospectively frozen repeated-decision showdown -------------
    showdown = load("strong_coupling_showdown.json")
    if showdown:
        protocol_path = ROOT / showdown["protocol_file"]
        protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        protocol = showdown["protocol"]
        summary = showdown["summary"]
        check("showdown embeds the exact committed protocol bytes",
              showdown["protocol_sha256"] == protocol_hash)
        check("showdown provenance is retrospective rather than mislabelled preregistration",
              protocol["status"] == "retrospectively_frozen_design"
              and "prior_observation_disclosure" in protocol
              and bool(docs_contain("retrospective")))
        branches = showdown.get("branches", [])
        branch_by_method = {
            branch.get("method"): branch for branch in branches
            if isinstance(branch, dict)
        }
        expected_methods = {"composed", "one_way", "frozen"}
        initial_values = [
            branch.get("objectives", [None])[0]
            for branch in branches if branch.get("objectives")
        ]
        shared_initial = (
            len(branches) == 3
            and set(branch_by_method) == expected_methods
            and len(initial_values) == 3
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(float(value)) for value in initial_values)
            and max(initial_values) - min(initial_values) <= 1e-12
        )
        check("showdown branches share one measured initial objective", shared_initial)

        if showdown.get("complete") is True:
            check("all showdown branches complete all eight accepted decisions",
                  summary["all_branches_complete"] is True
                  and summary["required_iterations_per_branch"] == 8
                  and summary["common_initial_objective_verified"] is True
                  and all(branch["completed_iterations"] == 8
                          and len(branch["rows"]) == 8
                          and len(branch["proposals"]) == 8
                          and len(branch["objectives"]) == 9
                          and all(row["status"] == "accepted" for row in branch["rows"])
                          for branch in branches))
            reductions = {
                branch["method"]: branch["metrics"]["reduction_percent"]
                for branch in branches
            }
            print("showdown reductions: " + ", ".join(
                f"{name}={value:.2f}%" for name, value in reductions.items()
            ))
            for method, value in reductions.items():
                check(f"README/PAPER quote the {method} showdown reduction",
                      bool(docs_contain(f"{value:.2f}%")),
                      f"measured {value:.2f}%")
            if summary["frozen_success_condition_met"]:
                check("stored comparisons support the documented composed showdown win",
                      all(row["relation"] == "composed_lower"
                          for row in summary["final_objective_comparisons"]))
        else:
            composed = branch_by_method.get("composed", {})
            shortcuts = [branch_by_method.get(name, {}) for name in ("one_way", "frozen")]
            failure = composed.get("failure", {})
            proposals = composed.get("proposals", [])
            frozen_failure_is_durable = (
                summary.get("all_branches_complete") is False
                and summary.get("frozen_success_condition_met") is False
                and summary.get("final_objective_comparisons") == []
                and composed.get("complete") is False
                and composed.get("completed_iterations") == 5
                and len(composed.get("rows", [])) == 5
                and len(proposals) == 6
                and all(row.get("status") == "accepted"
                        for row in composed.get("rows", []))
                and proposals[-1].get("status") == "candidate_not_converged"
                and failure.get("stage") == "candidate_forward"
                and failure.get("iteration") == 6
                and math.isclose(
                    float(failure.get("residual", math.nan)),
                    float(proposals[-1].get("candidate_residual", math.inf)),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                and all(branch.get("complete") is True
                        and branch.get("completed_iterations") == 8
                        and len(branch.get("rows", [])) == 8
                        and len(branch.get("proposals", [])) == 8
                        and branch.get("failure") is None
                        for branch in shortcuts)
            )
            check("incomplete showdown preserves the frozen candidate failure",
                  frozen_failure_is_durable,
                  f"composed stopped at step {failure.get('iteration')} with "
                  f"residual {failure.get('residual')}")
            common_horizon = min(
                int(branch.get("completed_iterations", 0)) for branch in branches
            ) if branches else 0
            prefix_reductions = {
                method: 100.0 * (
                    float(branch["objectives"][0])
                    - float(branch["objectives"][common_horizon])
                ) / float(branch["objectives"][0])
                for method, branch in branch_by_method.items()
                if len(branch.get("objectives", [])) > common_horizon
            }
            print(
                f"showdown: no eight-step endpoint verdict; common {common_horizon}-step "
                "descriptive reductions: "
                + ", ".join(
                    f"{name}={value:.2f}%" for name, value in prefix_reductions.items()
                )
            )
            check("showdown common-prefix comparison is five-step and fully observed",
                  common_horizon == 5
                  and set(prefix_reductions) == expected_methods)
            check("README/PAPER explicitly withhold an eight-step showdown verdict",
                  bool(docs_contain("no eight-step endpoint verdict")))
            check("README/PAPER label the showdown incomplete and non-evaluable",
                  bool(docs_contain("incomplete"))
                  and bool(docs_contain("not evaluable"))
                  and bool(docs_contain("post-hoc")))
            for method, value in prefix_reductions.items():
                check(f"README/PAPER quote the {method} five-step reduction",
                      bool(docs_contain(f"{value:.2f}%")),
                      f"measured {value:.2f}%")
            interpretation = load("strong_coupling_showdown_interpretation.json")
            source_hash = hashlib.sha256(
                (RES / "strong_coupling_showdown.json").read_bytes()
            ).hexdigest()
            interpreted_methods = (
                interpretation.get("descriptive_common_prefix", {}).get("methods", {})
                if isinstance(interpretation, dict) else {}
            )
            check("showdown interpretation is hash-bound and withholds primary ranking",
                  isinstance(interpretation, dict)
                  and interpretation.get("source_sha256") == source_hash
                  and interpretation.get("protocol_sha256") == showdown["protocol_sha256"]
                  and interpretation.get("execution_status")
                      == "incomplete_frozen_execution"
                  and interpretation.get("common_initial_objective_verified") is True
                  and interpretation.get("primary_endpoint", {}).get("evaluable") is False
                  and interpretation.get("primary_endpoint", {}).get("ranking") == []
                  and interpretation.get("primary_endpoint", {}).get("comparisons") == []
                  and interpretation.get("descriptive_common_prefix", {}).get("steps") == 5
                  and interpretation.get("descriptive_common_prefix", {}).get(
                      "pre_specified"
                  ) is False
                  and all(
                      method in interpreted_methods
                      and math.isclose(
                          interpreted_methods[method]["reduction_percent"],
                          value,
                          rel_tol=0.0,
                          abs_tol=1e-12,
                      )
                      for method, value in prefix_reductions.items()
                  ))
            forbidden_showdown_claims = (
                "composed wins the showdown",
                "composed finishes with the lowest",
                "all showdown branches completed",
                "eight-step showdown win",
            )
            check("docs make no winner claim for the incomplete showdown",
                  not any(docs_contain(phrase) for phrase in forbidden_showdown_claims))

    # ---- 48-attempt robustness matrix ----------------------------------
    matrix = load("intervention_robustness_matrix_48.json")
    if matrix:
        protocol_path = ROOT / "orchestrator" / "protocols" / "intervention_robustness_matrix_48.json"
        protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        summary = matrix["summary"]
        outcomes = summary["outcomes"]
        check("robustness matrix uses the exact committed protocol bytes",
              matrix["protocol_sha256"] == protocol_hash)
        check("all 48 matrix attempts have valid durable records",
              summary["study_complete"] is True
              and summary["attempts_planned"] == 48
              and summary["attempts_recorded"] == 48
              and summary["attempts_pending"] == 0
              and summary["invalid_attempt_record_count"] == 0
              and sum(outcomes.values()) == 48)
        check("every Rayleigh stratum accounts for sixteen attempts",
              len(matrix["by_rayleigh_number"]) == 3
              and all(row["attempts_planned"] == 16
                      and row["attempts_recorded"] == 16
                      and row["accounting_complete"]
                      for row in matrix["by_rayleigh_number"]))
        strata = matrix["by_prior_observation_status"]
        check("matrix discloses exactly 13 prior-overlap and 35 previously unstored cells",
              strata["observed_before_frozen_design"]["attempts_planned"] == 13
              and strata["not_stored_before_frozen_design"]["attempts_planned"] == 35
              and bool(docs_contain("13")) and bool(docs_contain("35")))
        cluster = summary["cluster_aware_seed_analysis"]
        bootstrap = cluster["bootstrap"]
        check("cluster-aware analysis preserves all sixteen repeated-seed clusters",
              cluster["all_planned_clusters_complete"] is True
              and cluster["complete_clusters"] == 16
              and bootstrap["samples_with_comparable_cases"]
              == bootstrap["samples_requested"]
              and bootstrap["lower"] is not None
              and bootstrap["upper"] is not None)
        comparable = summary["comparable_cases"]
        wins = outcomes["exact_wins"]
        losses = outcomes["shortcut_wins"]
        ties = outcomes["tie"]
        noncomparable = summary["attempts_recorded"] - comparable
        shortcut_word = "shortcut win" if losses == 1 else "shortcut wins"
        print(f"matrix: {wins} exact wins, {losses} {shortcut_word}, {ties} ties, "
              f"{noncomparable} noncomparable; {comparable}/48 comparable")
        quoted_outcomes = (
            (wins, "exact win" if wins == 1 else "exact wins"),
            (losses, shortcut_word),
            (ties, "tie" if ties == 1 else "ties"),
            (noncomparable, "noncomparable"),
        )
        for value, label in quoted_outcomes:
            check(f"README/PAPER quote matrix {label}",
                  bool(docs_contain(f"{value} {label}")), f"measured {value}")
        cluster_lower = 100.0 * bootstrap["lower"]
        check("README/PAPER quote the seed-cluster bootstrap lower bound",
              bool(docs_contain(f"{cluster_lower:.1f}%")),
              f"measured {cluster_lower:.1f}%")

    # ---- nonlinear reference and explicit SI map -----------------------
    cavity = load("de_vahl_davis.json")
    if cavity:
        max_error = max(error for row in cavity for error in row["relative_error"].values())
        check("both de Vahl Davis cases have outer and inner convergence evidence",
              len(cavity) == 2
              and all(row["solver"]["ok"]
                      and row["solver"]["fluid"]["converged"]
                      and row["solver"]["fluid"]["relative_residual"] <= 1e-12
                      for row in cavity))
        check("all six N=32 cavity metrics are within the stated 15% tolerance",
              all(row["N"] == 32
                      and row["within_coarse_grid_tolerance"]
                      and max(row["relative_error"].values())
                      <= row["coarse_grid_tolerance"]
                      for row in cavity))
        check("README/PAPER quote the measured maximum cavity error",
              bool(docs_contain(f"{100*max_error:.1f}%")),
              f"measured {100*max_error:.2f}%")

    physical = load("dimensional_coldplate.json")
    if physical:
        layouts = physical["layouts"]
        mesh = physical["mesh_refinement"]
        mesh_layouts = [
            layout
            for row in mesh["rows"]
            for layout in row["layouts"].values()
        ]
        converged_mesh_solves = sum(
            bool(layout["solver"]["ok"]) for layout in mesh_layouts
        )
        check("dimensional chip discretization preserves exactly one watt",
              abs(physical["grid"]["represented_heat_load_W"] - 1.0) <= 1e-12)
        check("dimensional artifact retains its invalid comparison instead of promoting it",
              physical["evidence_valid"] is False
              and layouts["baseline"]["solver"]["ok"] is True
              and layouts["finned"]["solver"]["ok"] is False
              and mesh["all_solves_converged"] is False)
        check("every dimensional inner fluid solve exposes convergence evidence",
              all(layout["solver"]["fluid"]["converged"]
                      for layout in mesh_layouts))
        check("dimensional comparison is explicitly unequal-material and illustrative",
              physical["comparison"]["equal_material_budget"] is False
              and physical["comparison"]["kind"] == "illustrative_unequal_material"
              and bool(docs_contain("unequal-material")))
        check("dimensional mesh audit covers 16, 24, and 32 and retains all six outcomes",
              mesh["grids"] == [16, 24, 32]
              and mesh["finest_grid"] == 32
              and len(mesh_layouts) == 6
              and converged_mesh_solves == 3)
        check("README/PAPER disclose that only 3 of 6 dimensional solves converged",
              len(docs_contain("3 of 6")) == 2)
        check("README/PAPER disclose the dimensional constitutive-model failure",
              len(docs_contain("outside the constant-property liquid-water regime")) == 2)
        baseline_rth = layouts["baseline"]["thermal_resistance_K_W"]
        invalid_finned_rth = layouts["finned"]["thermal_resistance_K_W"]
        invalid_reduction = physical["finned_thermal_resistance_reduction_percent"]
        check("README/PAPER do not promote any invalid dimensional performance number",
              not docs_contain(f"{baseline_rth:.2f}")
              and not docs_contain(f"{invalid_finned_rth:.2f}")
              and not docs_contain(f"{invalid_reduction:.2f}%"))

    # ---- interchangeable thermal backends -------------------------------
    # The strongest single claim in the README, and until now the only headline
    # value with no stored artefact behind it. compare_thermal_backends.py now
    # records what it measured, so the prose can be checked like the rest.
    parity = load("thermal_backend_parity.json")
    if parity:
        print(f"backend parity: end-to-end {parity['end_to_end_gradient']:.3e}, "
              f"cosine {parity['gradient_cosine']:.12f}")
        check("both thermal backends reached a converged coupled state",
              parity["converged"] is True and parity["interchangeable"] is True)

        def as_prose(value: float) -> str:
            """Render a measurement the way the README writes one."""
            mantissa, exponent = f"{value:.1e}".split("e")
            digits = "⁰¹²³⁴⁵⁶⁷⁸⁹"
            rendered = "".join(digits[int(d)] for d in str(abs(int(exponent))))
            sign = "\u207b" if int(exponent) < 0 else ""
            return f"{mantissa} \u00d7 10{sign}{rendered}"

        for key in ("component_forward_T", "component_jvp", "component_vjp",
                    "coupled_state_T", "end_to_end_gradient"):
            expected = as_prose(parity[key])
            # The detail stays ASCII: this script runs on a Windows console
            # too, and a crash while reporting a passing check helps nobody.
            check(f"README/PAPER quote the measured {key}",
                  bool(docs_contain(expected)), f"prose must carry {parity[key]:.1e}")
        # A cosine printed to twelve places must actually be one to twelve places.
        check("README/PAPER quote a cosine that rounds to 1.000000000000",
              f"{parity['gradient_cosine']:.12f}" == "1.000000000000"
              and bool(docs_contain("1.000000000000")))
        check("the parity artefact is a real swap, not one backend run twice",
              parity["J_jax"] != parity["J_fortran"]
              and abs(parity["J_jax"] - parity["J_fortran"]) < 1e-9)

    # ---- randomized generalization study -------------------------------
    gg = load("gamma_generalization.json")
    if gg:
        o = gg["overall"]
        print(f"generalization: n={gg['trials_usable']} random systems, "
              f"log-gamma {o['log_gamma_correlation']:+.4f} vs rho "
              f"{o['rho_correlation']:+.4f}")
        check("README/PAPER quote 2,377 usable random systems",
              gg["trials_usable"] == 2377
              and (bool(docs_contain("2,377")) or bool(docs_contain("2377"))),
              f"measured {gg['trials_usable']}")
        check("pooled log-gamma correlation is the quoted 0.989",
              abs(o["log_gamma_correlation"] - 0.989) < 0.002
              and bool(docs_contain("0.989")),
              f"measured {o['log_gamma_correlation']:.4f}")
        check("pooled rho correlation is the quoted 0.691",
              abs(o["rho_correlation"] - 0.691) < 0.002
              and bool(docs_contain("0.691")),
              f"measured {o['rho_correlation']:.4f}")
        check("gamma beats rho in every structural family",
              all(b["log_gamma_correlation"] > b["rho_correlation"]
                  for b in gg["per_family"].values()))
        check("gamma beats rho for linear and nonlinear loops alike",
              all(b["log_gamma_correlation"] > b["rho_correlation"]
                  for b in gg["per_kind"].values()))
        safe = gg["safe_bucket"]
        check("no draw called SAFE hid an error above 5%",
              safe["frac_under_5pct"] == 1.0,
              f"n={safe['n']}, worst {100*safe['worst_rel_err']:.1f}%")
        check("docs quote the worst SAFE error as 1.4%",
              abs(100 * safe["worst_rel_err"] - 1.4) < 0.1,
              f"measured {100*safe['worst_rel_err']:.2f}%")
        check("every draw called UNSAFE genuinely exceeded 5%",
              gg["unsafe_bucket"]["frac_over_5pct"] == 1.0,
              f"n={gg['unsafe_bucket']['n']}")
        # The documented limitation must stay documented.
        check("the repelling-regime limitation is real and disclosed",
              gg["repelling"]["log_gamma_correlation"] < 0.75
              and bool(docs_contain("repelling")),
              f"repelling corr {gg['repelling']['log_gamma_correlation']:+.3f}")

    # ---- inertia: is the Stokes limit justified where it is used? -------
    rows = load("inertia_study.json")
    if rows:
        design = [r for r in rows if abs(r["Ra"] - 3.0e4) < 1 and r["rho_mean"] == 0.5]
        print(f"inertia study: {len(rows)} configurations solved twice")
        check("dropping inertia is negligible at the tested headline point",
              all(r["grad_rel_change"] < 1e-3 for r in design) and bool(design),
              "; ".join(f"Pr={r['Pr']}: {100*r['grad_rel_change']:.3f}%"
                        for r in design))
        check("the Stokes and Navier-Stokes gradients are collinear there",
              all(r["grad_cosine"] > 0.9999999 for r in design))
        for r in design:
            quoted = f"{100 * r['grad_rel_change']:.3f}%"
            check(f"docs quote the Pr={r['Pr']:g} gradient change as {quoted}",
                  bool(docs_contain(quoted)), f"measured {quoted}")

    # ---- rendered submission video -------------------------------------
    video = ROOT / "demo" / "coldplate_submission.mp4"
    captions = ROOT / "demo" / "coldplate_submission.en.srt"
    poster = ROOT / "demo" / "poster.png"
    video_manifest = ROOT / "demo" / "video_manifest.json"
    media = (video, captions, poster, video_manifest)
    if any(path.exists() for path in media):
        check("all four final video deliverables exist and are nonempty",
              all(path.is_file() and path.stat().st_size > 0 for path in media))
        if all(path.is_file() and path.stat().st_size > 0 for path in media):
            from validate_video import validate_release_video

            # The stream checks shell out to ffprobe, which a reviewer running
            # this script is not obliged to have installed. Missing ffprobe
            # means "cannot verify here", not "the claims are wrong": say so
            # and carry on, rather than ending a claim audit in a traceback on
            # someone else's laptop. CI installs ffmpeg, so the checks below do
            # run where the result is recorded.
            if shutil.which("ffprobe") is None:
                NOTE.append(
                    "ffprobe not on PATH: skipped video stream verification "
                    "(install ffmpeg to run it locally; CI always does)"
                )
                report = manifest = None
            else:
                report = validate_release_video(video, video_manifest, captions,
                                                poster)
                manifest = json.loads(video_manifest.read_text(encoding="utf-8"))
        else:
            report = manifest = None
        if report is not None:
            check("rendered video is a verified sub-five-minute 1080p delivery",
                  180.0 <= report["duration_seconds"] <= 300.0
                  and report["width"] == 1920 and report["height"] == 1080
                  and report["video_codec"] == "h264"
                  and report["audio_codec"] == "aac")
            check("video is small enough for an ordinary GitHub repository",
                  report["bytes"] < 100_000_000,
                  f"{report['bytes'] / 1_000_000:.1f} MB")
            check("video manifest records all eleven data-backed sections",
                  manifest["sections"] == 11)
        # Independent of ffprobe: this one is about the documents, so it must
        # keep running when the stream checks are skipped.
        check("README links the rendered MP4 and English captions",
              bool(docs_contain("demo/coldplate_submission.mp4", files=("README.md",)))
              and bool(docs_contain("demo/coldplate_submission.en.srt", files=("README.md",))))
    else:
        NOTE.append("final video deliverables not rendered yet")

    # ---- claims that must NOT appear ----------------------------------
    print("\n=== retracted claims must not reappear ===")
    for bad, why in [
        ("79% of", "artefact of comparing projected against raw gradients"),
    ]:
        hits = [f for f in ("README.md", "PAPER.md")
                if re.search(bad, (ROOT / f).read_text(encoding="utf-8"), re.I)]
        # the README keeps one deliberate mention inside its correction note
        allowed = bad == "79% of" and hits == ["README.md"]
        check(f"no live claim of '{bad}'", not hits or allowed,
              f"({why})" + (f" found in {hits}" if hits and not allowed else ""))

    print("\n=== stale artefacts and unsafe helpers must not return ===")
    source_files = [
        ROOT / "README.md", ROOT / "PAPER.md", ROOT / "DEMO_SCRIPT.md",
        ROOT / "coupling_check.py", ROOT / "orchestrator" / "make_figures.py",
        ROOT / "orchestrator" / "pipeline.py",
        ROOT / "orchestrator" / "predict_error.py",
        ROOT / "prototype" / "coupling_strength.py",
        ROOT / "prototype" / "gradient_check.py",
        ROOT / "prototype" / "reference_jax.py",
        ROOT / "prototype" / "validate_ra3e4.py",
        ROOT / "scripts" / "build_demo_video.py",
        ROOT / "orchestrator" / "make_extended_figures.py",
    ]
    joined = "\n".join(p.read_text(encoding="utf-8") for p in source_files)
    delivered_files = [
        ROOT / "README.md",
        ROOT / "PAPER.md",
        ROOT / "DEMO_SCRIPT.md",
        ROOT / "release" / "RELEASE_NOTES.md",
    ]
    delivered = "\n".join(path.read_text(encoding="utf-8") for path in delivered_files)
    for bad in (
        "three converged random designs",
        "Seventy-three component tests",
        "73 tests",
        "every command is public",
    ):
        check(f"no stale delivered-prose phrase '{bad}'", bad.lower() not in delivered.lower())
    check(
        "delivered prose never claims an equal material budget for the intervention",
        re.search(r"(?:same|identical|exactly\s+the\s+same)\s+material[- ]budget", delivered, re.I)
        is None,
    )
    for pattern, label in (
        (r"composed\s+(?:branch\s+)?wins\s+the\s+showdown", "showdown winner claim"),
        (r"wins?\s+the\s+(?:frozen\s+)?eight[- ]step", "eight-step winner claim"),
    ):
        check(f"no delivered {label}", re.search(pattern, delivered, re.I) is None)
    for bad in ("40-150%", "74% wrong sign", "four languages", "four containers"):
        check(f"no stale '{bad}' wording", bad.lower() not in joined.lower())
    for pattern, label in (
        (r"Picard\s+(?:iteration\s+)?cannot\s+converge", "absolute Picard impossibility"),
        (r"worst[-\s]+case\s+over\s+all\s+directions", "spectral-radius/norm confusion"),
        (r"provably\s+interchangeable", "unproved universal interchangeability"),
        (r"all\s+the\s+nonlinearity\s+lives\s+in\s+the\s+composition",
         "missing nonlinear maps"),
        (r"(?:reaching\s+the\s+same\s+design|design\s+(?:is|was)\s+unchanged|"
         r"design\s+came\s+out\s+indistinguishable)", "unmeasured layout equivalence"),
        (r"only\s+needs\s+(?:the\s+)?linearisation(?:\s+\w+){0,8}\s+invertible",
         "incomplete Newton condition"),
        (r"what\s+you\s+are\s+forced\s+to", "false derivative-method dichotomy"),
        (r"from\s+0\.01%\s+to\s+86%", "incorrect coupling-sweep minimum"),
        (r"orders\s+of\s+magnitude\s+below\s+the\s+viscous",
         "unmeasured term-norm claim"),
        (r"adjoint\s+exists\s+only\s+as", "false operator-assembly impossibility"),
        (r"\bpreregistered\b", "false prospective-preregistration claim"),
    ):
        check(f"no {label}", re.search(pattern, joined, re.I) is None)
    check("obsolete N48 trajectory diagnostic removed",
          not (ROOT / "orchestrator" / "results" / "history_diag_N48.json").exists())
    composed = load("history_composed_N96.json")
    stale_fields = {
        "naive_rel_err", "naive_cos", "naive_sign_flip", "loop_gain",
        "naive_cos_projected", "naive_sign_flip_projected",
    }
    if composed:
        check("optimization history contains no retracted diagnostic fields",
              all(not stale_fields.intersection(row) for row in composed))

    for name in ("validate_both_backends.sh", "run_optimisations.sh"):
        body = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        check(f"{name} never removes every running Docker container",
              "docker ps -q |" not in body and "ancestor=$image" in body)

    print()
    if NOTE:
        print("notes:")
        for n in NOTE:
            print(f"  - {n}")
    if FAIL:
        print(f"\n{len(FAIL)} CHECK(S) FAILED:")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print("every audited headline number matches the stored measurements")
    return 0


if __name__ == "__main__":
    sys.exit(main())
