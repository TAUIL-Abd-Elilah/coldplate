# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Fast tests for the resumable retrospectively frozen robustness matrix."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "orchestrator"))

from intervention_robustness_matrix import (  # noqa: E402
    aggregate,
    ingest_report,
    load_protocol,
    planned_attempts,
    run_matrix,
    wilson_interval,
)
from intervention_robustness import (  # noqa: E402
    classify_outcome,
    summarize as summarize_execution_unit,
)


PROTOCOL = ROOT / "orchestrator" / "protocols" / "intervention_robustness_matrix_48.json"


def test_protocol_expands_to_frozen_48_attempts_and_discloses_overlap():
    protocol, _ = load_protocol(PROTOCOL)
    attempts = planned_attempts(protocol)
    assert len(attempts) == 48
    assert {a["N"] for a in attempts} == {20}
    assert {a["amplitude"] for a in attempts} == {0.025}
    assert {a["Ra"] for a in attempts} == {1.0e4, 2.0e4, 3.0e4}
    assert all(sum(a["Ra"] == ra for a in attempts) == 16 for ra in (1.0e4, 2.0e4, 3.0e4))
    assert protocol["status"] == "retrospectively_frozen_design"
    disclosure = protocol["prior_observation_disclosure"]
    assert disclosure["overlap_fraction"] == "13/48"
    assert disclosure["observed_cells_count"] == 13
    assert disclosure["not_previously_stored_cells_count"] == 35
    assert disclosure["observed_cells"][-1] == {"Ra": 30000.0, "seeds": [0]}
    assert "post-freeze" in " ".join(protocol["integrity_amendments_after_freeze"])


def test_wilson_interval_known_extremes_and_empty():
    empty = wilson_interval(0, 0, 1.959963984540054)
    assert empty["estimate"] is None
    assert wilson_interval(0, 10, 1.959963984540054)["upper"] == pytest.approx(0.2775328)
    assert wilson_interval(10, 10, 1.959963984540054)["lower"] == pytest.approx(0.7224672)


def _small_protocol(tmp_path: Path, *, rayleigh_numbers=None, seeds=None) -> Path:
    protocol = json.loads(PROTOCOL.read_text())
    protocol["study_id"] = "test-matrix"
    protocol["design"]["rayleigh_numbers"] = rayleigh_numbers or [10000.0]
    protocol["design"]["seeds"] = seeds or [0, 1, 2, 3]
    protocol["design"]["attempts_planned"] = (
        len(protocol["design"]["rayleigh_numbers"])
        * len(protocol["design"]["seeds"])
    )
    planned = protocol["design"]["attempts_planned"]
    protocol["prior_observation_disclosure"]["observed_cells"] = [{
        "Ra": protocol["design"]["rayleigh_numbers"][0],
        "seeds": [protocol["design"]["seeds"][0]],
    }]
    protocol["prior_observation_disclosure"]["observed_cells_count"] = 1
    protocol["prior_observation_disclosure"]["not_previously_stored_cells_count"] = planned - 1
    protocol["prior_observation_disclosure"]["overlap_fraction"] = f"1/{planned}"
    protocol["analysis"]["cluster_aware_secondary"]["bootstrap_samples"] = 200
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return path


def _case(attempt: dict, outcome: str) -> dict:
    common = {
        "seed": attempt["seed"],
        "gamma": 0.4,
        "naive_relative_error": 0.2,
        "naive_cosine": 0.9,
        "add_set_overlap": 0.5,
        "remove_set_overlap": 0.4,
    }
    if outcome == "base_not_converged":
        return {
            "seed": attempt["seed"], "outcome": outcome,
            "failure_stage": "base_forward", "reason": "test base failure",
        }
    if outcome == "runner_failure":
        return {
            "seed": attempt["seed"], "outcome": outcome,
            "failure_stage": "execution_unit_exception", "reason": "test runner failure",
        }
    if outcome == "inconclusive":
        return {
            **common,
            "outcome": outcome,
            "exact_action_ok": True,
            "naive_action_ok": False,
            "exact_action_reason": None,
            "naive_action_reason": "test candidate did not converge",
            "delta_J_exact_action": -0.03,
            "delta_J_naive_action": None,
        }
    changes = {
        "exact_wins": (-0.04, -0.02),
        "shortcut_wins": (-0.02, -0.04),
        "tie": (-0.03, -0.03),
    }
    exact, shortcut = changes[outcome]
    derived, advantage, tolerance = classify_outcome(
        exact,
        shortcut,
        attempt["outcome_absolute_tolerance"],
        attempt["outcome_relative_tolerance"],
    )
    assert derived == outcome
    return {
        **common,
        "outcome": outcome,
        "execution_unit_reported_outcome": outcome,
        "delta_J_exact_action": exact,
        "delta_J_naive_action": shortcut,
        "exact_advantage": advantage,
        "outcome_equivalence_tolerance": tolerance,
        "extra_cooling_fraction": abs(exact) / abs(shortcut) - 1.0,
    }


def _report_payload(attempts: list[dict], outcomes=None, **overrides) -> dict:
    outcomes = outcomes or ["exact_wins"] * len(attempts)
    cases = [_case(attempt, outcome) for attempt, outcome in zip(attempts, outcomes)]
    payload = summarize_execution_unit(
        cases, attempts[0]["N"], attempts[0]["Ra"], attempts[0]["amplitude"]
    )
    payload.update({
        "seeds_planned": len(attempts),
        "complete": True,
        "outcome_equivalence_tolerance": {
            "absolute_delta_J": attempts[0]["outcome_absolute_tolerance"],
            "relative_delta_J": attempts[0]["outcome_relative_tolerance"],
        },
    })
    payload.update(overrides)
    return payload


def _report(path: Path, attempts: list[dict], outcomes=None, **overrides) -> Path:
    payload = _report_payload(attempts, outcomes, **overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_run_is_resumable_and_accounts_for_every_failure_class(tmp_path):
    protocol_path = _small_protocol(tmp_path)
    calls = []
    outcomes = ["exact_wins", "shortcut_wins", "base_not_converged", "inconclusive"]

    def fake_executor(attempt, raw_out, timeout):
        calls.append(attempt["seed"])
        raw_out.write_text(json.dumps(_report_payload(
            [attempt], [outcomes[attempt["seed"]]]
        )))
        return subprocess.CompletedProcess([], 0 if attempt["seed"] < 2 else 1)

    attempt_dir, out = tmp_path / "attempts", tmp_path / "summary.json"
    first = run_matrix(protocol_path, out, attempt_dir, executor=fake_executor)
    second = run_matrix(protocol_path, out, attempt_dir, executor=fake_executor)
    assert calls == [0, 1, 2, 3]
    assert first == second == json.loads(out.read_text())
    summary = first["summary"]
    assert summary["study_complete"]
    assert summary["accounting_complete"]
    assert summary["outcomes"]["exact_wins"] == 1
    assert summary["outcomes"]["shortcut_wins"] == 1
    assert summary["outcomes"]["base_not_converged"] == 1
    assert summary["outcomes"]["inconclusive"] == 1
    assert summary["exact_win_rate_over_comparable"]["estimate"] == pytest.approx(0.5)
    assert summary["exact_win_rate_over_attempted"]["estimate"] == pytest.approx(0.25)


def test_missing_report_is_a_durable_runner_failure(tmp_path):
    protocol_path = _small_protocol(tmp_path)
    calls = 0

    def no_report(attempt, raw_out, timeout):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess([], 7)

    result = run_matrix(protocol_path, tmp_path / "out.json", tmp_path / "attempts",
                        executor=no_report)
    assert calls == 4
    assert result["summary"]["outcomes"]["runner_failure"] == 4
    assert result["summary"]["study_complete"]
    assert all(record["reason"] for record in result["attempts"])
    assert all(record["timed_out"] is False for record in result["attempts"])


def test_timeout_is_durable_and_is_not_retried(tmp_path):
    protocol_path = _small_protocol(tmp_path, seeds=[0])
    calls = 0

    def timeout_executor(attempt, raw_out, timeout):
        nonlocal calls
        calls += 1
        raw_out.write_text("partial staging evidence", encoding="utf-8")
        raise subprocess.TimeoutExpired(cmd="test-execution-unit", timeout=timeout)

    attempt_dir = tmp_path / "attempts"
    first = run_matrix(protocol_path, tmp_path / "out.json", attempt_dir,
                       executor=timeout_executor)
    second = run_matrix(protocol_path, tmp_path / "out.json", attempt_dir,
                        executor=timeout_executor)
    assert calls == 1
    assert first == second
    [record] = first["attempts"]
    assert record["outcome"] == "runner_failure"
    assert record["failure_stage"] == "timeout"
    assert record["timed_out"] is True
    assert record["attempt_timeout_seconds"] > 0
    assert "configured" in record["reason"]
    assert record["staging_report_present"] is True
    assert record["staging_report_bytes"] == len("partial staging evidence")
    assert len(record["staging_report_sha256"]) == 64
    assert not list(attempt_dir.glob(".*.raw.json"))


def test_saved_case_is_recomputed_on_every_read_and_tampering_is_not_rerun(tmp_path):
    protocol_path = _small_protocol(tmp_path, seeds=[0])
    attempt_dir = tmp_path / "attempts"

    def exact_executor(attempt, raw_out, timeout):
        raw_out.write_text(json.dumps(_report_payload([attempt])), encoding="utf-8")
        return subprocess.CompletedProcess([], 0)

    result = run_matrix(protocol_path, tmp_path / "first.json", attempt_dir,
                        executor=exact_executor)
    record_path = next(attempt_dir.glob("*.json"))
    tampered = json.loads(record_path.read_text())
    tampered["case"]["delta_J_exact_action"] = -0.001
    record_path.write_text(json.dumps(tampered), encoding="utf-8")

    calls = 0

    def must_not_replace(*args):
        nonlocal calls
        calls += 1
        raise AssertionError("invalid durable evidence must not be replaced")

    checked = run_matrix(protocol_path, tmp_path / "checked.json", attempt_dir,
                         executor=must_not_replace)
    assert result["summary"]["study_complete"]
    assert calls == 0
    assert checked["summary"]["attempts_recorded"] == 0
    assert checked["summary"]["invalid_attempt_record_count"] == 1
    assert checked["summary"]["invalid_attempt_records"][0]["error"] == (
        "record_invalid_case_evidence"
    )


def test_aggregate_surfaces_invalid_record_without_rerunning(tmp_path):
    protocol_path = _small_protocol(tmp_path)
    protocol, digest = load_protocol(protocol_path)
    attempts = planned_attempts(protocol)
    attempt_dir = tmp_path / "attempts"
    attempt_dir.mkdir()
    bad = attempt_dir / f"{attempts[0]['attempt_id']}.json"
    bad.write_text("not json")
    calls = 0

    def should_not_replace_bad_record(attempt, raw_out, timeout):
        nonlocal calls
        calls += 1
        raw_out.write_text(json.dumps(_report_payload([attempt])))
        return subprocess.CompletedProcess([], 0)

    result = run_matrix(protocol_path, tmp_path / "out.json", attempt_dir,
                        executor=should_not_replace_bad_record)
    assert calls == 3
    assert bad.read_text() == "not json"
    assert result["summary"]["attempts_recorded"] == 3
    assert result["summary"]["attempts_pending"] == 0
    assert result["summary"]["invalid_attempt_record_count"] == 1
    assert result["summary"]["invalid_attempt_records"] == [
        {"attempt_id": attempts[0]["attempt_id"], "error": "unreadable_json"}
    ]
    assert not result["summary"]["study_complete"]
    assert aggregate(protocol, digest, attempt_dir) == result


def test_ingest_report_writes_durable_records_and_never_executes(tmp_path):
    protocol_path = _small_protocol(tmp_path)
    protocol, _ = load_protocol(protocol_path)
    attempts = planned_attempts(protocol)
    report = _report(
        tmp_path / "report.json",
        attempts,
        ["exact_wins", "shortcut_wins", "base_not_converged", "inconclusive"],
    )

    def must_not_execute(*args):
        raise AssertionError("ingestion must not launch an attempt")

    result = run_matrix(
        protocol_path,
        tmp_path / "out.json",
        tmp_path / "attempts",
        executor=must_not_execute,
        ingest_reports=[report],
    )
    assert result["summary"]["study_complete"]
    assert result["summary"]["outcomes"]["exact_wins"] == 1
    records = result["attempts"]
    assert [record["seed"] for record in records] == [0, 1, 2, 3]
    assert all(record["source"] == "ingested_report" for record in records)
    assert len({record["source_report_sha256"] for record in records}) == 1


def test_ingest_recomputes_outcome_instead_of_trusting_case_label(tmp_path):
    protocol_path = _small_protocol(tmp_path, seeds=[0])
    protocol, digest = load_protocol(protocol_path)
    [attempt] = planned_attempts(protocol)
    report = _report(tmp_path / "reported.json", [attempt])
    payload = json.loads(report.read_text())
    payload["cases"][0]["outcome"] = "shortcut_wins"
    report.write_text(json.dumps(payload), encoding="utf-8")

    counts = ingest_report(report, protocol, digest, tmp_path / "attempts")
    assert counts["written"] == 1
    result = aggregate(protocol, digest, tmp_path / "attempts")
    [record] = result["attempts"]
    assert record["outcome"] == "exact_wins"
    assert record["case"]["matrix_input_reported_outcome"] == "shortcut_wins"
    assert record["case"]["execution_unit_reported_outcome"] == "exact_wins"


def test_repeatable_ingest_accepts_locked_16_seed_ra_slices(tmp_path):
    protocol, _ = load_protocol(PROTOCOL)
    attempts = planned_attempts(protocol)
    reports = []
    for ra in (1.0e4, 2.0e4):
        ra_attempts = [attempt for attempt in attempts if attempt["Ra"] == ra]
        assert len(ra_attempts) == 16
        reports.append(_report(tmp_path / f"Ra-{ra:.0f}.json", ra_attempts))

    def must_not_execute(*args):
        raise AssertionError("repeatable ingestion must not launch an attempt")

    result = run_matrix(
        PROTOCOL,
        tmp_path / "out.json",
        tmp_path / "attempts",
        executor=must_not_execute,
        ingest_reports=reports,
    )
    assert result["summary"]["attempts_recorded"] == 32
    assert result["summary"]["attempts_pending"] == 16
    assert not result["summary"]["study_complete"]
    assert [group["attempts_recorded"] for group in result["by_rayleigh_number"]] == [16, 16, 0]


def test_cluster_aware_secondary_resamples_complete_seed_clusters(tmp_path):
    protocol_path = _small_protocol(
        tmp_path, rayleigh_numbers=[10000.0, 20000.0, 30000.0], seeds=[0, 1]
    )

    def paired_executor(attempt, raw_out, timeout):
        outcome = "exact_wins" if attempt["seed"] == 0 else "shortcut_wins"
        raw_out.write_text(
            json.dumps(_report_payload([attempt], [outcome])), encoding="utf-8"
        )
        return subprocess.CompletedProcess([], 0)

    result = run_matrix(protocol_path, tmp_path / "out.json", tmp_path / "attempts",
                        executor=paired_executor)
    cluster = result["summary"]["cluster_aware_seed_analysis"]
    assert cluster["complete_clusters"] == 2
    assert cluster["incomplete_clusters"] == 0
    assert cluster["all_planned_clusters_complete"] is True
    assert cluster["pooled_exact_win_rate_over_comparable_in_complete_clusters"] == 0.5
    assert cluster["paired_seed_direction_counts"] == {
        "exact_dominant": 1,
        "shortcut_dominant": 1,
        "balanced": 0,
        "no_comparable_cases": 0,
    }
    assert cluster["bootstrap"]["samples_requested"] == 200
    assert cluster["bootstrap"]["samples_with_comparable_cases"] == 200
    assert cluster["bootstrap"]["lower"] == 0.0
    assert cluster["bootstrap"]["upper"] == 1.0


def test_prior_observation_strata_cover_exactly_13_and_35_planned_cells(tmp_path):
    protocol, digest = load_protocol(PROTOCOL)
    result = aggregate(protocol, digest, tmp_path / "no-attempts-yet")
    strata = result["by_prior_observation_status"]
    assert "added after" in strata["analysis_timing"]
    assert strata["observed_before_frozen_design"]["attempts_planned"] == 13
    assert strata["not_stored_before_frozen_design"]["attempts_planned"] == 35
    assert (
        strata["observed_before_frozen_design"]["attempts_planned"]
        + strata["not_stored_before_frozen_design"]["attempts_planned"]
        == 48
    )


def test_ingest_does_not_replace_valid_or_invalid_existing_records(tmp_path):
    protocol_path = _small_protocol(tmp_path)
    protocol, digest = load_protocol(protocol_path)
    attempts = planned_attempts(protocol)
    attempt_dir = tmp_path / "attempts"
    first_report = _report(tmp_path / "first.json", attempts)
    assert ingest_report(first_report, protocol, digest, attempt_dir)["written"] == 4

    valid_path = attempt_dir / f"{attempts[0]['attempt_id']}.json"
    valid_before = valid_path.read_bytes()
    invalid_path = attempt_dir / f"{attempts[1]['attempt_id']}.json"
    invalid_path.write_text("invalid durable record", encoding="utf-8")
    invalid_before = invalid_path.read_bytes()
    replacement = _report(
        tmp_path / "replacement.json", attempts, ["shortcut_wins"] * len(attempts)
    )
    counts = ingest_report(replacement, protocol, digest, attempt_dir)
    assert counts == {"written": 0, "existing_valid": 3, "existing_invalid": 1}
    assert valid_path.read_bytes() == valid_before
    assert invalid_path.read_bytes() == invalid_before


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(N=21),
        lambda payload: payload.update(Ra=99999.0),
        lambda payload: payload.update(amplitude=0.05),
        lambda payload: payload["cases"].append({"seed": 99, "outcome": "exact_wins"}),
        lambda payload: payload["cases"].__setitem__(0, {"seed": 0, "outcome": "unknown"}),
        lambda payload: payload["cases"].pop(),
    ],
)
def test_ingest_rejects_invalid_report_before_writing(tmp_path, mutation):
    protocol_path = _small_protocol(tmp_path)
    protocol, digest = load_protocol(protocol_path)
    attempts = planned_attempts(protocol)
    report = _report(tmp_path / "bad.json", attempts)
    payload = json.loads(report.read_text())
    mutation(payload)
    report.write_text(json.dumps(payload), encoding="utf-8")
    attempt_dir = tmp_path / "attempts"
    with pytest.raises(ValueError):
        ingest_report(report, protocol, digest, attempt_dir)
    assert not attempt_dir.exists()
