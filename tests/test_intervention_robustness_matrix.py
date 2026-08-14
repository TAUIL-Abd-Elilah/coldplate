# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Fast tests for the resumable preregistered robustness matrix."""

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


PROTOCOL = ROOT / "orchestrator" / "protocols" / "intervention_robustness_matrix_48.json"


def test_protocol_expands_to_preregistered_48_attempts():
    protocol, _ = load_protocol(PROTOCOL)
    attempts = planned_attempts(protocol)
    assert len(attempts) == 48
    assert {a["N"] for a in attempts} == {20}
    assert {a["amplitude"] for a in attempts} == {0.025}
    assert {a["Ra"] for a in attempts} == {1.0e4, 2.0e4, 3.0e4}
    assert all(sum(a["Ra"] == ra for a in attempts) == 16 for ra in (1.0e4, 2.0e4, 3.0e4))


def test_wilson_interval_known_extremes_and_empty():
    empty = wilson_interval(0, 0, 1.959963984540054)
    assert empty["estimate"] is None
    assert wilson_interval(0, 10, 1.959963984540054)["upper"] == pytest.approx(0.2775328)
    assert wilson_interval(10, 10, 1.959963984540054)["lower"] == pytest.approx(0.7224672)


def _small_protocol(tmp_path: Path) -> Path:
    protocol = json.loads(PROTOCOL.read_text())
    protocol["study_id"] = "test-matrix"
    protocol["design"]["rayleigh_numbers"] = [10000.0]
    protocol["design"]["seeds"] = [0, 1, 2, 3]
    protocol["design"]["attempts_planned"] = 4
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return path


def _report(path: Path, attempts: list[dict], outcomes=None, **overrides) -> Path:
    outcomes = outcomes or ["exact_wins"] * len(attempts)
    payload = {
        "N": attempts[0]["N"],
        "Ra": attempts[0]["Ra"],
        "amplitude": attempts[0]["amplitude"],
        "seeds_planned": len(attempts),
        "complete": True,
        "cases": [
            {"seed": attempt["seed"], "outcome": outcome}
            for attempt, outcome in zip(attempts, outcomes)
        ],
        **overrides,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_run_is_resumable_and_accounts_for_every_failure_class(tmp_path):
    protocol_path = _small_protocol(tmp_path)
    calls = []
    outcomes = ["exact_wins", "shortcut_wins", "base_not_converged", "inconclusive"]

    def fake_executor(attempt, raw_out, timeout):
        calls.append(attempt["seed"])
        case = {"seed": attempt["seed"], "outcome": outcomes[attempt["seed"]]}
        raw_out.write_text(json.dumps({
            "N": attempt["N"], "Ra": attempt["Ra"], "amplitude": attempt["amplitude"],
            "cases": [case],
        }))
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
        raw_out.write_text(json.dumps({
            "N": attempt["N"], "Ra": attempt["Ra"], "amplitude": attempt["amplitude"],
            "cases": [{"seed": attempt["seed"], "outcome": "exact_wins"}],
        }))
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
