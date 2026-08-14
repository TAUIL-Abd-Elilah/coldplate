# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Run and aggregate the preregistered 48-case robustness matrix.

Each ``(Ra, seed)`` invocation of :mod:`intervention_robustness` gets an atomic,
durable record. Re-running this script skips valid records, making interruption
and resume safe without silently replacing an inconvenient solver outcome.

Usage::

    python intervention_robustness_matrix.py
    python intervention_robustness_matrix.py --aggregate-only
    python intervention_robustness_matrix.py --ingest-report results/ra-10000.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable


HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / "protocols" / "intervention_robustness_matrix_48.json"
DEFAULT_OUT = HERE / "results" / "intervention_robustness_matrix_48.json"
DEFAULT_ATTEMPT_DIR = HERE / "results" / "intervention_robustness_matrix_48_attempts"
OUTCOMES = {"exact_wins", "shortcut_wins", "tie", "inconclusive", "base_not_converged"}
COMPARABLE = {"exact_wins", "shortcut_wins", "tie"}


def load_protocol(path: Path) -> tuple[dict, str]:
    """Load and validate the study protocol, returning its SHA-256 digest."""
    raw = path.read_bytes()
    protocol = json.loads(raw)
    design = protocol["design"]
    ras, seeds = design["rayleigh_numbers"], design["seeds"]
    expected = len(ras) * len(seeds)
    if design["attempts_planned"] != expected:
        raise ValueError("attempts_planned does not match the design matrix")
    if len(set(ras)) != len(ras) or len(set(seeds)) != len(seeds):
        raise ValueError("Rayleigh numbers and seeds must be unique")
    if design["N"] <= 0 or design["amplitude"] <= 0:
        raise ValueError("N and amplitude must be positive")
    z = protocol["analysis"]["z"]
    if not isinstance(z, (int, float)) or z <= 0:
        raise ValueError("analysis.z must be positive")
    return protocol, hashlib.sha256(raw).hexdigest()


def planned_attempts(protocol: dict) -> list[dict]:
    """Expand the matrix in its preregistered deterministic order."""
    design = protocol["design"]
    return [
        {
            "attempt_id": f"Ra{float(ra):012.3f}_seed{int(seed):03d}",
            "N": int(design["N"]),
            "Ra": float(ra),
            "amplitude": float(design["amplitude"]),
            "seed": int(seed),
        }
        for ra in design["rayleigh_numbers"]
        for seed in design["seeds"]
    ]


def atomic_json_write(path: Path, value: dict) -> None:
    """Replace a JSON checkpoint atomically and with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def wilson_interval(successes: int, trials: int, z: float) -> dict:
    """Return a deterministic Wilson score estimate and interval."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= trials")
    if trials == 0:
        return {"successes": successes, "trials": trials, "estimate": None,
                "lower": None, "upper": None}
    p = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    centre = (p + z2 / (2 * trials)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / denominator
    return {"successes": successes, "trials": trials, "estimate": p,
            "lower": max(0.0, centre - half_width),
            "upper": min(1.0, centre + half_width)}


def _matches_attempt(record: dict, attempt: dict, protocol_hash: str) -> bool:
    return (
        record.get("schema_version") == 1
        and record.get("protocol_sha256") == protocol_hash
        and all(record.get(key) == value for key, value in attempt.items())
        and record.get("outcome") in OUTCOMES | {"runner_failure"}
    )


def read_record(path: Path, attempt: dict, protocol_hash: str) -> tuple[dict | None, str | None]:
    """Read a record without allowing a corrupt/mismatched record to be rerun."""
    if not path.exists():
        return None, None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "unreadable_json"
    if not isinstance(record, dict) or not _matches_attempt(record, attempt, protocol_hash):
        return None, "record_does_not_match_protocol_attempt"
    return record, None


def execute_attempt(attempt: dict, raw_out: Path, timeout: float) -> subprocess.CompletedProcess:
    """Invoke the existing audited robustness script for exactly one seed."""
    command = [
        sys.executable,
        str(HERE / "intervention_robustness.py"),
        "--N", str(attempt["N"]),
        "--Ra", str(attempt["Ra"]),
        "--seed-start", str(attempt["seed"]),
        "--n-seeds", "1",
        "--amplitude", str(attempt["amplitude"]),
        "--out", str(raw_out),
    ]
    return subprocess.run(
        command,
        cwd=HERE,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def normalize_report(attempt: dict, protocol_hash: str, raw_out: Path,
                     returncode: int) -> dict:
    """Convert the execution-unit report into one matrix attempt record."""
    base = {"schema_version": 1, "protocol_sha256": protocol_hash, **attempt,
            "execution_returncode": returncode}
    try:
        report = json.loads(raw_out.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {**base, "outcome": "runner_failure", "failure_stage": "missing_report"}
    except (OSError, json.JSONDecodeError):
        return {**base, "outcome": "runner_failure", "failure_stage": "invalid_report_json"}

    if (report.get("N") != attempt["N"] or report.get("Ra") != attempt["Ra"]
            or report.get("amplitude") != attempt["amplitude"]):
        return {**base, "outcome": "runner_failure", "failure_stage": "report_design_mismatch"}
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != 1 or cases[0].get("seed") != attempt["seed"]:
        return {**base, "outcome": "runner_failure", "failure_stage": "report_case_mismatch"}
    case = cases[0]
    if case.get("outcome") not in OUTCOMES:
        return {**base, "outcome": "runner_failure", "failure_stage": "unknown_outcome"}
    return {**base, "outcome": case["outcome"], "case": case}


def failure_record(attempt: dict, protocol_hash: str, stage: str) -> dict:
    """Make a terminal, auditable orchestration-failure record."""
    return {"schema_version": 1, "protocol_sha256": protocol_hash, **attempt,
            "execution_returncode": None, "outcome": "runner_failure",
            "failure_stage": stage}


def _validated_ingest_cases(report_path: Path, protocol: dict) -> tuple[list[tuple[dict, dict]], str]:
    """Validate a complete one-Ra execution-unit report before any writes."""
    try:
        raw = report_path.read_bytes()
        report = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read ingest report {report_path}: {type(exc).__name__}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"ingest report {report_path} must contain a JSON object")

    design = protocol["design"]
    if report.get("N") != int(design["N"]):
        raise ValueError(f"ingest report {report_path} has the wrong N")
    if report.get("amplitude") != float(design["amplitude"]):
        raise ValueError(f"ingest report {report_path} has the wrong amplitude")
    allowed_ras = {float(ra) for ra in design["rayleigh_numbers"]}
    try:
        ra = float(report["Ra"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"ingest report {report_path} has an invalid Ra") from exc
    if ra not in allowed_ras:
        raise ValueError(f"ingest report {report_path} has a non-preregistered Ra")

    attempts = [attempt for attempt in planned_attempts(protocol) if attempt["Ra"] == ra]
    by_seed = {attempt["seed"]: attempt for attempt in attempts}
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"ingest report {report_path} has no cases list")
    seen: dict[int, dict] = {}
    for case in cases:
        if not isinstance(case, dict) or type(case.get("seed")) is not int:
            raise ValueError(f"ingest report {report_path} contains an invalid seed")
        seed = case["seed"]
        if seed not in by_seed:
            raise ValueError(f"ingest report {report_path} contains non-preregistered seed {seed}")
        if seed in seen:
            raise ValueError(f"ingest report {report_path} contains duplicate seed {seed}")
        if case.get("outcome") not in OUTCOMES:
            raise ValueError(f"ingest report {report_path} contains an invalid outcome")
        seen[seed] = case
    if set(seen) != set(by_seed):
        missing = sorted(set(by_seed) - set(seen))
        raise ValueError(f"ingest report {report_path} is missing seeds {missing}")
    if report.get("seeds_planned") != len(attempts):
        raise ValueError(f"ingest report {report_path} has the wrong seeds_planned")
    if report.get("complete") is not True:
        raise ValueError(f"ingest report {report_path} is not complete")

    ordered = [(attempt, seen[attempt["seed"]]) for attempt in attempts]
    return ordered, hashlib.sha256(raw).hexdigest()


def ingest_report(report_path: Path, protocol: dict, protocol_hash: str,
                  attempt_dir: Path) -> dict:
    """Ingest one complete Ra slice without replacing any existing record."""
    cases, report_hash = _validated_ingest_cases(report_path, protocol)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    counts = {"written": 0, "existing_valid": 0, "existing_invalid": 0}
    for attempt, case in cases:
        record_path = attempt_dir / f"{attempt['attempt_id']}.json"
        existing, error = read_record(record_path, attempt, protocol_hash)
        if existing is not None:
            counts["existing_valid"] += 1
            continue
        if error is not None:
            counts["existing_invalid"] += 1
            continue
        record = {
            "schema_version": 1,
            "protocol_sha256": protocol_hash,
            **attempt,
            "execution_returncode": None,
            "outcome": case["outcome"],
            "case": case,
            "source": "ingested_report",
            "source_report_sha256": report_hash,
        }
        atomic_json_write(record_path, record)
        counts["written"] += 1
    return counts


def _group_summary(attempts: list[dict], records: list[dict], z: float) -> dict:
    outcomes = {name: 0 for name in sorted(OUTCOMES | {"runner_failure"})}
    for record in records:
        outcomes[record["outcome"]] += 1
    comparable = sum(outcomes[name] for name in COMPARABLE)
    exact_wins = outcomes["exact_wins"]
    attempted = len(records)
    accounted = comparable + outcomes["inconclusive"] + outcomes["base_not_converged"] + outcomes["runner_failure"]
    return {
        "attempts_planned": len(attempts),
        "attempts_recorded": attempted,
        "attempts_pending": len(attempts) - attempted,
        "outcomes": outcomes,
        "comparable_cases": comparable,
        "accounting_complete": accounted == attempted,
        "exact_win_rate_over_comparable": wilson_interval(exact_wins, comparable, z),
        "exact_win_rate_over_attempted": wilson_interval(exact_wins, attempted, z),
        "comparability_rate_over_attempted": wilson_interval(comparable, attempted, z),
    }


def aggregate(protocol: dict, protocol_hash: str, attempt_dir: Path) -> dict:
    """Aggregate valid records, exposing missing and invalid records explicitly."""
    attempts = planned_attempts(protocol)
    records, invalid = [], []
    by_id: dict[str, dict] = {}
    for attempt in attempts:
        record, error = read_record(attempt_dir / f"{attempt['attempt_id']}.json", attempt, protocol_hash)
        if record is not None:
            records.append(record)
            by_id[attempt["attempt_id"]] = record
        elif error is not None:
            invalid.append({"attempt_id": attempt["attempt_id"], "error": error})
    z = float(protocol["analysis"]["z"])
    by_ra = []
    for ra in protocol["design"]["rayleigh_numbers"]:
        group_attempts = [a for a in attempts if a["Ra"] == float(ra)]
        group_records = [by_id[a["attempt_id"]] for a in group_attempts if a["attempt_id"] in by_id]
        by_ra.append({"Ra": float(ra), **_group_summary(group_attempts, group_records, z)})
    summary = _group_summary(attempts, records, z)
    summary["invalid_attempt_records"] = invalid
    summary["invalid_attempt_record_count"] = len(invalid)
    summary["attempts_pending"] -= len(invalid)
    summary["study_complete"] = (
        summary["attempts_recorded"] == summary["attempts_planned"] and not invalid
    )
    return {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "protocol_sha256": protocol_hash,
        "confidence_level": protocol["analysis"]["confidence_level"],
        "summary": summary,
        "by_rayleigh_number": by_ra,
        "attempts": records,
    }


Executor = Callable[[dict, Path, float], subprocess.CompletedProcess]


def run_matrix(protocol_path: Path = DEFAULT_PROTOCOL, out: Path = DEFAULT_OUT,
               attempt_dir: Path = DEFAULT_ATTEMPT_DIR, aggregate_only: bool = False,
               executor: Executor = execute_attempt,
               ingest_reports: list[Path] | None = None) -> dict:
    """Run missing attempts, checkpoint after each one, and return aggregation."""
    protocol, digest = load_protocol(protocol_path)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    timeout = float(protocol["execution"]["attempt_timeout_seconds"])
    ingest_reports = ingest_reports or []

    for report_path in ingest_reports:
        ingest_report(report_path, protocol, digest, attempt_dir)
        atomic_json_write(out, aggregate(protocol, digest, attempt_dir))

    # Ingestion is deliberately non-executing. This lets CI ingest one or more
    # already-produced Ra slices without unexpectedly starting missing cases.
    if not aggregate_only and not ingest_reports:
        for attempt in planned_attempts(protocol):
            record_path = attempt_dir / f"{attempt['attempt_id']}.json"
            record, error = read_record(record_path, attempt, digest)
            if record is not None:
                continue
            if error is not None:
                # Never erase or rerun an invalid durable record: that could
                # replace an observed result. Surface it for manual inspection.
                continue
            raw_out = attempt_dir / f".{attempt['attempt_id']}.raw.json"
            # A raw report is only staging, never a completed record. Remove a
            # remnant from an interrupted invocation so it cannot be mistaken
            # for the output of the new subprocess.
            try:
                raw_out.unlink()
            except FileNotFoundError:
                pass
            try:
                completed = executor(attempt, raw_out, timeout)
                record = normalize_report(attempt, digest, raw_out, completed.returncode)
            except subprocess.TimeoutExpired:
                record = failure_record(attempt, digest, "timeout")
            except Exception:  # noqa: BLE001 - orchestration failure is study data
                record = failure_record(attempt, digest, "execution_error")
            finally:
                try:
                    raw_out.unlink()
                except FileNotFoundError:
                    pass
            atomic_json_write(record_path, record)
            atomic_json_write(out, aggregate(protocol, digest, attempt_dir))

    result = aggregate(protocol, digest, attempt_dir)
    atomic_json_write(out, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--attempt-dir", type=Path, default=DEFAULT_ATTEMPT_DIR)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--ingest-report", action="append", type=Path,
                        dest="ingest_reports", default=[])
    args = parser.parse_args()
    result = run_matrix(**vars(args))
    summary = result["summary"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0 if summary["study_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
