# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Run and aggregate the retrospectively frozen 48-case robustness matrix.

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
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from intervention_robustness import classify_outcome

HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / "protocols" / "intervention_robustness_matrix_48.json"
DEFAULT_OUT = HERE / "results" / "intervention_robustness_matrix_48.json"
DEFAULT_ATTEMPT_DIR = HERE / "results" / "intervention_robustness_matrix_48_attempts"
OUTCOMES = {
    "exact_wins", "shortcut_wins", "tie", "inconclusive",
    "base_not_converged", "runner_failure",
}
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
    if type(design["N"]) is not int or design["N"] <= 0:
        raise ValueError("N must be a positive integer")
    if any(type(seed) is not int for seed in seeds):
        raise ValueError("seeds must be integers")
    if any(isinstance(ra, bool) or not isinstance(ra, (int, float))
           or not math.isfinite(float(ra)) or ra <= 0 for ra in ras):
        raise ValueError("Rayleigh numbers must be finite and positive")
    if (isinstance(design["amplitude"], bool)
            or not isinstance(design["amplitude"], (int, float))
            or not math.isfinite(float(design["amplitude"]))
            or design["amplitude"] <= 0):
        raise ValueError("amplitude must be finite and positive")
    if protocol.get("status") != "retrospectively_frozen_design":
        raise ValueError("protocol must identify the design as retrospectively frozen")
    provenance = protocol.get("frozen_design_provenance", {})
    if not isinstance(provenance.get("commit"), str) or len(provenance["commit"]) != 40:
        raise ValueError("protocol must record the full frozen-design commit")
    z = protocol["analysis"]["z"]
    if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z) or z <= 0:
        raise ValueError("analysis.z must be positive")
    confidence = protocol["analysis"].get("confidence_level")
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence)) or not 0 < confidence < 1):
        raise ValueError("analysis.confidence_level must lie in (0, 1)")
    equivalence = protocol["analysis"]["outcome_numerical_equivalence"]
    for key in ("absolute_delta_J", "relative_delta_J"):
        value = equivalence.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"analysis.outcome_numerical_equivalence.{key} must be finite and non-negative")
    cluster = protocol["analysis"]["cluster_aware_secondary"]
    if not isinstance(cluster.get("analysis_timing"), str):
        raise ValueError("cluster analysis must disclose when it was added")
    if type(cluster.get("bootstrap_seed")) is not int:
        raise ValueError("cluster bootstrap seed must be an integer")
    if type(cluster.get("bootstrap_samples")) is not int or cluster["bootstrap_samples"] < 1:
        raise ValueError("cluster bootstrap samples must be a positive integer")
    timeout = protocol.get("execution", {}).get("attempt_timeout_seconds")
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout)) or timeout <= 0):
        raise ValueError("execution.attempt_timeout_seconds must be finite and positive")
    disclosure = protocol.get("prior_observation_disclosure", {})
    groups = disclosure.get("observed_cells")
    if not isinstance(groups, list) or not groups:
        raise ValueError("prior overlap must be disclosed as observed cell groups")
    observed_pairs: list[tuple[float, int]] = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("seeds"), list):
            raise ValueError("each observed cell group needs Ra and a seed list")
        observed_pairs.extend(
            (float(group["Ra"]), int(seed)) for seed in group["seeds"]
        )
    if len(observed_pairs) != len(set(observed_pairs)):
        raise ValueError("prior overlap disclosure contains duplicate cells")
    planned_pairs = {(float(ra), int(seed)) for ra in ras for seed in seeds}
    if not set(observed_pairs) <= planned_pairs:
        raise ValueError("prior overlap disclosure contains cells outside the design")
    if disclosure.get("observed_cells_count") != len(observed_pairs):
        raise ValueError("observed_cells_count does not match the disclosed cells")
    if disclosure.get("not_previously_stored_cells_count") != expected - len(observed_pairs):
        raise ValueError("not_previously_stored_cells_count does not match the matrix")
    if disclosure.get("overlap_fraction") != f"{len(observed_pairs)}/{expected}":
        raise ValueError("overlap_fraction does not match the disclosed cells")
    stratification = protocol["analysis"].get("prior_observation_stratification", {})
    if not isinstance(stratification.get("analysis_timing"), str):
        raise ValueError("prior-observation stratification must disclose its timing")
    return protocol, hashlib.sha256(raw).hexdigest()


def planned_attempts(protocol: dict) -> list[dict]:
    """Expand the matrix in its frozen deterministic order."""
    design = protocol["design"]
    equivalence = protocol["analysis"]["outcome_numerical_equivalence"]
    return [
        {
            "attempt_id": f"Ra{float(ra):012.3f}_seed{int(seed):03d}",
            "N": int(design["N"]),
            "Ra": float(ra),
            "amplitude": float(design["amplitude"]),
            "seed": int(seed),
            "outcome_absolute_tolerance": float(equivalence["absolute_delta_J"]),
            "outcome_relative_tolerance": float(equivalence["relative_delta_J"]),
            "attempt_timeout_seconds": float(
                protocol["execution"]["attempt_timeout_seconds"]
            ),
        }
        for ra in design["rayleigh_numbers"]
        for seed in design["seeds"]
    ]


def _observed_attempt_ids(protocol: dict, attempts: list[dict]) -> set[str]:
    """Resolve the disclosed union of all previously observed matrix cells."""
    groups = protocol["prior_observation_disclosure"]["observed_cells"]
    pairs = {
        (float(group["Ra"]), int(seed))
        for group in groups
        for seed in group["seeds"]
    }
    return {
        attempt["attempt_id"] for attempt in attempts
        if (attempt["Ra"], attempt["seed"]) in pairs
    }


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


def _staging_report_metadata(path: Path) -> dict:
    """Preserve a compact fingerprint before ephemeral staging evidence is removed."""
    try:
        raw = path.read_bytes()
    except OSError:
        return {"staging_report_present": False}
    return {
        "staging_report_present": True,
        "staging_report_bytes": len(raw),
        "staging_report_sha256": hashlib.sha256(raw).hexdigest(),
    }


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
    return _durable_record_error(record, attempt, protocol_hash) is None


def read_record(path: Path, attempt: dict, protocol_hash: str) -> tuple[dict | None, str | None]:
    """Read a record without allowing a corrupt/mismatched record to be rerun."""
    if not path.exists():
        return None, None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "unreadable_json"
    error = _durable_record_error(record, attempt, protocol_hash)
    if error is not None:
        return None, error
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
        "--outcome-atol", str(attempt["outcome_absolute_tolerance"]),
        "--outcome-rtol", str(attempt["outcome_relative_tolerance"]),
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


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _unit_interval(value: object, field: str) -> float:
    result = _finite_number(value, field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must lie in [0, 1]")
    return result


def _nonempty_reason(case: dict, field: str = "reason") -> str:
    value = case.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _normalize_case(case: dict, attempt: dict) -> dict:
    """Validate case evidence and derive its outcome rather than trusting a label."""
    if not isinstance(case, dict) or type(case.get("seed")) is not int:
        raise ValueError("case must be an object with an integer seed")
    if case["seed"] != attempt["seed"]:
        raise ValueError("case seed does not match the planned attempt")
    reported = case.get("outcome")
    if reported not in OUTCOMES:
        raise ValueError("case has an unknown reported outcome")

    normalized = dict(case)
    matrix_input = case.get("matrix_input_reported_outcome", reported)
    if matrix_input not in OUTCOMES:
        raise ValueError("matrix_input_reported_outcome is invalid")
    normalized["matrix_input_reported_outcome"] = matrix_input
    if reported == "base_not_converged":
        _nonempty_reason(case)
        normalized.setdefault("failure_stage", "base_forward")
        return normalized
    if reported == "runner_failure":
        _nonempty_reason(case)
        normalized.setdefault("failure_stage", "execution_unit")
        return normalized

    if reported == "inconclusive":
        exact_ok = case.get("exact_action_ok")
        shortcut_ok = case.get("naive_action_ok")
        # Older runners used "inconclusive" as a catch-all for unexpected
        # Python exceptions. Without action-status evidence this is not a
        # scientific non-comparison; preserve it explicitly as runner failure.
        if type(exact_ok) is not bool or type(shortcut_ok) is not bool:
            reason = _nonempty_reason(case)
            normalized.update({
                "outcome": "runner_failure",
                "failure_stage": "unstructured_execution_unit_exception",
                "reason": reason,
            })
            return normalized
        if exact_ok and shortcut_ok:
            raise ValueError("an inconclusive case cannot have two successful actions")
        for prefix, ok in (("exact", exact_ok), ("naive", shortcut_ok)):
            delta = case.get(f"delta_J_{prefix}_action")
            reason = case.get(f"{prefix}_action_reason")
            if ok:
                _finite_number(delta, f"delta_J_{prefix}_action")
                if reason is not None:
                    raise ValueError(f"successful {prefix} action cannot have a failure reason")
            else:
                if delta is not None:
                    raise ValueError(f"failed {prefix} action cannot have delta_J evidence")
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError(f"failed {prefix} action needs a non-empty reason")
        gamma = _finite_number(case.get("gamma"), "gamma")
        relative_error = _finite_number(
            case.get("naive_relative_error"), "naive_relative_error"
        )
        cosine = _finite_number(case.get("naive_cosine"), "naive_cosine")
        if gamma < 0 or relative_error < 0 or not -1.0 <= cosine <= 1.0:
            raise ValueError("gradient diagnostics are outside their valid ranges")
        _unit_interval(case.get("add_set_overlap"), "add_set_overlap")
        _unit_interval(case.get("remove_set_overlap"), "remove_set_overlap")
        return normalized

    gamma = _finite_number(case.get("gamma"), "gamma")
    relative_error = _finite_number(
        case.get("naive_relative_error"), "naive_relative_error"
    )
    cosine = _finite_number(case.get("naive_cosine"), "naive_cosine")
    if gamma < 0 or relative_error < 0 or not -1.0 <= cosine <= 1.0:
        raise ValueError("gradient diagnostics are outside their valid ranges")
    _unit_interval(case.get("add_set_overlap"), "add_set_overlap")
    _unit_interval(case.get("remove_set_overlap"), "remove_set_overlap")
    source_outcome = case.get("execution_unit_reported_outcome")
    if source_outcome not in COMPARABLE:
        raise ValueError("comparable case needs a recognized execution-unit outcome")
    exact = _finite_number(case.get("delta_J_exact_action"), "delta_J_exact_action")
    shortcut = _finite_number(case.get("delta_J_naive_action"), "delta_J_naive_action")
    outcome, advantage, tolerance = classify_outcome(
        exact,
        shortcut,
        attempt["outcome_absolute_tolerance"],
        attempt["outcome_relative_tolerance"],
    )
    supplied_advantage = _finite_number(case.get("exact_advantage"), "exact_advantage")
    if not math.isclose(supplied_advantage, advantage, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("exact_advantage is inconsistent with the two objective changes")
    supplied_tolerance = _finite_number(
        case.get("outcome_equivalence_tolerance"), "outcome_equivalence_tolerance"
    )
    if not math.isclose(supplied_tolerance, tolerance, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("case outcome tolerance is inconsistent with the protocol")
    expected_extra = None if shortcut == 0 else abs(exact) / abs(shortcut) - 1.0
    supplied_extra = case.get("extra_cooling_fraction")
    if expected_extra is None:
        if supplied_extra is not None:
            raise ValueError("extra_cooling_fraction must be null for zero shortcut change")
    else:
        extra = _finite_number(supplied_extra, "extra_cooling_fraction")
        if not math.isclose(extra, expected_extra, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("extra_cooling_fraction is inconsistent with objective changes")
    normalized.update({
        "outcome": outcome,
        "exact_advantage": advantage,
        "outcome_equivalence_tolerance": tolerance,
        "extra_cooling_fraction": expected_extra,
    })
    return normalized


def _durable_record_error(record: object, attempt: dict,
                          protocol_hash: str) -> str | None:
    """Validate saved evidence, including recomputing every scientific label."""
    if not isinstance(record, dict):
        return "record_is_not_an_object"
    if record.get("schema_version") != 1:
        return "unsupported_record_schema"
    if record.get("protocol_sha256") != protocol_hash:
        return "record_protocol_mismatch"
    if any(record.get(key) != value for key, value in attempt.items()):
        return "record_attempt_mismatch"
    outcome = record.get("outcome")
    if outcome not in OUTCOMES:
        return "record_unknown_outcome"
    returncode = record.get("execution_returncode")
    if returncode is not None and (isinstance(returncode, bool) or type(returncode) is not int):
        return "record_invalid_returncode"
    if type(record.get("timed_out")) is not bool:
        return "record_missing_timeout_status"
    staging_present = record.get("staging_report_present")
    if type(staging_present) is not bool:
        return "record_missing_staging_report_status"
    if staging_present:
        size = record.get("staging_report_bytes")
        digest = record.get("staging_report_sha256")
        if type(size) is not int or size < 0:
            return "record_invalid_staging_report_size"
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            return "record_invalid_staging_report_digest"
    elapsed = record.get("elapsed_seconds")
    if elapsed is not None:
        try:
            if _finite_number(elapsed, "elapsed_seconds") < 0:
                return "record_invalid_elapsed_seconds"
        except ValueError:
            return "record_invalid_elapsed_seconds"

    if "case" in record:
        try:
            normalized = _normalize_case(record["case"], attempt)
        except ValueError:
            return "record_invalid_case_evidence"
        if normalized != record["case"]:
            return "record_case_evidence_not_normalized"
        if outcome != normalized["outcome"]:
            return "record_outcome_disagrees_with_case"
        if record["timed_out"]:
            return "case_record_cannot_be_timeout"
        return None

    if outcome != "runner_failure":
        return "scientific_outcome_missing_case_evidence"
    stage = record.get("failure_stage")
    reason = record.get("reason")
    if not isinstance(stage, str) or not stage.strip():
        return "runner_failure_missing_stage"
    if not isinstance(reason, str) or not reason.strip():
        return "runner_failure_missing_reason"
    if record["timed_out"] != (stage == "timeout"):
        return "runner_failure_timeout_status_mismatch"
    return None


def _validate_report_object(report: dict, attempts: list[dict], source: str) -> list[tuple[dict, dict]]:
    """Validate one complete execution-unit report and all case evidence."""
    if not isinstance(report, dict):
        raise ValueError(f"{source} must contain a JSON object")
    if not attempts:
        raise ValueError("at least one expected attempt is required")
    first = attempts[0]
    if report.get("N") != first["N"] or report.get("Ra") != first["Ra"]:
        raise ValueError(f"{source} has a design mismatch")
    if report.get("amplitude") != first["amplitude"]:
        raise ValueError(f"{source} has the wrong amplitude")
    if report.get("seeds_planned") != len(attempts) or report.get("complete") is not True:
        raise ValueError(f"{source} is not a complete planned slice")
    declared_tolerance = report.get("outcome_equivalence_tolerance")
    if declared_tolerance != {
        "absolute_delta_J": first["outcome_absolute_tolerance"],
        "relative_delta_J": first["outcome_relative_tolerance"],
    }:
        raise ValueError(f"{source} declares a different outcome tolerance")

    by_seed = {attempt["seed"]: attempt for attempt in attempts}
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != len(attempts):
        raise ValueError(f"{source} must contain exactly the planned cases")
    normalized_by_seed: dict[int, dict] = {}
    for case in cases:
        if not isinstance(case, dict) or type(case.get("seed")) is not int:
            raise ValueError(f"{source} contains an invalid seed")
        if "matrix_input_reported_outcome" in case:
            raise ValueError(f"{source} uses a matrix-reserved case field")
        seed = case["seed"]
        if seed not in by_seed:
            raise ValueError(f"{source} contains a non-frozen seed {seed}")
        if seed in normalized_by_seed:
            raise ValueError(f"{source} contains duplicate seed {seed}")
        normalized_by_seed[seed] = _normalize_case(case, by_seed[seed])
    if set(normalized_by_seed) != set(by_seed):
        raise ValueError(f"{source} is missing planned seeds")

    normalized_cases = list(normalized_by_seed.values())
    derived = {
        "seeds_attempted": len(normalized_cases),
        "seeds_comparable": sum(c["outcome"] in COMPARABLE for c in normalized_cases),
        "seeds_inconclusive": sum(c["outcome"] == "inconclusive" for c in normalized_cases),
        "base_state_failures": sum(c["outcome"] == "base_not_converged" for c in normalized_cases),
        "runner_failures": sum(c["outcome"] == "runner_failure" for c in normalized_cases),
        "exact_wins": sum(c["outcome"] == "exact_wins" for c in normalized_cases),
        "shortcut_wins": sum(c["outcome"] == "shortcut_wins" for c in normalized_cases),
        "ties": sum(c["outcome"] == "tie" for c in normalized_cases),
    }
    for key, expected in derived.items():
        if report.get(key) != expected:
            raise ValueError(f"{source} has inconsistent derived count {key}")
    return [(attempt, normalized_by_seed[attempt["seed"]]) for attempt in attempts]


def normalize_report(attempt: dict, protocol_hash: str, raw_out: Path,
                     returncode: int, elapsed_seconds: float) -> dict:
    """Convert a fully validated execution-unit report into one durable record."""
    base = {"schema_version": 1, "protocol_sha256": protocol_hash, **attempt,
            "execution_returncode": returncode, "timed_out": False,
            "elapsed_seconds": elapsed_seconds}
    try:
        raw = raw_out.read_bytes()
    except FileNotFoundError:
        return {**base, "outcome": "runner_failure", "failure_stage": "missing_report",
                "reason": "execution unit did not create its requested JSON report",
                "staging_report_present": False}
    except OSError as exc:
        return {**base, "outcome": "runner_failure", "failure_stage": "unreadable_report",
                "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
                "staging_report_present": False}
    source = {
        "source": "executed_attempt",
        "staging_report_present": True,
        "staging_report_bytes": len(raw),
        "staging_report_sha256": hashlib.sha256(raw).hexdigest(),
    }
    try:
        report = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {**base, "outcome": "runner_failure", "failure_stage": "invalid_report_json",
                "reason": f"{type(exc).__name__}: {str(exc)[:300]}", **source}
    try:
        [(_, case)] = _validate_report_object(report, [attempt], str(raw_out))
    except ValueError as exc:
        return {
            **base,
            "outcome": "runner_failure",
            "failure_stage": "invalid_report_evidence",
            "reason": str(exc)[:400],
            **source,
        }
    return {**base, "outcome": case["outcome"], "case": case, **source}


def failure_record(attempt: dict, protocol_hash: str, stage: str, reason: str,
                   *, elapsed_seconds: float, timed_out: bool,
                   staging_metadata: dict) -> dict:
    """Make a terminal, auditable orchestration-failure record."""
    return {"schema_version": 1, "protocol_sha256": protocol_hash, **attempt,
            "execution_returncode": None, "outcome": "runner_failure",
            "failure_stage": stage, "reason": reason, "timed_out": timed_out,
            "elapsed_seconds": elapsed_seconds, **staging_metadata}


def _validated_ingest_cases(report_path: Path, protocol: dict) -> tuple[list[tuple[dict, dict]], str]:
    """Validate a complete one-Ra execution-unit report before any writes."""
    try:
        raw = report_path.read_bytes()
        report = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read ingest report {report_path}: {type(exc).__name__}") from exc
    design = protocol["design"]
    allowed_ras = {float(ra) for ra in design["rayleigh_numbers"]}
    try:
        ra = float(report["Ra"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"ingest report {report_path} has an invalid Ra") from exc
    if ra not in allowed_ras:
        raise ValueError(f"ingest report {report_path} has a non-frozen Ra")

    attempts = [attempt for attempt in planned_attempts(protocol) if attempt["Ra"] == ra]
    ordered = _validate_report_object(report, attempts, f"ingest report {report_path}")
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
            "timed_out": False,
            "elapsed_seconds": None,
            "staging_report_present": False,
            "outcome": case["outcome"],
            "case": case,
            "source": "ingested_report",
            "source_report_sha256": report_hash,
        }
        atomic_json_write(record_path, record)
        counts["written"] += 1
    return counts


def _group_summary(attempts: list[dict], records: list[dict], z: float) -> dict:
    outcomes = {name: 0 for name in sorted(OUTCOMES)}
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


def _percentile(values: list[float], probability: float) -> float | None:
    """Linearly interpolated percentile without an additional dependency."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _cluster_aware_seed_summary(protocol: dict, attempts: list[dict],
                                records: list[dict]) -> dict:
    """Bootstrap seed clusters, preserving the three correlated Ra outcomes."""
    design = protocol["design"]
    planned_ras = {float(ra) for ra in design["rayleigh_numbers"]}
    records_by_seed: dict[int, list[dict]] = {int(seed): [] for seed in design["seeds"]}
    for record in records:
        records_by_seed[record["seed"]].append(record)

    seed_clusters = []
    complete = []
    direction_counts = {
        "exact_dominant": 0,
        "shortcut_dominant": 0,
        "balanced": 0,
        "no_comparable_cases": 0,
    }
    for seed in design["seeds"]:
        rows = records_by_seed[int(seed)]
        ras = {row["Ra"] for row in rows}
        is_complete = ras == planned_ras and len(rows) == len(planned_ras)
        exact = sum(row["outcome"] == "exact_wins" for row in rows)
        shortcut = sum(row["outcome"] == "shortcut_wins" for row in rows)
        ties = sum(row["outcome"] == "tie" for row in rows)
        comparable = exact + shortcut + ties
        cluster = {
            "seed": int(seed),
            "complete": is_complete,
            "records": len(rows),
            "comparable_cases": comparable,
            "exact_wins": exact,
            "shortcut_wins": shortcut,
            "ties": ties,
        }
        seed_clusters.append(cluster)
        if is_complete:
            complete.append(cluster)
            if comparable == 0:
                direction_counts["no_comparable_cases"] += 1
            elif exact > shortcut:
                direction_counts["exact_dominant"] += 1
            elif shortcut > exact:
                direction_counts["shortcut_dominant"] += 1
            else:
                direction_counts["balanced"] += 1

    config = protocol["analysis"]["cluster_aware_secondary"]
    bootstrap_seed = int(config["bootstrap_seed"])
    requested = int(config["bootstrap_samples"])
    confidence = float(protocol["analysis"]["confidence_level"])
    exact_total = sum(cluster["exact_wins"] for cluster in complete)
    comparable_total = sum(cluster["comparable_cases"] for cluster in complete)
    estimate = exact_total / comparable_total if comparable_total else None
    bootstrap_values: list[float] = []
    if complete:
        rng = random.Random(bootstrap_seed)
        for _ in range(requested):
            sampled = [complete[rng.randrange(len(complete))] for _ in complete]
            numerator = sum(cluster["exact_wins"] for cluster in sampled)
            denominator = sum(cluster["comparable_cases"] for cluster in sampled)
            if denominator:
                bootstrap_values.append(numerator / denominator)
    tail = (1.0 - confidence) / 2.0
    return {
        "analysis_timing": config["analysis_timing"],
        "estimand": "pooled exact wins divided by pooled comparable cases in resampled complete seed clusters",
        "cluster_unit": "seed",
        "clusters_planned": len(design["seeds"]),
        "complete_clusters": len(complete),
        "incomplete_clusters": len(design["seeds"]) - len(complete),
        "all_planned_clusters_complete": len(complete) == len(design["seeds"]),
        "pooled_exact_win_rate_over_comparable_in_complete_clusters": estimate,
        "paired_seed_direction_counts": direction_counts,
        "bootstrap": {
            "method": "resample complete seed clusters with replacement; pool their attempt outcomes",
            "seed": bootstrap_seed,
            "samples_requested": requested,
            "samples_with_comparable_cases": len(bootstrap_values),
            "confidence_level": confidence,
            "lower": _percentile(bootstrap_values, tail),
            "upper": _percentile(bootstrap_values, 1.0 - tail),
        },
        "seed_clusters": seed_clusters,
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
    summary["cluster_aware_seed_analysis"] = _cluster_aware_seed_summary(
        protocol, attempts, records
    )
    observed_ids = _observed_attempt_ids(protocol, attempts)
    observed_attempts = [attempt for attempt in attempts if attempt["attempt_id"] in observed_ids]
    new_attempts = [attempt for attempt in attempts if attempt["attempt_id"] not in observed_ids]
    observed_records = [record for record in records if record["attempt_id"] in observed_ids]
    new_records = [record for record in records if record["attempt_id"] not in observed_ids]
    observation_strata = {
        "analysis_timing": protocol["analysis"]["prior_observation_stratification"][
            "analysis_timing"
        ],
        "interpretation": protocol["analysis"]["prior_observation_stratification"][
            "interpretation"
        ],
        "observed_before_frozen_design": _group_summary(observed_attempts, observed_records, z),
        "not_stored_before_frozen_design": _group_summary(new_attempts, new_records, z),
    }
    return {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "protocol_sha256": protocol_hash,
        "confidence_level": protocol["analysis"]["confidence_level"],
        "summary": summary,
        "by_rayleigh_number": by_ra,
        "by_prior_observation_status": observation_strata,
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
            started = time.perf_counter()
            try:
                completed = executor(attempt, raw_out, timeout)
                elapsed = time.perf_counter() - started
                record = normalize_report(
                    attempt, digest, raw_out, completed.returncode, elapsed
                )
            except subprocess.TimeoutExpired as exc:
                elapsed = time.perf_counter() - started
                record = failure_record(
                    attempt,
                    digest,
                    "timeout",
                    f"execution exceeded the configured {timeout:g}-second timeout: {exc}",
                    elapsed_seconds=elapsed,
                    timed_out=True,
                    staging_metadata=_staging_report_metadata(raw_out),
                )
            except Exception as exc:  # noqa: BLE001 - orchestration failure is study data
                elapsed = time.perf_counter() - started
                record = failure_record(
                    attempt,
                    digest,
                    "execution_error",
                    f"{type(exc).__name__}: {str(exc)[:300]}",
                    elapsed_seconds=elapsed,
                    timed_out=False,
                    staging_metadata=_staging_report_metadata(raw_out),
                )
            # The terminal record must exist before deleting staging evidence.
            # If this atomic write fails, the raw report remains for diagnosis.
            atomic_json_write(record_path, record)
            try:
                raw_out.unlink()
            except FileNotFoundError:
                pass
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
