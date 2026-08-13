# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Robustness statistics for the one-VJP predictor result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def correlation(rows, field: str) -> float:
    error = np.log10(np.maximum([r["rel_err"] for r in rows], 1e-12))
    value = np.asarray([r[field] for r in rows], dtype=float)
    if field == "gamma":
        value = np.log10(np.maximum(value, 1e-12))
    return float(np.corrcoef(value, error)[0, 1])


def summarize(rows, n_bootstrap: int = 10_000, seed: int = 2026) -> dict:
    if len(rows) < 3:
        raise ValueError("at least three converged configurations are required")
    families = sorted({r["design"] for r in rows})
    levels = sorted({float(r["Ra"]) for r in rows})
    lofo = {}
    for held_out in families:
        train = [r for r in rows if r["design"] != held_out]
        lofo[held_out] = {
            "n": len(train),
            "log_gamma_correlation": correlation(train, "gamma"),
            "rho_correlation": correlation(train, "rho_phi"),
        }

    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(n_bootstrap):
        sample = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        value = correlation(sample, "gamma")
        if np.isfinite(value):
            bootstrap.append(value)
    low, median, high = np.quantile(bootstrap, [0.025, 0.5, 0.975])
    return {
        "n_converged": len(rows),
        "n_design_families": len(families),
        "design_families": families,
        "n_rayleigh_levels_represented": len(levels),
        "rayleigh_levels_represented": levels,
        "attempted_configurations": 20,
        "log_gamma_correlation": correlation(rows, "gamma"),
        "rho_correlation": correlation(rows, "rho_phi"),
        "leave_one_family_out": lofo,
        "bootstrap_seed": seed,
        "bootstrap_samples": n_bootstrap,
        "bootstrap_95_percent_interval": [float(low), float(high)],
        "bootstrap_median": float(median),
    }


def main(source="results/predict_error.json", out="results/predictor_statistics.json"):
    rows = json.loads(Path(source).read_text())
    result = summarize(rows)
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print(f"n={result['n_converged']}; log-gamma corr={result['log_gamma_correlation']:.4f}")
    print("leave-one-family-out log-gamma correlations:")
    for family, row in result["leave_one_family_out"].items():
        print(f"  {family:>13}: {row['log_gamma_correlation']:.4f} (n={row['n']})")
    lo, hi = result["bootstrap_95_percent_interval"]
    print(f"bootstrap 95% interval: [{lo:.4f}, {hi:.4f}]")
    print(f"wrote {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="results/predict_error.json")
    parser.add_argument("--out", default="results/predictor_statistics.json")
    main(**vars(parser.parse_args()))
