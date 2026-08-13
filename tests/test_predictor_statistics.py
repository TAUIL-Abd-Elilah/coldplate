# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "orchestrator"))

from predictor_statistics import summarize  # noqa: E402


def test_summary_reports_holdout_and_bootstrap_statistics():
    rows = []
    for family, scale in (("a", 1.0), ("b", 1.2), ("c", 0.8)):
        for i in range(1, 5):
            gamma = scale * 10 ** (-i)
            rows.append({"design": family, "Ra": i, "gamma": gamma,
                         "rho_phi": 0.1 * i, "rel_err": 2.0 * gamma})
    report = summarize(rows, n_bootstrap=100, seed=1)
    assert report["n_converged"] == 12
    assert set(report["leave_one_family_out"]) == {"a", "b", "c"}
    assert report["log_gamma_correlation"] > 0.99
    assert len(report["bootstrap_95_percent_interval"]) == 2
