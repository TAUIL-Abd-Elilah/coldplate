# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "orchestrator"))

from intervention_robustness import summarize  # noqa: E402


def test_robustness_summary_counts_forward_validated_wins():
    def report(seed, exact, naive):
        return {
            "N": 20,
            "Ra": 2.0e4,
            "seed": seed,
            "gamma": 0.3,
            "naive_relative_error": 0.7,
            "naive_cosine": 0.8,
            "add_set_overlap": 0.4,
            "remove_set_overlap": 0.2,
            "rows": [{
                "amplitude": 0.025,
                "delta_J_exact_action": exact,
                "delta_J_naive_action": naive,
                "exact_advantage": naive - exact,
            }],
        }

    result = summarize([
        report(1, -0.04, -0.035),
        report(3, -0.06, -0.02),
        report(4, -0.05, -0.03),
    ])
    assert result["exact_wins"] == result["n_seeds"] == 3
    assert result["all_actions_reduce_J"]
    assert result["cases"][0]["extra_cooling_fraction"] > 0
