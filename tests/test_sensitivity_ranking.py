# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Check the ranking statistics themselves, independently of the physics.

sensitivity_ranking.py reports numbers that are only as trustworthy as the
rank-correlation routine underneath them, which is hand-written here to avoid a
scipy.stats dependency. So test it against scipy where available, and against
closed-form cases regardless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "orchestrator"))

from sensitivity_ranking import rank_report, spearman  # noqa: E402


def test_spearman_perfect_and_reversed():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(x, 2 * x + 1) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_spearman_is_rank_based_not_value_based():
    # a monotone but violently nonlinear remap must not change the answer
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(x, np.exp(10 * x)) == pytest.approx(1.0)


def test_spearman_handles_ties():
    x = np.array([1.0, 1.0, 2.0, 3.0])
    y = np.array([5.0, 5.0, 6.0, 7.0])
    assert spearman(x, y) == pytest.approx(1.0)


def test_spearman_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.normal(size=200)
        b = 0.4 * a + rng.normal(size=200)
        assert spearman(a, b) == pytest.approx(
            scipy_stats.spearmanr(a, b).statistic, rel=1e-10, abs=1e-12
        )


def test_spearman_matches_scipy_with_ties():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    a = rng.integers(0, 5, size=300).astype(float)
    b = rng.integers(0, 4, size=300).astype(float)
    assert spearman(a, b) == pytest.approx(
        scipy_stats.spearmanr(a, b).statistic, rel=1e-10, abs=1e-12
    )


def test_rank_report_identical_gradients_are_perfect():
    rng = np.random.default_rng(2)
    g = rng.normal(size=(8, 8))
    rep = rank_report(g, g.copy(), ks=(5, 10))
    assert rep["spearman_magnitude"] == pytest.approx(1.0)
    assert rep["top1_correct"] is True
    for row in rep["per_k"]:
        assert row["recall"] == pytest.approx(1.0)
        assert row["sign_agreement_on_true_topk"] == pytest.approx(1.0)
        assert row["n_phantom"] == 0


def test_rank_report_detects_a_planted_phantom():
    # exact: influence decreasing with index. naive: agrees, except it promotes
    # the genuinely least influential cell to the very top.
    n = 100
    a = np.linspace(1.0, 0.01, n)
    b = a.copy()
    b[-1] = 10.0
    rep = rank_report(a, b, ks=(5,))
    row = rep["per_k"][0]
    assert rep["top1_correct"] is False
    assert rep["top1_true_rank_of_naive_pick"] == n - 1
    assert row["n_phantom"] == 1
    assert row["worst_true_rank_promoted"] == n - 1
    assert row["recall"] == pytest.approx(0.8)


def test_sign_agreement_is_measured_on_the_true_top_cells():
    # naive has the right magnitudes everywhere but flips the sign of the two
    # most influential cells: rankings look fine, attribution is dangerous
    a = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    b = np.array([-5.0, -4.0, 3.0, 2.0, 1.0])
    rep = rank_report(a, b, ks=(4,))
    row = rep["per_k"][0]
    assert row["recall"] == pytest.approx(1.0)
    assert row["sign_agreement_on_true_topk"] == pytest.approx(0.5)
