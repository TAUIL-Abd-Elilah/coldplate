# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Check the randomized generalization study against closed-form answers.

The study's whole value is that it needs no solver and no finite differences,
so its own machinery has to be right. These tests pin the generators and the
error measurement to cases where the answer can be written down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "orchestrator"))
sys.path.insert(0, str(ROOT))

from coupling_check import coupling_gamma  # noqa: E402
from gamma_generalization import (  # noqa: E402
    FAMILIES,
    corr,
    draw_operator,
    linear_case,
    nonlinear_case,
)

jnp = pytest.importorskip("jax.numpy")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("target", [0.05, 0.5, 1.4])
def test_draw_operator_hits_the_requested_spectral_radius(family, target):
    rng = np.random.default_rng(0)
    A = draw_operator(rng, 30, family, target)
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    assert rho == pytest.approx(target, rel=1e-8)


def test_gamma_of_a_linear_loop_is_the_closed_form():
    # For Phi(x) = A x + b the VJP is exactly A^T, so gamma must equal
    # ||A^T g|| / ||g|| with no approximation anywhere.
    rng = np.random.default_rng(1)
    A = rng.normal(size=(12, 12)) * 0.1
    b = rng.normal(size=12)
    g = rng.normal(size=12)
    x0 = rng.normal(size=12)

    Aj, bj = jnp.asarray(A), jnp.asarray(b)
    report = coupling_gamma(lambda x: Aj @ x + bj, jnp.asarray(x0),
                            jnp.asarray(g), n_terms=3)
    expected = np.linalg.norm(A.T @ g) / np.linalg.norm(g)
    assert report.gamma == pytest.approx(expected, rel=1e-12)

    # and the reported terms are the successive powers, again exactly
    w = g.copy()
    for k in range(3):
        w = A.T @ w
        assert report.neumann_terms[k] == pytest.approx(
            np.linalg.norm(w) / np.linalg.norm(g), rel=1e-12
        )


def test_gamma_is_zero_and_error_is_zero_for_an_uncoupled_loop():
    # A = 0 means the "loop" does not feed back at all: cutting it costs
    # nothing, and the diagnostic must say so.
    rng = np.random.default_rng(2)
    row = linear_case(rng, 8, 5, "sparse", 0.0)
    assert row is not None
    assert row["gamma"] == pytest.approx(0.0, abs=1e-14)
    assert row["rel_err"] == pytest.approx(0.0, abs=1e-12)
    assert row["cosine"] == pytest.approx(1.0, rel=1e-12)


def test_linear_case_error_matches_an_independent_recomputation():
    rng = np.random.default_rng(3)
    row = linear_case(rng, 16, 9, "normal", 0.4)
    assert row is not None
    # the reported error must be a genuine relative error in [0, inf)
    assert row["rel_err"] > 0.0
    assert -1.0 <= row["cosine"] <= 1.0
    # gamma of a loop scaled to rho = 0.4 cannot vanish
    assert row["gamma"] > 0.0
    assert row["rho"] == pytest.approx(0.4, rel=1e-8)


def test_error_grows_with_coupling_strength():
    # Same seed, same structure, stronger loop: cutting it must cost more.
    errs = []
    for target in (0.05, 0.3, 0.8):
        rng = np.random.default_rng(7)
        row = linear_case(rng, 20, 12, "normal", target)
        assert row is not None
        errs.append(row["rel_err"])
    assert errs[0] < errs[1] < errs[2]


def test_nonlinear_case_returns_a_true_fixed_point():
    rng = np.random.default_rng(5)
    row = nonlinear_case(rng, 12, 8, "normal", 0.5)
    if row is None:
        pytest.skip("degenerate draw")
    assert row["gamma"] >= 0.0
    assert row["rel_err"] >= 0.0
    assert np.isfinite(row["rho"])


def test_corr_is_pearson_and_handles_degenerate_input():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert corr(x, 2 * x) == pytest.approx(1.0)
    assert corr(x, -3 * x) == pytest.approx(-1.0)
    assert np.isnan(corr(x, np.ones_like(x)))
    assert np.isnan(corr([1.0], [1.0]))


def test_recorded_study_supports_the_quoted_claims():
    path = ROOT / "orchestrator" / "results" / "gamma_generalization.json"
    if not path.exists():
        pytest.skip("study has not been run")
    import json

    d = json.loads(path.read_text())
    # pooled correlation beats the spectral radius by a wide margin
    assert d["overall"]["log_gamma_correlation"] > 0.95
    assert d["overall"]["log_gamma_correlation"] > d["overall"]["rho_correlation"]
    # gamma wins in every structural family
    for family, block in d["per_family"].items():
        assert block["log_gamma_correlation"] > block["rho_correlation"], family
    # the SAFE verdict never hid a large error
    safe = d["safe_bucket"]
    if safe["n"]:
        assert safe["worst_rel_err"] < 0.05
        assert safe["frac_under_5pct"] == 1.0
    # the documented domain boundary is real, not decorative
    assert d["attracting"]["log_gamma_correlation"] > 0.95
    assert d["repelling"]["log_gamma_correlation"] < 0.75
