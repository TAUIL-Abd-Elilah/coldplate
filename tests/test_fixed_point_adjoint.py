# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import FrozenInstanceError
import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fixed_point_adjoint import (  # noqa: E402
    FixedPointStability,
    ResidualThresholds,
    fixed_point_adjoint_residual,
)


def test_array_residual_and_repeats_are_exact():
    report = fixed_point_adjoint_residual(
        lambda x: 0.25 * x,
        jnp.zeros(2),
        jnp.array([3.0, 4.0]),
        num_repeats=3,
    )
    assert report.cotangent_norm == pytest.approx(5.0)
    assert report.residual_norm == pytest.approx(1.25)
    assert report.adjoint_residual_norm == pytest.approx(report.residual_norm)
    assert report.objective_cotangent_norm == pytest.approx(report.cotangent_norm)
    assert report.relative_residual == pytest.approx(0.25)
    assert report.gamma == pytest.approx(report.relative_residual)
    assert report.repeated_residual_norms == pytest.approx((1.25, 0.3125, 0.078125))
    assert report.repeated_relative_norms == pytest.approx((0.25, 0.0625, 0.015625))
    assert report.verdict is None
    with pytest.raises(FrozenInstanceError):
        report.relative_residual = 0.0


def test_arbitrary_pytrees_use_one_aggregate_leaf_norm():
    point = {"a": jnp.zeros(2), "nested": (jnp.zeros(1),)}
    cotangent = {"a": jnp.array([3.0, 0.0]), "nested": (jnp.array([4.0]),)}

    def phi(tree):
        return {"a": 2.0 * tree["a"], "nested": (0.5 * tree["nested"][0],)}

    report = fixed_point_adjoint_residual(phi, point, cotangent, num_repeats=1)
    assert report.cotangent_norm == pytest.approx(5.0)
    assert report.residual_norm == pytest.approx(jnp.sqrt(40.0))
    assert report.gamma == pytest.approx(jnp.sqrt(40.0) / 5.0)


def test_nonlinear_map_is_linearized_at_supplied_fixed_point():
    # d/dx tanh(2x) at x=0 is 2, independent of the nonlinear map elsewhere.
    report = fixed_point_adjoint_residual(
        lambda x: jnp.tanh(2.0 * x),
        jnp.zeros(3),
        jnp.array([1.0, -2.0, 3.0]),
        num_repeats=2,
    )
    assert report.repeated_relative_norms == pytest.approx((2.0, 4.0))


@pytest.mark.parametrize("num_repeats", [0, -1])
def test_invalid_repeat_count_is_rejected(num_repeats):
    with pytest.raises(ValueError, match="at least 1"):
        fixed_point_adjoint_residual(lambda x: x, jnp.ones(1), jnp.ones(1), num_repeats)


def test_zero_cotangent_is_rejected_for_a_pytree():
    with pytest.raises(ValueError, match="identically zero"):
        fixed_point_adjoint_residual(
            lambda x: x,
            {"x": jnp.ones(2)},
            {"x": jnp.zeros(2)},
        )


def test_fixed_point_check_is_optional_and_aggregated():
    phi = lambda tree: {"x": tree["x"] + jnp.array([3.0, 4.0])}  # noqa: E731
    without_check = fixed_point_adjoint_residual(phi, {"x": jnp.zeros(2)}, {"x": jnp.ones(2)})
    with_check = fixed_point_adjoint_residual(
        phi, {"x": jnp.zeros(2)}, {"x": jnp.ones(2)}, check_fixed_point=True
    )
    assert without_check.fixed_point_residual is None
    assert with_check.fixed_point_residual == pytest.approx(5.0)
    assert with_check.fixed_point_residual_norm == pytest.approx(5.0)


def test_only_caller_calibrated_thresholds_produce_a_verdict():
    args = (lambda x: 0.2 * x, jnp.zeros(2), jnp.ones(2))
    assert fixed_point_adjoint_residual(*args).verdict is None
    assert fixed_point_adjoint_residual(
        *args, thresholds=ResidualThresholds(0.3, 0.5)
    ).verdict == "SAFE"
    assert fixed_point_adjoint_residual(*args, thresholds=(0.1, 0.5)).verdict == "MARGINAL"
    assert fixed_point_adjoint_residual(
        *args, thresholds={"safe": 0.1, "risky": 0.15}
    ).verdict == "UNSAFE"
    with pytest.raises(ValueError, match="0 <= safe < risky"):
        ResidualThresholds(0.5, 0.1)


@pytest.mark.parametrize("stability", [FixedPointStability.REPELLING, "repelling", 1.01])
def test_known_repelling_map_never_returns_safe(stability):
    report = fixed_point_adjoint_residual(
        lambda x: 0.01 * x,
        jnp.zeros(2),
        jnp.ones(2),
        thresholds=(0.1, 0.5),
        stability=stability,
    )
    assert report.verdict == "UNSAFE"
    assert report.verdict != "SAFE"
