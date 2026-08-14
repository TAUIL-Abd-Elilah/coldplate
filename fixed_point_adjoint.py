# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Objective-aware residual diagnostics for fixed-point adjoints.

For a fixed point ``x = phi(x)`` and objective cotangent ``g``, the common
loop-cut adjoint approximation is ``lambda_0 = g``.  Its residual in

    (I - dphi/dx).T lambda = g

is ``(dphi/dx).T g``.  This module measures that residual without assuming
that the fixed-point map is attracting and without attaching universal safety
meaning to its magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import operator
from collections.abc import Mapping, Sequence
from typing import Any, Callable, TypeAlias

import jax
import jax.numpy as jnp

PyTree: TypeAlias = Any


class FixedPointStability(str, Enum):
    """Known local stability of ordinary fixed-point iteration."""

    ATTRACTING = "attracting"
    REPELLING = "repelling"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ResidualThresholds:
    """Caller-calibrated boundaries for interpreting a relative residual.

    Values below ``safe`` are labelled ``SAFE``; values below ``risky`` are
    labelled ``MARGINAL``; and larger values are labelled ``UNSAFE``.  These
    boundaries are application calibration, not mathematical guarantees.
    """

    safe: float
    risky: float

    def __post_init__(self) -> None:
        safe = float(self.safe)
        risky = float(self.risky)
        if not (math.isfinite(safe) and math.isfinite(risky)):
            raise ValueError("thresholds must be finite")
        if not 0.0 <= safe < risky:
            raise ValueError("thresholds must satisfy 0 <= safe < risky")
        object.__setattr__(self, "safe", safe)
        object.__setattr__(self, "risky", risky)


@dataclass(frozen=True)
class FixedPointAdjointResidualReport:
    """Immutable result of :func:`fixed_point_adjoint_residual`."""

    relative_residual: float
    residual_norm: float
    cotangent_norm: float
    repeated_residual_norms: tuple[float, ...]
    repeated_relative_norms: tuple[float, ...]
    fixed_point_residual: float | None
    verdict: str | None
    stability: FixedPointStability

    @property
    def gamma(self) -> float:
        """Alias for the first normalized adjoint residual."""

        return self.relative_residual

    @property
    def adjoint_residual_norm(self) -> float:
        """Equation-specific alias for :attr:`residual_norm`."""

        return self.residual_norm

    @property
    def objective_cotangent_norm(self) -> float:
        """Explicit alias for :attr:`cotangent_norm`."""

        return self.cotangent_norm

    @property
    def fixed_point_residual_norm(self) -> float | None:
        """Explicit alias for :attr:`fixed_point_residual`."""

        return self.fixed_point_residual


ThresholdInput: TypeAlias = (
    ResidualThresholds | tuple[float, float] | Sequence[float] | Mapping[str, float]
)
StabilityInput: TypeAlias = FixedPointStability | str | float


def _tree_l2_norm(tree: PyTree) -> float:
    """Aggregate leaf L2 norms as one Euclidean product-space norm."""

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return 0.0
    squared = [jnp.sum(jnp.abs(jnp.asarray(leaf)) ** 2) for leaf in leaves]
    return float(jnp.sqrt(jnp.sum(jnp.stack(squared))))


def _normalize_thresholds(thresholds: ThresholdInput | None) -> ResidualThresholds | None:
    if thresholds is None:
        return None
    if isinstance(thresholds, ResidualThresholds):
        return thresholds
    if isinstance(thresholds, Mapping):
        try:
            return ResidualThresholds(thresholds["safe"], thresholds["risky"])
        except KeyError as exc:
            raise ValueError("threshold mapping must contain 'safe' and 'risky'") from exc
    if isinstance(thresholds, Sequence) and not isinstance(thresholds, (str, bytes)):
        if len(thresholds) != 2:
            raise ValueError("thresholds must contain exactly (safe, risky)")
        return ResidualThresholds(thresholds[0], thresholds[1])
    raise TypeError("thresholds must be ResidualThresholds or a (safe, risky) pair")


def _normalize_stability(stability: StabilityInput | None) -> FixedPointStability:
    if stability is None:
        return FixedPointStability.UNKNOWN
    if isinstance(stability, FixedPointStability):
        return stability
    if isinstance(stability, str):
        try:
            return FixedPointStability(stability.lower())
        except ValueError as exc:
            allowed = ", ".join(item.value for item in FixedPointStability)
            raise ValueError(f"stability must be one of: {allowed}") from exc
    try:
        spectral_radius = float(stability)
    except (TypeError, ValueError) as exc:
        raise TypeError("stability must be a stability label or spectral radius") from exc
    if not math.isfinite(spectral_radius) or spectral_radius < 0.0:
        raise ValueError("a stability spectral radius must be finite and non-negative")
    if spectral_radius < 1.0:
        return FixedPointStability.ATTRACTING
    return FixedPointStability.REPELLING


def _verdict(
    relative_residual: float,
    thresholds: ResidualThresholds | None,
    stability: FixedPointStability,
) -> str | None:
    if thresholds is None:
        return None
    if stability is FixedPointStability.REPELLING:
        return "UNSAFE"
    if relative_residual < thresholds.safe:
        return "SAFE"
    if relative_residual < thresholds.risky:
        return "MARGINAL"
    return "UNSAFE"


def fixed_point_adjoint_residual(
    phi: Callable[[PyTree], PyTree],
    fixed_point: PyTree,
    objective_cotangent: PyTree,
    num_repeats: int = 4,
    check_fixed_point: bool = False,
    thresholds: ThresholdInput | None = None,
    stability: StabilityInput | None = None,
) -> FixedPointAdjointResidualReport:
    """Measure a loop-cut adjoint's objective-aware equation residual.

    ``phi``, ``fixed_point``, and ``objective_cotangent`` may use any JAX
    PyTree structure.  Norms combine all array leaves into a single L2 norm.
    Repeated pullbacks are diagnostics only; they are not assumed to form a
    convergent Neumann series.

    No verdict is produced unless caller-calibrated ``thresholds`` are given.
    A known repelling map is always reported ``UNSAFE`` when thresholds are
    requested, regardless of residual size.  ``stability`` accepts one of the
    :class:`FixedPointStability` values or a non-negative spectral radius.
    """

    try:
        repeats = operator.index(num_repeats)
    except TypeError as exc:
        raise TypeError("num_repeats must be an integer") from exc
    if repeats < 1:
        raise ValueError("num_repeats must be at least 1")

    calibrated_thresholds = _normalize_thresholds(thresholds)
    known_stability = _normalize_stability(stability)

    cotangent_norm = _tree_l2_norm(objective_cotangent)
    if cotangent_norm == 0.0:
        raise ValueError("objective_cotangent is identically zero; residual is undefined")

    phi_value, pullback = jax.vjp(phi, fixed_point)
    repeated_raw: list[float] = []
    repeated_relative: list[float] = []
    cotangent = objective_cotangent
    for _ in range(repeats):
        (cotangent,) = pullback(cotangent)
        raw_norm = _tree_l2_norm(cotangent)
        repeated_raw.append(raw_norm)
        repeated_relative.append(raw_norm / cotangent_norm)

    fixed_point_residual = None
    if check_fixed_point:
        try:
            difference = jax.tree_util.tree_map(
                lambda actual, expected: jnp.asarray(actual) - jnp.asarray(expected),
                phi_value,
                fixed_point,
            )
        except ValueError as exc:
            raise ValueError("phi(fixed_point) must have the fixed point's PyTree structure") from exc
        fixed_point_residual = _tree_l2_norm(difference)

    relative_residual = repeated_relative[0]
    return FixedPointAdjointResidualReport(
        relative_residual=relative_residual,
        residual_norm=repeated_raw[0],
        cotangent_norm=cotangent_norm,
        repeated_residual_norms=tuple(repeated_raw),
        repeated_relative_norms=tuple(repeated_relative),
        fixed_point_residual=fixed_point_residual,
        verdict=_verdict(relative_residual, calibrated_thresholds, known_stability),
        stability=known_stability,
    )


__all__ = [
    "FixedPointAdjointResidualReport",
    "FixedPointStability",
    "ResidualThresholds",
    "fixed_point_adjoint_residual",
]
