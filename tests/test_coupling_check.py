# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coupling_check import coupling_gamma  # noqa: E402


@pytest.mark.parametrize("gain", [0.25, 1.25])
def test_gamma_is_exact_normalized_adjoint_residual_even_when_repelling(gain):
    # The rho>1 case is deliberate: its Neumann series diverges, but the exact
    # residual identity r_0 = Phi_x^T g still holds.
    phi = lambda x: gain * x  # noqa: E731
    x = jnp.array([0.0, 0.0])
    g = jnp.array([3.0, 4.0])
    report = coupling_gamma(phi, x, g)
    assert report.gamma == pytest.approx(abs(gain))


def test_coupling_thresholds_are_configurable_and_checked():
    phi = lambda x: 0.2 * x  # noqa: E731
    report = coupling_gamma(
        phi, jnp.ones(2), jnp.ones(2), safe_threshold=0.3, risky_threshold=0.5
    )
    assert report.verdict == "SAFE"
    with pytest.raises(ValueError):
        coupling_gamma(phi, jnp.ones(2), jnp.ones(2), safe_threshold=0.5,
                       risky_threshold=0.1)


def test_legacy_wrapper_uses_strict_repeat_validation():
    with pytest.raises(ValueError, match="at least 1"):
        coupling_gamma(lambda x: x, jnp.ones(2), jnp.ones(2), n_terms=0)
