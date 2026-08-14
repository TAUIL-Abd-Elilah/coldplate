# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Mode-2 parity between the JAX and Fortran/Enzyme thermal backends."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def thermal_backends(_paths):
    from conftest import ROOT, load_tesseract_api

    lib = ROOT / "tesseracts" / "thermal_fortran" / "lib" / "libthermal_ad.so"
    if not lib.exists():
        pytest.skip("Fortran/Enzyme library is built in its toolchain image, not on this host")
    try:
        return load_tesseract_api("thermal_advdiff"), load_tesseract_api("thermal_fortran")
    except OSError as exc:
        pytest.skip(f"Fortran/Enzyme library is not loadable on this platform: {exc}")


def test_mode2_zero_flow_conduction_matches_between_backends(thermal_backends):
    jax_api, fortran_api = thermal_backends
    N = 12
    u = np.zeros((N, N + 1))
    v = np.zeros((N + 1, N))
    k = np.ones((N, N))
    expected = np.tile(1.0 - (np.arange(N) + 0.5) / N, (N, 1))

    def solve(api):
        inputs = api.InputSchema(
            # Nonzero on purpose: mode 2 must make the bottom adiabatic rather
            # than falling through to the chip-flux condition.
            u=u, v=v, k=k, q_chip=13.0, chip_frac=0.4,
            bc_mode=2.0, t_hot=1.0,
        )
        return np.asarray(api.apply(inputs).T)

    T_jax, T_fortran = solve(jax_api), solve(fortran_api)
    assert np.max(np.abs(T_jax - expected)) < 3e-14
    assert np.max(np.abs(T_fortran - expected)) < 3e-12
    assert np.max(np.abs(T_fortran - T_jax)) < 3e-12


def test_mode2_residual_matches_between_backends(thermal_backends):
    jax_api, fortran_api = thermal_backends
    N = 9
    rng = np.random.default_rng(42)
    T = rng.normal(size=(N, N))
    u = rng.normal(size=(N, N + 1))
    u[:, (0, -1)] = 0.0
    v = rng.normal(size=(N + 1, N))
    v[(0, -1), :] = 0.0
    k = rng.uniform(0.2, 2.0, size=(N, N))
    r_jax = np.asarray(jax_api.residual(T, u, v, k, 0.0, 0.0, 2.0, 1.0))
    r_fortran = fortran_api.residual(T, u, v, k, 0.0, 0.0, N, N, 2.0, 1.0)
    scale = max(float(np.max(np.abs(r_jax))), 1.0)
    assert np.max(np.abs(r_fortran.reshape(N, N) - r_jax)) < 2e-12 * scale
