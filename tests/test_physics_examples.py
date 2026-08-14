# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Container-free checks for the benchmark observables and SI mapping."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "orchestrator"))

from benchmark_de_vahl_davis import cavity_metrics  # noqa: E402
from dimensional_coldplate import (  # noqa: E402
    PhysicalCase,
    base_layout,
    discretized_chip,
    finned_layout,
    nondimensionalise,
)


def test_cavity_nusselt_definition_recovers_pure_conduction():
    N = 16
    x = (np.arange(N) + 0.5) / N
    T = np.tile(1.0 - x, (N, 1))
    u = np.zeros((N, N + 1))
    v = np.zeros((N + 1, N))
    metrics = cavity_metrics(T, u, v)
    assert abs(metrics["Nu_mean"] - 1.0) < 1e-14
    assert metrics["u_max"] == metrics["v_max"] == 0.0


def test_dimensional_mapping_matches_definitions():
    case = PhysicalCase()
    nd = nondimensionalise(case)
    a = case.water_conductivity_W_mK / (
        case.water_density_kg_m3 * case.water_cp_J_kgK
    )
    nu = case.water_dynamic_viscosity_Pa_s / case.water_density_kg_m3
    expected_flux = case.heat_load_W / (case.chip_width_m * case.depth_m)
    assert np.isclose(nd["Pr"], nu / a)
    assert np.isclose(nd["velocity_scale_m_s"], a / case.width_m)
    assert np.isclose(nd["heat_flux_W_m2"], expected_flux)
    assert np.isclose(
        nd["q_chip"],
        expected_flux * case.width_m
        / (case.water_conductivity_W_mK * case.reference_delta_T_K),
    )
    assert 1.0e4 < nd["Ra"] < 1.0e5
    assert 5.0 < nd["Pr"] < 8.0


def test_finned_layout_is_binary_and_connected_to_base():
    rho = finned_layout(24)
    baseline = base_layout(24)
    assert set(np.unique(rho)) == {0.0, 1.0}
    assert np.all(rho[:2, :] == 1.0)
    assert np.all(rho >= baseline)
    assert 0.1 < np.mean(rho) < 0.3


def test_discretized_chip_preserves_requested_heat_load():
    case = PhysicalCase()
    chip = discretized_chip(case, 24)
    assert chip["selected_cells"] > 0
    assert np.isclose(chip["represented_heat_load_W"], case.heat_load_W)
    assert np.isclose(
        chip["heat_flux_W_m2"] * chip["effective_width_m"] * case.depth_m,
        case.heat_load_W,
    )
