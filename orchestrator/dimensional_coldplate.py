# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Run a dimensional, sealed-water 2-D cold-plate example.

The solver is nondimensional and two-dimensional. This example makes that
mapping explicit: heat load is converted using an assumed out-of-plane depth,
and reported thermal resistance is for that finite depth. Properties are
constant (water and aluminium near 25 C), radiation/contact resistance are
neglected, the top wall is an isothermal coolant sink, and all other external
walls are adiabatic except the chip footprint.

Usage: python dimensional_coldplate.py --N 24
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tesseract_jax import apply_tesseract

from pipeline import ColdPlate, Params


@dataclass(frozen=True)
class PhysicalCase:
    width_m: float = 0.005
    height_m: float = 0.005
    depth_m: float = 0.002
    chip_width_m: float = 0.002
    heat_load_W: float = 1.0
    sink_temperature_K: float = 298.15
    reference_delta_T_K: float = 10.0
    gravity_m_s2: float = 9.81
    water_density_kg_m3: float = 997.0
    water_dynamic_viscosity_Pa_s: float = 8.90e-4
    water_cp_J_kgK: float = 4181.0
    water_conductivity_W_mK: float = 0.606
    water_expansion_1_K: float = 2.57e-4
    aluminium_conductivity_W_mK: float = 205.0
    blocked_solid_permeability_m2: float = 1.0e-12

    def validate(self) -> None:
        if not np.isclose(self.width_m, self.height_m):
            raise ValueError("the current MAC grid uses square cells; width must equal height")
        if not 0.0 < self.chip_width_m <= self.width_m:
            raise ValueError("chip width must lie in (0, plate width]")
        if min(asdict(self).values()) <= 0.0:
            raise ValueError("all physical inputs must be positive")


def nondimensionalise(case: PhysicalCase) -> dict[str, float]:
    """Return the exact scales and nondimensional inputs used by the solver."""
    case.validate()
    thermal_diffusivity = (
        case.water_conductivity_W_mK
        / (case.water_density_kg_m3 * case.water_cp_J_kgK)
    )
    kinematic_viscosity = (
        case.water_dynamic_viscosity_Pa_s / case.water_density_kg_m3
    )
    heat_flux = case.heat_load_W / (case.chip_width_m * case.depth_m)
    L, dT = case.width_m, case.reference_delta_T_K
    return {
        "thermal_diffusivity_m2_s": thermal_diffusivity,
        "kinematic_viscosity_m2_s": kinematic_viscosity,
        "velocity_scale_m_s": thermal_diffusivity / L,
        "heat_flux_W_m2": heat_flux,
        "Pr": kinematic_viscosity / thermal_diffusivity,
        "Ra": (
            case.gravity_m_s2 * case.water_expansion_1_K * dT * L**3
            / (kinematic_viscosity * thermal_diffusivity)
        ),
        "q_chip": heat_flux * L / (case.water_conductivity_W_mK * dT),
        "chip_frac": case.chip_width_m / L,
        "k_fluid": 1.0,
        "k_solid": (
            case.aluminium_conductivity_W_mK / case.water_conductivity_W_mK
        ),
        "alpha_max": L**2 / case.blocked_solid_permeability_m2,
    }


def finned_layout(N: int) -> np.ndarray:
    """Simple aluminium base plus four vertical fins; one means solid."""
    rho = np.zeros((N, N), dtype=np.float64)
    base = max(1, round(0.10 * N))
    rho[:base, :] = 1.0
    fin_width = max(1, round(0.04 * N))
    for frac in (0.2, 0.4, 0.6, 0.8):
        i = min(N - fin_width, round(frac * N - 0.5 * fin_width))
        rho[base : round(0.70 * N), i : i + fin_width] = 1.0
    return rho


def base_layout(N: int) -> np.ndarray:
    """Aluminium base only, used as the preregistered engineering baseline."""
    rho = np.zeros((N, N), dtype=np.float64)
    rho[: max(1, round(0.10 * N)), :] = 1.0
    return rho


def discretized_chip(case: PhysicalCase, N: int) -> dict[str, float | np.ndarray]:
    """Map the finite chip to cells while preserving its total heat exactly."""
    if N < 4:
        raise ValueError("N must be at least 4")
    p = Params(Nx=N, Ny=N, chip_frac=case.chip_width_m / case.width_m)
    mask = p.chip_mask().astype(bool)
    effective_width = case.width_m * float(np.mean(mask))
    if effective_width <= 0.0:
        raise ValueError("the chip footprint selected no grid cells")
    flux = case.heat_load_W / (effective_width * case.depth_m)
    return {
        "mask": mask,
        "selected_cells": int(np.sum(mask)),
        "effective_width_m": effective_width,
        "heat_flux_W_m2": flux,
        "represented_heat_load_W": flux * effective_width * case.depth_m,
    }


def run(case: PhysicalCase, N: int = 24, verbose: bool = False) -> dict:
    nd = nondimensionalise(case)
    chip = discretized_chip(case, N)
    solver_q_chip = (
        float(chip["heat_flux_W_m2"]) * case.width_m
        / (case.water_conductivity_W_mK * case.reference_delta_T_K)
    )
    p = Params(
        Nx=N, Ny=N, Ra=nd["Ra"], Pr=nd["Pr"], inertia=1.0,
        q_chip=solver_q_chip, chip_frac=nd["chip_frac"], bc_mode=0.0,
        filter_radius=1.0, beta=8.0, penal=3.0,
        k_solid=nd["k_solid"], k_fluid=nd["k_fluid"],
        alpha_max=nd["alpha_max"],
    )

    def solve_layout(cp: ColdPlate, name: str, rho: np.ndarray) -> dict:
        material = cp.material(rho)
        T, info = cp.solve_coupled(material["alpha"], material["k"], tol=1e-9)
        flow = apply_tesseract(
            cp._t["fluid"],
            {
                "alpha": material["alpha"],
                "T": T,
                "Ra": p.Ra,
                "Pr": p.Pr,
                "inertia": p.inertia,
            },
        )
        mask = np.asarray(chip["mask"], dtype=bool)
        T_array = np.asarray(T)
        k_array = np.asarray(material["k"])
        # Neumann data prescribe the inward heat flux. Reconstruct the wall
        # value from the first cell centre at a half-cell distance.
        wall_T = T_array[0, mask] + p.q_chip * (0.5 / N) / k_array[0, mask]
        cell_mean_nd = float(np.mean(T_array[0, mask]))
        wall_mean_nd = float(np.mean(wall_T))
        rise = case.reference_delta_T_K * wall_mean_nd
        velocity_scale = nd["velocity_scale_m_s"]
        return {
            "name": name,
            "solid_volume_fraction": float(np.mean(np.asarray(material["rho_phys"]))),
            "solver": info,
            "chip_cell_mean_nondimensional": cell_mean_nd,
            "chip_wall_mean_nondimensional": wall_mean_nd,
            "chip_wall_mean_temperature_K": case.sink_temperature_K + rise,
            "chip_wall_mean_temperature_C": case.sink_temperature_K + rise - 273.15,
            "temperature_rise_K": rise,
            "thermal_resistance_K_W": rise / case.heat_load_W,
            "max_abs_u_m_s": float(np.max(np.abs(np.asarray(flow["u"])))) * velocity_scale,
            "max_abs_v_m_s": float(np.max(np.abs(np.asarray(flow["v"])))) * velocity_scale,
        }

    with ColdPlate(params=p, verbose=verbose) as cp:
        baseline = solve_layout(cp, "aluminium_base_only", base_layout(N))
        finned = solve_layout(cp, "base_plus_four_fins", finned_layout(N))

    baseline_rth = float(baseline["thermal_resistance_K_W"])
    finned_rth = float(finned["thermal_resistance_K_W"])
    return {
        "assumptions": {
            "model": "steady 2-D Boussinesq; heat load uses stated out-of-plane depth",
            "fluid": "constant-property water near 25 C",
            "solid": "isotropic aluminium represented by Brinkman penalisation",
            "momentum": "steady nonlinear Navier-Stokes-Brinkman (inertia=1)",
            "boundaries": "chip heat flux at bottom; isothermal top; other walls adiabatic",
            "omissions": "radiation, contact resistance, phase change, and temperature-dependent properties",
        },
        "physical_inputs": asdict(case),
        "nondimensional_inputs": {**nd, "q_chip_used": solver_q_chip, "inertia": 1.0},
        "grid": {
            "Nx": N,
            "Ny": N,
            "chip_cells": chip["selected_cells"],
            "effective_chip_width_m": chip["effective_width_m"],
            "represented_heat_load_W": chip["represented_heat_load_W"],
        },
        "layouts": {"baseline": baseline, "finned": finned},
        "finned_thermal_resistance_reduction_percent": (
            100.0 * (baseline_rth - finned_rth) / baseline_rth
        ),
    }


def main(N=24, out="results/dimensional_coldplate.json", verbose=False):
    result = run(PhysicalCase(), N, verbose)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(f"Ra={result['nondimensional_inputs']['Ra']:.4g}, "
          f"Pr={result['nondimensional_inputs']['Pr']:.3f}")
    for name, row in result["layouts"].items():
        print(f"{name}: chip wall={row['chip_wall_mean_temperature_C']:.2f} C, "
              f"rise={row['temperature_rise_K']:.2f} K, "
              f"Rth={row['thermal_resistance_K_W']:.4f} K/W, "
              f"converged={row['solver']['ok']}")
    print("finned Rth change: "
          f"{result['finned_thermal_resistance_reduction_percent']:.2f}% reduction")
    print(f"wrote {path}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=24)
    ap.add_argument("--out", default="results/dimensional_coldplate.json")
    ap.add_argument("--verbose", action="store_true")
    main(**vars(ap.parse_args()))
