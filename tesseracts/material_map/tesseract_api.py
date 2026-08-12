# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Tesseract mapping raw design variables to physical properties (PyTorch).

This is the third differentiation strategy in the pipeline: plain
`torch.autograd`. It is not a solver, but it is load-bearing -- it is what
turns an unconstrained optimisation variable into a manufacturable material
layout, and its Jacobian multiplies into every downstream gradient.

    rho_raw --filter--> rho_f --projection--> rho_phys --SIMP--> (k, alpha)

* density filter: convolution with a cone kernel of radius r. Imposes a minimum
  length scale and kills the checkerboard instability that unfiltered topology
  optimisation always falls into.
* Heaviside projection: pushes intermediate densities toward 0/1 so the final
  design is a real solid/fluid layout rather than a grey mush. Sharpness beta is
  meant to be continued upward during optimisation.
* property maps: conductivity interpolates with a SIMP exponent; the Brinkman
  drag uses a RAMP-style form that penalises intermediate density hard, so
  "half solid" is never a cheap way to fake a wall.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------
# core map, written once and reused by every endpoint
# --------------------------------------------------------------------------


def _cone_kernel(radius: float) -> torch.Tensor:
    """Standard linear-decay (cone) filter weights w = max(0, r - dist)."""
    r = int(np.ceil(radius))
    ax = torch.arange(-r, r + 1, dtype=torch.float64)
    dy, dx = torch.meshgrid(ax, ax, indexing="ij")
    w = torch.clamp(radius - torch.sqrt(dx**2 + dy**2), min=0.0)
    return w[None, None]


def material_map(
    rho_raw: torch.Tensor,
    filter_radius: float,
    beta: float,
    eta: float,
    penal: float,
    k_solid: float,
    k_fluid: float,
    alpha_max: float,
):
    """rho_raw -> (k, alpha, rho_phys). Pure torch so autograd handles it all."""
    w = _cone_kernel(filter_radius)
    pad = (w.shape[-1] - 1) // 2
    x = rho_raw[None, None]

    # Normalising by the same convolution applied to ones keeps the filter
    # unbiased at the walls, instead of darkening the border.
    num = torch.nn.functional.conv2d(
        torch.nn.functional.pad(x, (pad,) * 4, mode="constant", value=0.0), w
    )
    den = torch.nn.functional.conv2d(
        torch.nn.functional.pad(torch.ones_like(x), (pad,) * 4, mode="constant", value=0.0), w
    )
    rho_f = (num / den)[0, 0]

    # smoothed Heaviside projection about eta
    tb_eta = torch.tanh(torch.tensor(beta * eta))
    rho_phys = (tb_eta + torch.tanh(beta * (rho_f - eta))) / (
        tb_eta + torch.tanh(torch.tensor(beta * (1.0 - eta)))
    )

    k = k_fluid + (k_solid - k_fluid) * rho_phys**penal
    alpha = alpha_max * rho_phys / (1.0 + 8.0 * (1.0 - rho_phys))
    return k, alpha, rho_phys


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------


class InputSchema(BaseModel):
    rho_raw: Differentiable[Array[(None, None), Float64]] = Field(
        description="Raw design field in [0,1], shape (Ny, Nx)."
    )
    filter_radius: Float64 = Field(default=1.5, description="Cone filter radius in cells.")
    beta: Float64 = Field(default=1.0, description="Heaviside projection sharpness.")
    eta: Float64 = Field(default=0.5, description="Projection threshold.")
    penal: Float64 = Field(default=3.0, description="SIMP exponent for conductivity.")
    k_solid: Float64 = Field(default=1.0, description="Conductivity of solid.")
    k_fluid: Float64 = Field(default=0.02, description="Conductivity of fluid.")
    alpha_max: Float64 = Field(default=1.0e5, description="Brinkman drag inside solid.")


class OutputSchema(BaseModel):
    k: Differentiable[Array[(None, None), Float64]] = Field(
        description="Cell-centred thermal conductivity."
    )
    alpha: Differentiable[Array[(None, None), Float64]] = Field(
        description="Cell-centred Brinkman drag coefficient."
    )
    rho_phys: Differentiable[Array[(None, None), Float64]] = Field(
        description="Filtered and projected density, used for the volume constraint."
    )


_STATIC = ("filter_radius", "beta", "eta", "penal", "k_solid", "k_fluid", "alpha_max")
_OUTS = ("k", "alpha", "rho_phys")


def _params(inputs: InputSchema) -> dict[str, float]:
    return {n: float(getattr(inputs, n)) for n in _STATIC}


def _fn(inputs: InputSchema):
    p = _params(inputs)
    return lambda r: material_map(r, **p)


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    rho = torch.as_tensor(np.array(inputs.rho_raw, dtype=np.float64))
    k, alpha, rho_phys = _fn(inputs)(rho)
    return OutputSchema(
        k=k.detach().numpy(), alpha=alpha.detach().numpy(), rho_phys=rho_phys.detach().numpy()
    )


def abstract_eval(abstract_inputs):
    shape = tuple(abstract_inputs.rho_raw.shape)
    return {n: ShapeDType(shape=shape, dtype="float64") for n in _OUTS}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    rho = torch.as_tensor(np.array(inputs.rho_raw, dtype=np.float64))
    tan = torch.as_tensor(np.array(tangent_vector["rho_raw"], dtype=np.float64))
    _, jvp_out = torch.autograd.functional.jvp(_fn(inputs), rho, tan, create_graph=False)
    named = dict(zip(_OUTS, jvp_out))
    return {n: named[n].detach().numpy() for n in jvp_outputs}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    # np.array, not asarray: torch refuses to take a non-writable buffer, and
    # arrays arriving from JAX are read-only.
    rho = torch.as_tensor(np.array(inputs.rho_raw, dtype=np.float64)).requires_grad_(True)
    k, alpha, rho_phys = _fn(inputs)(rho)
    outs = {"k": k, "alpha": alpha, "rho_phys": rho_phys}

    # Sum the requested cotangent contractions and backprop once.
    total = sum(
        (outs[n] * torch.as_tensor(np.asarray(cotangent_vector[n], dtype=np.float64))).sum()
        for n in vjp_outputs
    )
    (grad,) = torch.autograd.grad(total, rho)
    return {"rho_raw": grad.detach().numpy()} if "rho_raw" in vjp_inputs else {}
