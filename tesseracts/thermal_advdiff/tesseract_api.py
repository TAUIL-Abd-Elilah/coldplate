# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Tesseract solving steady advection-diffusion for temperature (JAX).

Given a velocity field and a conductivity map, solve

    div(u T) - div(k grad T) = 0

with a chip heat flux on part of the bottom wall, a cold sink on the top wall,
and adiabatic sides. The equation is linear in T, so

    A(u, v, k) T = b(q_chip)

Differentiation strategy -- deliberately different from the C++ block it
composes with:

  * the linear algebra is a sparse LU (the operator is a 5-point stencil,
    ~0.8% dense; a dense solve is ~75x slower at 64x64);
  * every *parameter* derivative comes from JAX autodiff of the residual.
    That matters because the upwind switch and the face-averaged conductivity
    are exactly the kind of thing that is annoying and error-prone to
    differentiate by hand.

So the implicit-function-theorem step is explicit here, while the messy
partials are autodiff'd:

    JVP   A dT   = -(dR/dtheta) dtheta
    VJP   lam    = A^{-T} Tbar ,  dtheta_bar = -(dR/dtheta)^T lam
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------
# residual (JAX) -- the single source of truth for the physics
# --------------------------------------------------------------------------


def chip_mask(Nx: int, chip_frac: float) -> jnp.ndarray:
    i = jnp.arange(Nx)
    lo = 0.5 * (1.0 - chip_frac) * Nx
    hi = 0.5 * (1.0 + chip_frac) * Nx
    return ((i >= lo) & (i < hi)).astype(jnp.float64)


def _face_conductivities(k):
    """Face-centred conductivities, full-size arrays with walls padded.

    Shared by the diffusion term and the Peclet weighting so the two can never
    drift apart. Wall faces are filled by edge replication; wall-normal
    velocities vanish there, so those entries never affect the answer.
    """
    Ny, Nx = k.shape
    kx = jnp.zeros((Ny, Nx + 1))
    kx = kx.at[:, 1:Nx].set(0.5 * (k[:, 0 : Nx - 1] + k[:, 1:Nx]))
    kx = kx.at[:, 0].set(k[:, 0]).at[:, Nx].set(k[:, Nx - 1])
    ky = jnp.zeros((Ny + 1, Nx))
    ky = ky.at[1:Ny, :].set(0.5 * (k[0 : Ny - 1, :] + k[1:Ny, :]))
    ky = ky.at[0, :].set(k[0, :]).at[Ny, :].set(k[Ny - 1, :])
    return kx, ky


def residual(T, u, v, k, q_chip: float, chip_frac: float,
             bc_mode: float = 0.0, t_hot: float = 1.0):
    """R(T, u, v, k) = div(uT) - div(k grad T) - source. Linear in T.

    bc_mode selects the wall conditions:
      0  cold-plate: a chip heat flux over part of the wall (the design problem)
      1  Rayleigh-Benard: an isothermal hot wall at t_hot (the benchmark)
      2  de Vahl Davis: left wall at t_hot, right wall at 0, horizontal
         walls adiabatic (the differentially heated cavity benchmark)

    The benchmark modes check both the classical critical Rayleigh number and
    the nonlinear de Vahl Davis reference solution.
    """
    Ny, Nx = T.shape
    h = 1.0 / Nx

    # Advection, conservative form, with a smooth Peclet-weighted face value.
    #
    # A hard upwind switch -- where(u > 0, T_L, T_R) -- makes this residual
    # non-differentiable wherever a face velocity crosses zero. That is fatal
    # here for two reasons: Newton limit-cycles at ~1e-2 as faces flip their
    # upwind direction, and the kink sits directly in the gradient path.
    #
    # Instead blend upwind -> central as the local cell Peclet number falls.
    # Where |Pe| is small, diffusion dominates and the advection scheme is
    # immaterial, so this costs nothing physically; where |Pe| is large it
    # recovers pure upwind. Smooth everywhere, which is what the fixed point
    # and the adjoint both need.
    kf_x, kf_y = _face_conductivities(k)
    w_x = 0.5 * (1.0 + jnp.tanh(0.5 * u * h / jnp.maximum(kf_x, 1e-12)))
    w_y = 0.5 * (1.0 + jnp.tanh(0.5 * v * h / jnp.maximum(kf_y, 1e-12)))

    Tp_x = jnp.concatenate([T[:, 0:1], T, T[:, Nx - 1 : Nx]], axis=1)
    Fx = u * (w_x * Tp_x[:, 0 : Nx + 1] + (1.0 - w_x) * Tp_x[:, 1 : Nx + 2])
    Tp_y = jnp.concatenate([T[0:1, :], T, T[Ny - 1 : Ny, :]], axis=0)
    Fy = v * (w_y * Tp_y[0 : Ny + 1, :] + (1.0 - w_y) * Tp_y[1 : Ny + 2, :])
    adv = (Fx[:, 1 : Nx + 1] - Fx[:, 0:Nx]) / h + (Fy[1 : Ny + 1, :] - Fy[0:Ny, :]) / h

    # diffusion, arithmetic-mean face conductivities
    kx = 0.5 * (k[:, 0 : Nx - 1] + k[:, 1:Nx])
    ky = 0.5 * (k[0 : Ny - 1, :] + k[1:Ny, :])
    qx = jnp.zeros((Ny, Nx + 1)).at[:, 1:Nx].set(-kx * (T[:, 1:Nx] - T[:, 0 : Nx - 1]) / h)
    # Differentially heated cavity (mode 2): hot left wall, cold right wall.
    # In modes 0/1 these remain the original adiabatic side conditions.
    side_heated = bc_mode > 1.5
    qx = qx.at[:, 0].set(
        jnp.where(side_heated, -k[:, 0] * (T[:, 0] - t_hot) / (0.5 * h), 0.0)
    )
    qx = qx.at[:, Nx].set(
        jnp.where(side_heated, -k[:, Nx - 1] * (0.0 - T[:, Nx - 1]) / (0.5 * h), 0.0)
    )
    qy = jnp.zeros((Ny + 1, Nx)).at[1:Ny, :].set(-ky * (T[1:Ny, :] - T[0 : Ny - 1, :]) / h)
    # Bottom wall: chip flux in mode 0, hot Dirichlet in mode 1, and adiabatic
    # in mode 2. Values are the +y component of heat flux at that face.
    qy = qy.at[0, :].set(
        jnp.where(
            (bc_mode > 0.5) & (bc_mode < 1.5),
            -k[0, :] * (T[0, :] - t_hot) / (0.5 * h),
            jnp.where(bc_mode < 0.5, q_chip * chip_mask(Nx, chip_frac), 0.0),
        )
    )
    qy = qy.at[Ny, :].set(
        jnp.where(
            side_heated,
            0.0,
            -k[Ny - 1, :] * (0.0 - T[Ny - 1, :]) / (0.5 * h),
        )
    )  # cold top in modes 0/1; adiabatic in mode 2
    diff = (qx[:, 1 : Nx + 1] - qx[:, 0:Nx]) / h + (qy[1 : Ny + 1, :] - qy[0:Ny, :]) / h

    return adv + diff


# --------------------------------------------------------------------------
# sparse assembly -- must reproduce `residual` exactly (see tests)
# --------------------------------------------------------------------------


def assemble(u, v, k, Nx: int, Ny: int, bc_mode: float = 0.0):
    """Build A and b such that A T = b is equivalent to residual(T,...) = 0."""
    h = 1.0 / Nx
    u, v, k = np.asarray(u), np.asarray(v), np.asarray(k)
    # Keep the grid shape here; `add` flattens. Ravelling early would break
    # broadcasting against the face arrays.
    cell = lambda j, i: j * Nx + i  # noqa: E731

    rows, cols, vals = [], [], []

    def add(r, c, val):
        rows.append(np.asarray(r).ravel())
        cols.append(np.asarray(c).ravel())
        vals.append(np.asarray(val).ravel())

    J, I = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing="ij")

    # Peclet-weighted face values, matching `residual` exactly. Each face
    # contributes to both of its neighbours now, rather than only the donor.
    kf_x = 0.5 * (k[:, 0 : Nx - 1] + k[:, 1:Nx])  # (Ny, Nx-1), interior x-faces
    kf_y = 0.5 * (k[0 : Ny - 1, :] + k[1:Ny, :])  # (Ny-1, Nx), interior y-faces

    # ---- advection across interior x-faces i = 1..Nx-1 ----
    jj, ii = np.meshgrid(np.arange(Ny), np.arange(1, Nx), indexing="ij")
    uf = u[jj, ii]
    w = 0.5 * (1.0 + np.tanh(0.5 * uf * h / np.maximum(kf_x, 1e-12)))
    L, R = cell(jj, ii - 1), cell(jj, ii)
    add(L, L, uf * w / h)  # east face of the left cell
    add(L, R, uf * (1.0 - w) / h)
    add(R, L, -uf * w / h)  # west face of the right cell
    add(R, R, -uf * (1.0 - w) / h)

    # ---- advection across interior y-faces j = 1..Ny-1 ----
    jj, ii = np.meshgrid(np.arange(1, Ny), np.arange(Nx), indexing="ij")
    vf = v[jj, ii]
    w = 0.5 * (1.0 + np.tanh(0.5 * vf * h / np.maximum(kf_y, 1e-12)))
    B, T_ = cell(jj - 1, ii), cell(jj, ii)
    add(B, B, vf * w / h)
    add(B, T_, vf * (1.0 - w) / h)
    add(T_, B, -vf * w / h)
    add(T_, T_, -vf * (1.0 - w) / h)

    # ---- diffusion across interior x-faces ----
    jj, ii = np.meshgrid(np.arange(Ny), np.arange(1, Nx), indexing="ij")
    kf = 0.5 * (k[jj, ii - 1] + k[jj, ii])
    L, R = cell(jj, ii - 1), cell(jj, ii)
    add(L, L, kf / h**2)
    add(L, R, -kf / h**2)
    add(R, L, -kf / h**2)
    add(R, R, kf / h**2)

    # ---- diffusion across interior y-faces ----
    jj, ii = np.meshgrid(np.arange(1, Ny), np.arange(Nx), indexing="ij")
    kf = 0.5 * (k[jj - 1, ii] + k[jj, ii])
    B, T_ = cell(jj - 1, ii), cell(jj, ii)
    add(B, B, kf / h**2)
    add(B, T_, -kf / h**2)
    add(T_, B, -kf / h**2)
    add(T_, T_, kf / h**2)

    # ---- cold top wall (Dirichlet T=0 at half-cell distance), modes 0/1 ----
    ii = np.arange(Nx)
    if bc_mode < 1.5:
        add(cell(np.full(Nx, Ny - 1), ii), cell(np.full(Nx, Ny - 1), ii),
            2.0 * k[Ny - 1, :] / h**2)

    # ---- hot bottom wall, Rayleigh-Benard mode only ----
    # The chip mode puts a Neumann flux there, which loads b rather than A.
    if 0.5 < bc_mode < 1.5:
        add(cell(np.zeros(Nx, dtype=int), ii), cell(np.zeros(Nx, dtype=int), ii),
            2.0 * k[0, :] / h**2)

    # ---- de Vahl Davis side walls, mode 2 ----
    if bc_mode > 1.5:
        jj = np.arange(Ny)
        add(cell(jj, np.zeros(Ny, dtype=int)),
            cell(jj, np.zeros(Ny, dtype=int)), 2.0 * k[:, 0] / h**2)
        add(cell(jj, np.full(Ny, Nx - 1, dtype=int)),
            cell(jj, np.full(Ny, Nx - 1, dtype=int)),
            2.0 * k[:, Nx - 1] / h**2)

    A = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(Nx * Ny, Nx * Ny),
    ).tocsc()
    A.sum_duplicates()
    return A


def rhs(Nx: int, Ny: int, q_chip: float, chip_frac: float,
        bc_mode: float = 0.0, t_hot: float = 1.0, k=None):
    """Loads from the inhomogeneous wall conditions."""
    h = 1.0 / Nx
    b = np.zeros((Ny, Nx))
    if bc_mode > 1.5:
        # hot left wall: 2 k t_hot / h^2 into the first column; right is zero
        b[:, 0] = 2.0 * np.asarray(k)[:, 0] * t_hot / h**2
    elif bc_mode > 0.5:
        # isothermal hot wall: 2 k t_hot / h^2 into the first row
        b[0, :] = 2.0 * np.asarray(k)[0, :] * t_hot / h**2
    else:
        b[0, :] = q_chip * np.asarray(chip_mask(Nx, chip_frac)) / h
    return b.ravel()


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------


class InputSchema(BaseModel):
    u: Differentiable[Array[(None, None), Float64]] = Field(
        description="x-velocity on vertical faces, shape (Ny, Nx+1)."
    )
    v: Differentiable[Array[(None, None), Float64]] = Field(
        description="y-velocity on horizontal faces, shape (Ny+1, Nx)."
    )
    k: Differentiable[Array[(None, None), Float64]] = Field(
        description="Cell-centred thermal conductivity, shape (Ny, Nx)."
    )
    q_chip: Float64 = Field(default=1.0, description="Chip heat flux into the bottom wall.")
    chip_frac: Float64 = Field(default=0.4, description="Chip width as a fraction of the wall.")
    bc_mode: Float64 = Field(
        default=0.0,
        description="Wall mode: 0 = chip heat flux on bottom/cold top, "
        "1 = isothermal hot bottom/cold top (Rayleigh-Benard), "
        "2 = hot left/cold right/adiabatic horizontal walls (de Vahl Davis).",
    )
    t_hot: Float64 = Field(default=1.0, description="Hot wall temperature for bc_mode 1 or 2.")


class OutputSchema(BaseModel):
    T: Differentiable[Array[(None, None), Float64]] = Field(
        description="Cell-centred temperature, shape (Ny, Nx)."
    )


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------


_CACHE: OrderedDict[str, tuple] = OrderedDict()
_CACHE_MAX = 4


def _solve(inputs: InputSchema):
    """Assemble, factorise and solve -- reusing the LU when the state repeats.

    This cache is what makes the Krylov solves affordable. A GMRES iteration
    calls jvp (or vjp) many times with identical (u, v, k) and only the tangent
    changing, so without it every matvec would redo the assembly and the sparse
    factorisation, which dominate the cost of this component.
    """
    u = np.ascontiguousarray(inputs.u, dtype=np.float64)
    v = np.ascontiguousarray(inputs.v, dtype=np.float64)
    k = np.ascontiguousarray(inputs.k, dtype=np.float64)
    q, cf = float(inputs.q_chip), float(inputs.chip_frac)
    bc, th = float(inputs.bc_mode), float(inputs.t_hot)

    key = hashlib.blake2b(
        u.tobytes() + v.tobytes() + k.tobytes()
        + np.array([q, cf, bc, th], dtype=np.float64).tobytes(),
        digest_size=16,
    ).hexdigest()
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]

    Ny, Nx = k.shape
    lu = spla.splu(assemble(u, v, k, Nx, Ny, bc).tocsc())
    b = rhs(Nx, Ny, q, cf, bc, th, k)
    entry = (lu.solve(b).reshape(Ny, Nx), lu, Nx, Ny)

    _CACHE[key] = entry
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return entry


def apply(inputs: InputSchema) -> OutputSchema:
    T, _, _, _ = _solve(inputs)
    return OutputSchema(T=T)


def abstract_eval(abstract_inputs):
    return {"T": ShapeDType(shape=tuple(abstract_inputs.k.shape), dtype="float64")}


def _residual_wrt_params(T, inputs):
    """Closure R(u, v, k) at fixed T, for JAX to differentiate."""
    q, cf = float(inputs.q_chip), float(inputs.chip_frac)
    bc, th = float(inputs.bc_mode), float(inputs.t_hot)
    return lambda u, v, k: residual(jnp.asarray(T), u, v, k, q, cf, bc, th)


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    T, lu, Nx, Ny = _solve(inputs)
    f = _residual_wrt_params(T, inputs)
    prim = (jnp.asarray(inputs.u), jnp.asarray(inputs.v), jnp.asarray(inputs.k))
    tang = tuple(
        jnp.asarray(tangent_vector[n]) if n in jvp_inputs else jnp.zeros_like(p)
        for n, p in zip(("u", "v", "k"), prim)
    )
    _, dR = jax.jvp(f, prim, tang)
    dT = lu.solve(-np.asarray(dR).ravel()).reshape(Ny, Nx)
    return {k: dT for k in jvp_outputs}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    T, lu, Nx, Ny = _solve(inputs)
    Tbar = np.asarray(cotangent_vector["T"], dtype=np.float64).ravel()
    # transpose solve against the same factorisation
    lam = lu.solve(Tbar, trans="T").reshape(Ny, Nx)

    f = _residual_wrt_params(T, inputs)
    _, vjp_fn = jax.vjp(f, jnp.asarray(inputs.u), jnp.asarray(inputs.v), jnp.asarray(inputs.k))
    gu, gv, gk = vjp_fn(-jnp.asarray(lam))  # minus: dT/dtheta = -A^{-1} dR/dtheta

    out = {"u": np.asarray(gu), "v": np.asarray(gv), "k": np.asarray(gk)}
    return {n: out[n] for n in vjp_inputs}
