# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Tesseract solving steady advection-diffusion for temperature (Fortran + Enzyme).

A drop-in replacement for `thermal_advdiff`: same schema, same physics, same
answers. The difference is where the derivatives come from.

    thermal_advdiff   residual in JAX      -> jax.jvp / jax.vjp
    thermal_fortran   residual in Fortran  -> Enzyme LLVM pass  (this module)

Nothing here differentiates anything by hand and no AD library is imported.
Every derivative is produced by a compiler pass over the Fortran source.

Two things are worth pointing out about how that gets used:

* The operator dR/dT is recovered *exactly* from nine Enzyme JVPs. The stencil
  is five-point, so two cells whose (i, j) agree modulo 3 never appear in each
  other's equations; seeding all cells of one colour at once therefore reads
  off one entry of every row without interference. Nine colours cover the 3x3
  residue classes. This is the standard sparse-Jacobian-by-colouring trick, and
  it turns an N^2-column Jacobian into nine calls.

* The implicit function theorem step is explicit here, exactly as in the JAX
  block: solve A T = b forward, and A^T for the adjoint, with Enzyme supplying
  the parameter partials.
"""

from __future__ import annotations

import ctypes
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

# --------------------------------------------------------------------------
# shared library
# --------------------------------------------------------------------------

_LIB_PATH = Path(__file__).parent / "lib" / "libthermal_ad.so"
if not _LIB_PATH.exists():
    _LIB_PATH = Path("/tesseract/lib/libthermal_ad.so")
_lib = ctypes.CDLL(str(_LIB_PATH))

_dp = ctypes.POINTER(ctypes.c_double)
_ci, _cd = ctypes.c_int, ctypes.c_double

_lib.th_forward.restype = None
_lib.th_forward.argtypes = [_ci, _ci, _dp, _dp, _dp, _dp, _cd, _cd, _dp]
_lib.th_jvp.restype = None
_lib.th_jvp.argtypes = [_ci, _ci, _dp, _dp, _dp, _dp, _dp, _dp, _dp, _dp, _cd, _cd, _dp, _dp]
_lib.th_vjp.restype = None
_lib.th_vjp.argtypes = [_ci, _ci, _dp, _dp, _dp, _dp, _dp, _dp, _dp, _dp, _cd, _cd, _dp, _dp]


def _p(a: np.ndarray):
    return a.ctypes.data_as(_dp)


def _c(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64).ravel()


# --------------------------------------------------------------------------
# Enzyme-backed primitives
# --------------------------------------------------------------------------


def residual(T, u, v, k, q_chip: float, chip_frac: float, Nx: int, Ny: int) -> np.ndarray:
    R = np.zeros(Nx * Ny, dtype=np.float64)
    _lib.th_forward(Nx, Ny, _p(_c(T)), _p(_c(u)), _p(_c(v)), _p(_c(k)),
                    _cd(q_chip), _cd(chip_frac), _p(R))
    return R


def _jvp(T, u, v, k, dT, du, dv, dk, q, cf, Nx, Ny) -> np.ndarray:
    R = np.zeros(Nx * Ny, dtype=np.float64)
    dR = np.zeros(Nx * Ny, dtype=np.float64)
    _lib.th_jvp(Nx, Ny, _p(_c(T)), _p(_c(dT)), _p(_c(u)), _p(_c(du)),
                _p(_c(v)), _p(_c(dv)), _p(_c(k)), _p(_c(dk)),
                _cd(q), _cd(cf), _p(R), _p(dR))
    return dR


def _vjp(T, u, v, k, Rb, q, cf, Nx, Ny):
    """Reverse mode. Enzyme accumulates into the shadow buffers, so zero them."""
    Tb = np.zeros(Nx * Ny, dtype=np.float64)
    ub = np.zeros(Ny * (Nx + 1), dtype=np.float64)
    vb = np.zeros((Ny + 1) * Nx, dtype=np.float64)
    kb = np.zeros(Nx * Ny, dtype=np.float64)
    R = np.zeros(Nx * Ny, dtype=np.float64)
    _lib.th_vjp(Nx, Ny, _p(_c(T)), _p(Tb), _p(_c(u)), _p(ub), _p(_c(v)), _p(vb),
                _p(_c(k)), _p(kb), _cd(q), _cd(cf), _p(R), _p(_c(Rb)))
    return Tb, ub, vb, kb


def assemble_by_colouring(T, u, v, k, q, cf, Nx: int, Ny: int) -> sp.csc_matrix:
    """Recover dR/dT exactly using nine Enzyme JVPs.

    Cell (j, i) only appears in the equations of itself and its four
    neighbours, so cells sharing (i mod 3, j mod 3) never collide. Seeding one
    colour at a time therefore reads one entry of every row per call, and the
    row's column index is recoverable from the offset that produced it.
    """
    n = Nx * Ny
    zerosT = np.zeros(n)
    zu = np.zeros(Ny * (Nx + 1))
    zv = np.zeros((Ny + 1) * Nx)
    zk = np.zeros(n)

    J, I = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing="ij")
    rows, cols, vals = [], [], []

    for cj in range(3):
        for ci in range(3):
            seed = np.zeros(n)
            sel = (I % 3 == ci) & (J % 3 == cj)
            seed[(J[sel] * Nx + I[sel])] = 1.0

            col = _jvp(T, u, v, k, seed, zu, zv, zk, q, cf, Nx, Ny)

            # Row r received a contribution only from the seeded cell inside its
            # own stencil; find which one that is.
            for dj, di in ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)):
                jj = J + dj
                ii = I + di
                ok = (jj >= 0) & (jj < Ny) & (ii >= 0) & (ii < Nx)
                ok &= (ii % 3 == ci) & (jj % 3 == cj)
                if not ok.any():
                    continue
                r = (J[ok] * Nx + I[ok])
                c = (jj[ok] * Nx + ii[ok])
                rows.append(r)
                cols.append(c)
                vals.append(col[r])

    A = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsc()
    A.sum_duplicates()
    A.eliminate_zeros()
    return A


# --------------------------------------------------------------------------
# schemas -- identical to thermal_advdiff, so the two are interchangeable
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


class OutputSchema(BaseModel):
    T: Differentiable[Array[(None, None), Float64]] = Field(
        description="Cell-centred temperature, shape (Ny, Nx)."
    )


# --------------------------------------------------------------------------
# factorisation cache
# --------------------------------------------------------------------------

_CACHE: OrderedDict[str, tuple] = OrderedDict()
_CACHE_MAX = 4


def _solve(inputs: InputSchema):
    u = np.ascontiguousarray(inputs.u, dtype=np.float64)
    v = np.ascontiguousarray(inputs.v, dtype=np.float64)
    k = np.ascontiguousarray(inputs.k, dtype=np.float64)
    q, cf = float(inputs.q_chip), float(inputs.chip_frac)
    Ny, Nx = k.shape

    key = hashlib.blake2b(
        u.tobytes() + v.tobytes() + k.tobytes()
        + np.array([q, cf], dtype=np.float64).tobytes(),
        digest_size=16,
    ).hexdigest()
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]

    # R is affine in T: A = dR/dT and b = -R(T=0).
    A = assemble_by_colouring(np.zeros(Nx * Ny), u, v, k, q, cf, Nx, Ny)
    b = -residual(np.zeros(Nx * Ny), u, v, k, q, cf, Nx, Ny)
    lu = spla.splu(A)
    entry = (lu.solve(b).reshape(Ny, Nx), lu, Nx, Ny, u, v, k, q, cf)

    _CACHE[key] = entry
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return entry


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    T, _, _, _, *_ = _solve(inputs)
    return OutputSchema(T=T)


def abstract_eval(abstract_inputs):
    return {"T": ShapeDType(shape=tuple(abstract_inputs.k.shape), dtype="float64")}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    T, lu, Nx, Ny, u, v, k, q, cf = _solve(inputs)
    z = {"u": np.zeros(Ny * (Nx + 1)), "v": np.zeros((Ny + 1) * Nx), "k": np.zeros(Nx * Ny)}
    tan = {n: (_c(tangent_vector[n]) if n in jvp_inputs else z[n]) for n in ("u", "v", "k")}

    dR = _jvp(T, u, v, k, np.zeros(Nx * Ny), tan["u"], tan["v"], tan["k"], q, cf, Nx, Ny)
    dT = lu.solve(-dR).reshape(Ny, Nx)
    return {name: dT for name in jvp_outputs}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    T, lu, Nx, Ny, u, v, k, q, cf = _solve(inputs)
    Tbar = np.array(cotangent_vector["T"], dtype=np.float64).ravel()

    # dT/dtheta = -A^{-1} dR/dtheta, so seed the reverse pass with -A^{-T} Tbar.
    lam = -lu.solve(Tbar, trans="T")
    _, ub, vb, kb = _vjp(T, u, v, k, lam, q, cf, Nx, Ny)

    out = {
        "u": ub.reshape(Ny, Nx + 1),
        "v": vb.reshape(Ny + 1, Nx),
        "k": kb.reshape(Ny, Nx),
    }
    return {n: out[n] for n in vjp_inputs}
