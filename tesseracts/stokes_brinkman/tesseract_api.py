# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Tesseract wrapping a C++/Eigen Brinkman-Boussinesq flow solver.

Derivatives here come from a hand-derived discrete adjoint, not from an AD
tool. With `inertia = 0` the system is linear in the unknown w = (u, v, p):

    A(alpha) w = b(T)

so the exact derivative of the solve is a transpose solve against the same
sparse factorisation:

    VJP   lam = A^{-T} wbar ,  then scatter lam against dA/dalpha and db/dT
    JVP   dw  = A^{-1} ( db/dT dT - (dA/dalpha dalpha) w )

With `inertia = 1` the convective acceleration (u.grad)u is included and the
block becomes steady Navier-Stokes, i.e. nonlinear in w:

    R(w) = A(alpha) w + N(w) - b(T) = 0

solved by damped Newton. The adjoint survives that intact, because N is
bilinear and involves neither alpha nor T: every scatter above is unchanged and
only the operator being inverted moves from A to the Jacobian at the converged
state, J = A + dN/dw. That is the practical point of deriving an adjoint by
hand rather than reaching for a tool -- the structure tells you exactly which
part of the derivation the nonlinearity touches, and it is a small part.

The factorisation is cached across endpoint calls keyed on the design field, so
within one optimisation step the forward solve, the tangent and the adjoint all
share a single LU. This is the component that "disagrees" with the JAX thermal
block it is composed with: different language, different memory layout,
different derivative strategy.
"""

from __future__ import annotations

import ctypes
import hashlib
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

# --------------------------------------------------------------------------
# shared library
# --------------------------------------------------------------------------

if sys.platform == "win32":
    # A checkout shared with WSL may contain both ignored build artefacts.
    # Never hand an ELF .so to Windows' loader (or a PE DLL to dlopen).
    _LIB_CANDIDATES = (
        Path(__file__).parent / "lib" / "stokes_brinkman.dll",
    )
else:
    _LIB_CANDIDATES = (
        Path(__file__).parent / "lib" / "libstokes_brinkman.so",
        Path("/tesseract/lib/libstokes_brinkman.so"),
    )
_LIB_PATH = next(
    (path for path in _LIB_CANDIDATES if path.exists()), _LIB_CANDIDATES[-1]
)
_lib = ctypes.CDLL(str(_LIB_PATH))

_dp = ctypes.POINTER(ctypes.c_double)

_lib.sb_create_ns.restype = ctypes.c_void_p
_lib.sb_create_ns.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double,
                              ctypes.c_double, ctypes.c_double, _dp]
_lib.sb_residual.restype = ctypes.c_double
_lib.sb_residual.argtypes = [ctypes.c_void_p, _dp]
_lib.sb_converged.restype = ctypes.c_int
_lib.sb_converged.argtypes = [ctypes.c_void_p]
_lib.sb_newton_iterations.restype = ctypes.c_int
_lib.sb_newton_iterations.argtypes = [ctypes.c_void_p]
_lib.sb_last_status.restype = ctypes.c_int
_lib.sb_last_status.argtypes = [ctypes.c_void_p]
_lib.sb_destroy.restype = None
_lib.sb_destroy.argtypes = [ctypes.c_void_p]
_lib.sb_apply.restype = ctypes.c_int
_lib.sb_apply.argtypes = [ctypes.c_void_p, _dp, _dp, _dp, _dp]
_lib.sb_jvp.restype = ctypes.c_int
_lib.sb_jvp.argtypes = [ctypes.c_void_p, _dp, _dp, _dp, _dp, _dp, _dp]
_lib.sb_vjp.restype = ctypes.c_int
_lib.sb_vjp.argtypes = [ctypes.c_void_p, _dp, _dp, _dp, _dp, _dp, _dp, _dp]


def _p(a: np.ndarray):
    return a.ctypes.data_as(_dp)


def _c(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


# --------------------------------------------------------------------------
# factorisation cache
# --------------------------------------------------------------------------

_CACHE: OrderedDict[str, Any] = OrderedDict()
_CACHE_MAX = 4


class _Handle:
    """Owns a C++ solver handle and frees it when evicted."""

    def __init__(self, ptr, Nx, Ny):
        self.ptr, self.Nx, self.Ny = ptr, Nx, Ny

    def __del__(self):
        if getattr(self, "ptr", None):
            _lib.sb_destroy(ctypes.c_void_p(self.ptr))
            self.ptr = None


def _solver(alpha: np.ndarray, Pr: float, Ra: float, inertia: float = 0.0) -> _Handle:
    """Assemble+factorise A(alpha), reusing the LU when the design repeats.

    With inertia the cached object still owns A, but the operator the tangent
    and adjoint invert is the Jacobian at the converged state, which depends on
    T as well. Every `sb_apply` either validates and reuses that exact state or
    rebuilds its Jacobian, and both derivative endpoints below call the forward
    solve first, so the Jacobian always belongs to the state being
    differentiated.
    """
    Ny, Nx = alpha.shape
    key = hashlib.blake2b(
        alpha.tobytes()
        + np.array([Pr, Ra, inertia, Nx, Ny], dtype=np.float64).tobytes(),
        digest_size=16,
    ).hexdigest()
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]

    ptr = _lib.sb_create_ns(Nx, Ny, ctypes.c_double(Pr), ctypes.c_double(Ra),
                            ctypes.c_double(inertia), _p(alpha))
    if not ptr:
        raise RuntimeError(
            "Stokes-Brinkman factorisation failed (singular system). "
            "Check that alpha is finite and the grid is at least 3x3."
        )
    h = _Handle(ptr, Nx, Ny)
    _CACHE[key] = h
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return h


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------


class InputSchema(BaseModel):
    """Design field and temperature; grid size is inferred from the shapes."""

    alpha: Differentiable[Array[(None, None), Float64]] = Field(
        description="Cell-centred Brinkman drag coefficient, shape (Ny, Nx). "
        "Large inside solid material (blocks flow), ~0 in open fluid."
    )
    T: Differentiable[Array[(None, None), Float64]] = Field(
        description="Cell-centred temperature driving buoyancy, shape (Ny, Nx)."
    )
    Ra: Float64 = Field(default=3.0e4, description="Rayleigh number.")
    Pr: Float64 = Field(default=7.0, description="Prandtl number.")
    inertia: Float64 = Field(
        default=0.0,
        description="Weight on the convective acceleration (u.grad)u. 0 is the "
        "infinite-Prandtl Stokes limit, in which the block is linear in "
        "(u,v,p) and one factorisation serves the solve, the tangent and the "
        "adjoint. 1 is steady Navier-Stokes: the solve becomes a damped Newton "
        "iteration and the derivatives invert the Jacobian at the converged "
        "state instead. Values in between exist for continuation.",
    )


class OutputSchema(BaseModel):
    u: Differentiable[Array[(None, None), Float64]] = Field(
        description="x-velocity on vertical faces, shape (Ny, Nx+1). Zero on walls."
    )
    v: Differentiable[Array[(None, None), Float64]] = Field(
        description="y-velocity on horizontal faces, shape (Ny+1, Nx). Zero on walls."
    )
    nonlinear_converged: Float64 = Field(
        description="1 when the most recent fluid solve satisfied its relative "
        "nonlinear residual tolerance; failed solves raise instead of returning output."
    )
    nonlinear_residual: Float64 = Field(
        description="Infinity-norm momentum/continuity residual relative to the "
        "buoyancy load (0 for the linear inertia=0 path)."
    )
    nonlinear_iterations: Float64 = Field(
        description="Number of accepted Newton updates (0 for inertia=0)."
    )


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------


def _forward(
    alpha: np.ndarray,
    T: np.ndarray,
    Pr: float,
    Ra: float,
    inertia: float = 0.0,
):
    Ny, Nx = alpha.shape
    h = _solver(alpha, Pr, Ra, inertia)
    u = np.zeros((Ny, Nx + 1), dtype=np.float64)
    v = np.zeros((Ny + 1, Nx), dtype=np.float64)
    p = np.zeros((Ny, Nx), dtype=np.float64)
    rc = int(_lib.sb_apply(ctypes.c_void_p(h.ptr), _p(T), _p(u), _p(v), _p(p)))
    residual = float(_lib.sb_residual(ctypes.c_void_p(h.ptr), _p(T)))
    converged = int(_lib.sb_converged(ctypes.c_void_p(h.ptr)))
    iterations = int(_lib.sb_newton_iterations(ctypes.c_void_p(h.ptr)))
    status = int(_lib.sb_last_status(ctypes.c_void_p(h.ptr)))
    if rc or not converged:
        reasons = {
            1: "linear solve failed or produced non-finite values",
            2: "nonlinear Jacobian factorisation/solve failed",
            3: "non-finite nonlinear residual",
            4: "Newton line search could not reduce the residual",
            5: "Newton iteration budget exhausted",
            6: "invalid nonlinear solver options",
        }
        reason = reasons.get(status or rc, "unknown solver failure")
        raise RuntimeError(
            "Stokes-Brinkman forward solve did not converge: "
            f"{reason}; status={status or rc}, accepted Newton updates={iterations}, "
            f"relative residual={residual:.6e}."
        )
    diagnostics = {
        "nonlinear_converged": np.float64(converged),
        "nonlinear_residual": np.float64(residual),
        "nonlinear_iterations": np.float64(iterations),
    }
    return u, v, diagnostics


def apply(inputs: InputSchema) -> OutputSchema:
    alpha, T = _c(inputs.alpha), _c(inputs.T)
    u, v, diagnostics = _forward(alpha, T, float(inputs.Pr), float(inputs.Ra),
                                 float(inputs.inertia))
    return OutputSchema(u=u, v=v, **diagnostics)


def abstract_eval(abstract_inputs):
    """Output shapes follow from the grid, with no solve required."""
    Ny, Nx = abstract_inputs.alpha.shape
    return {
        "u": ShapeDType(shape=(Ny, Nx + 1), dtype="float64"),
        "v": ShapeDType(shape=(Ny + 1, Nx), dtype="float64"),
        "nonlinear_converged": ShapeDType(shape=(), dtype="float64"),
        "nonlinear_residual": ShapeDType(shape=(), dtype="float64"),
        "nonlinear_iterations": ShapeDType(shape=(), dtype="float64"),
    }


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    alpha, T = _c(inputs.alpha), _c(inputs.T)
    Pr, Ra = float(inputs.Pr), float(inputs.Ra)
    inertia = float(inputs.inertia)
    Ny, Nx = alpha.shape

    u, v, _ = _forward(alpha, T, Pr, Ra, inertia)
    h = _solver(alpha, Pr, Ra, inertia)

    d_alpha = _c(tangent_vector["alpha"]) if "alpha" in jvp_inputs else None
    d_T = _c(tangent_vector["T"]) if "T" in jvp_inputs else None

    du = np.zeros((Ny, Nx + 1), dtype=np.float64)
    dv = np.zeros((Ny + 1, Nx), dtype=np.float64)
    if _lib.sb_jvp(
        ctypes.c_void_p(h.ptr),
        _p(u),
        _p(v),
        _p(d_alpha) if d_alpha is not None else None,
        _p(d_T) if d_T is not None else None,
        _p(du),
        _p(dv),
    ):
        raise RuntimeError("Stokes-Brinkman JVP solve failed.")

    return {k: {"u": du, "v": dv}[k] for k in jvp_outputs}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    alpha, T = _c(inputs.alpha), _c(inputs.T)
    Pr, Ra = float(inputs.Pr), float(inputs.Ra)
    inertia = float(inputs.inertia)
    Ny, Nx = alpha.shape

    u, v, _ = _forward(alpha, T, Pr, Ra, inertia)
    h = _solver(alpha, Pr, Ra, inertia)

    ubar = _c(cotangent_vector["u"]) if "u" in vjp_outputs else np.zeros((Ny, Nx + 1))
    vbar = _c(cotangent_vector["v"]) if "v" in vjp_outputs else np.zeros((Ny + 1, Nx))
    ubar, vbar = _c(ubar), _c(vbar)

    alphabar = np.zeros((Ny, Nx), dtype=np.float64)
    Tbar = np.zeros((Ny, Nx), dtype=np.float64)
    if _lib.sb_vjp(
        ctypes.c_void_p(h.ptr),
        _p(u),
        _p(v),
        _p(ubar),
        _p(vbar),
        None,
        _p(alphabar),
        _p(Tbar),
    ):
        raise RuntimeError("Stokes-Brinkman VJP solve failed.")

    return {k: {"alpha": alphabar, "T": Tbar}[k] for k in vjp_inputs}
