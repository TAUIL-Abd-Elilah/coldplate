# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""The inertial (Navier-Stokes) fluid solver, against the JAX reference.

Adding the convective acceleration turns the fluid block from a linear system
into a nonlinear one, which means the hand-derived adjoint now has to
differentiate through a Newton solve rather than a single factorisation. The
saving grace is that (u.grad)u is *bilinear*: its Jacobian entries are the two
factors read off in turn, and it involves neither alpha nor T, so the parameter
scatters in the JVP and VJP are untouched.

That is exactly the kind of hand derivation that is easy to get subtly wrong,
so every claim here is checked against `prototype/reference_jax.py`, where the
same residual is differentiated by autodiff instead. Machine-precision
agreement between an analytic Jacobian and an autodiff one is the only evidence
worth having.
"""

from __future__ import annotations

import ctypes
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import reference_jax as ref  # noqa: E402

# Pr = 0.71 (air) and a high Rayleigh number, because inertia has to actually
# matter for these tests to mean anything. In this scaling the convective term
# is weighted by roughly 1/Pr, so at the water-like Pr = 7 used elsewhere in
# the repository inertia moves the solution by under 1e-4 -- which is precisely
# why the Stokes limit was a defensible model there, and precisely why it is
# useless as a test bed for the nonlinear code path.
N = 12
PR, RA = 0.71, 1.0e6


def make_cfg(inertia):
    return ref.Config(Nx=N, Ny=N, Ra=RA, Pr=PR, inertia=inertia)


@pytest.fixture(scope="module")
def lib(stokes_lib):
    dp = ctypes.POINTER(ctypes.c_double)
    L = ctypes.CDLL(str(stokes_lib))
    L.sb_create_ns.restype = ctypes.c_void_p
    L.sb_create_ns.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double,
                               ctypes.c_double, ctypes.c_double, dp]
    L.sb_create.restype = ctypes.c_void_p
    L.sb_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double,
                            ctypes.c_double, dp]
    L.sb_destroy.argtypes = [ctypes.c_void_p]
    L.sb_apply.argtypes = [ctypes.c_void_p, dp, dp, dp, dp]
    L.sb_apply.restype = ctypes.c_int
    L.sb_residual.argtypes = [ctypes.c_void_p, dp]
    L.sb_residual.restype = ctypes.c_double
    L.sb_jvp.argtypes = [ctypes.c_void_p, dp, dp, dp, dp, dp, dp]
    L.sb_jvp.restype = ctypes.c_int
    L.sb_vjp.argtypes = [ctypes.c_void_p, dp, dp, dp, dp, dp, dp, dp]
    L.sb_vjp.restype = ctypes.c_int
    return L


def P(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


@pytest.fixture(scope="module")
def case():
    """A design and temperature field that produce a genuinely inertial flow."""
    rng = np.random.default_rng(0)
    # Low density: heavy Brinkman drag would damp exactly the inertia we are
    # trying to exercise.
    rho = rng.uniform(0.0, 0.15, size=(N, N))
    _, alpha = ref.material_maps(jnp.asarray(rho), make_cfg(0.0))
    T = np.asarray(rng.uniform(0.0, 1.0, size=(N, N)))
    return {"alpha": np.ascontiguousarray(np.asarray(alpha, dtype=np.float64)),
            "T": np.ascontiguousarray(T)}


def solve_cpp(lib, case, inertia):
    u = np.zeros((N, N + 1))
    v = np.zeros((N + 1, N))
    p = np.zeros((N, N))
    h = lib.sb_create_ns(N, N, PR, RA, ctypes.c_double(inertia), P(case["alpha"]))
    assert h, "solver construction failed"
    rc = lib.sb_apply(ctypes.c_void_p(h), P(case["T"]), P(u), P(v), P(p))
    assert rc == 0, f"sb_apply returned {rc}"
    return h, u, v, p


def relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


# -- the zero-inertia path must be untouched -------------------------------


def test_zero_inertia_is_bitwise_the_old_solver(lib, case):
    """sb_create and sb_create_ns(inertia=0) must agree exactly, not closely."""
    u0 = np.zeros((N, N + 1)); v0 = np.zeros((N + 1, N)); p0 = np.zeros((N, N))
    h0 = lib.sb_create(N, N, PR, RA, P(case["alpha"]))
    assert lib.sb_apply(ctypes.c_void_p(h0), P(case["T"]), P(u0), P(v0), P(p0)) == 0

    h1, u1, v1, p1 = solve_cpp(lib, case, 0.0)
    assert np.array_equal(u0, u1)
    assert np.array_equal(v0, v1)
    assert np.array_equal(p0, p1)
    lib.sb_destroy(ctypes.c_void_p(h0))
    lib.sb_destroy(ctypes.c_void_p(h1))


def test_zero_inertia_matches_the_linear_reference(lib, case):
    h, u, v, _ = solve_cpp(lib, case, 0.0)
    cfg = make_cfg(0.0)
    u_ref, v_ref, _ = ref.solve_fluid(jnp.asarray(case["T"]),
                                      jnp.asarray(case["alpha"]), cfg)
    assert relerr(u, u_ref) < 1e-9
    assert relerr(v, v_ref) < 1e-9
    lib.sb_destroy(ctypes.c_void_p(h))


# -- the inertial forward solve --------------------------------------------


def test_inertia_actually_changes_the_flow(lib, case):
    """Guard against the term being silently multiplied by zero somewhere."""
    h0, u0, v0, _ = solve_cpp(lib, case, 0.0)
    h1, u1, v1, _ = solve_cpp(lib, case, 1.0)
    change = np.max(np.abs(u1 - u0)) / max(np.max(np.abs(u0)), 1e-300)
    assert change > 1e-3, f"inertia moved the solution by only {change:.2e}"
    lib.sb_destroy(ctypes.c_void_p(h0))
    lib.sb_destroy(ctypes.c_void_p(h1))


def test_newton_converges_and_residual_is_tiny(lib, case):
    # Relative to the load: the buoyancy term carries Ra*Pr ~ 7e5 here, so an
    # absolute threshold would be a statement about the Rayleigh number.
    h, _, _, _ = solve_cpp(lib, case, 1.0)
    r = lib.sb_residual(ctypes.c_void_p(h), P(case["T"]))
    assert r < 1e-12, f"relative nonlinear residual {r:.3e}"
    lib.sb_destroy(ctypes.c_void_p(h))


def test_inertial_forward_matches_the_reference(lib, case):
    h, u, v, _ = solve_cpp(lib, case, 1.0)
    cfg = make_cfg(1.0)
    u_ref, v_ref, _ = ref.solve_fluid(jnp.asarray(case["T"]),
                                      jnp.asarray(case["alpha"]), cfg)
    assert relerr(u, u_ref) < 1e-8
    assert relerr(v, v_ref) < 1e-8
    lib.sb_destroy(ctypes.c_void_p(h))


# -- derivatives through the nonlinear solve -------------------------------


def _ref_flow(alpha, T, inertia):
    cfg = make_cfg(inertia)
    u, v, _ = ref.solve_fluid(T, alpha, cfg)
    return u, v


@pytest.mark.parametrize("inertia", [0.0, 1.0])
def test_jvp_matches_autodiff(lib, case, inertia):
    rng = np.random.default_rng(3)
    d_alpha = np.ascontiguousarray(rng.normal(size=(N, N)) * 1e3)
    d_T = np.ascontiguousarray(rng.normal(size=(N, N)))

    h, u, v, _ = solve_cpp(lib, case, inertia)
    du = np.zeros((N, N + 1)); dv = np.zeros((N + 1, N))
    assert lib.sb_jvp(ctypes.c_void_p(h), P(u), P(v), P(d_alpha), P(d_T),
                      P(du), P(dv)) == 0

    f = lambda a, t: _ref_flow(a, t, inertia)  # noqa: E731
    _, (du_ref, dv_ref) = jax.jvp(
        f, (jnp.asarray(case["alpha"]), jnp.asarray(case["T"])),
        (jnp.asarray(d_alpha), jnp.asarray(d_T)),
    )
    assert relerr(du, du_ref) < 1e-7
    assert relerr(dv, dv_ref) < 1e-7
    lib.sb_destroy(ctypes.c_void_p(h))


@pytest.mark.parametrize("inertia", [0.0, 1.0])
def test_vjp_matches_autodiff(lib, case, inertia):
    rng = np.random.default_rng(4)
    ubar = np.ascontiguousarray(rng.normal(size=(N, N + 1)))
    vbar = np.ascontiguousarray(rng.normal(size=(N + 1, N)))

    h, u, v, _ = solve_cpp(lib, case, inertia)
    abar = np.zeros((N, N)); Tbar = np.zeros((N, N))
    assert lib.sb_vjp(ctypes.c_void_p(h), P(u), P(v), P(ubar), P(vbar),
                      None, P(abar), P(Tbar)) == 0

    f = lambda a, t: _ref_flow(a, t, inertia)  # noqa: E731
    _, vjp_fn = jax.vjp(f, jnp.asarray(case["alpha"]), jnp.asarray(case["T"]))
    abar_ref, Tbar_ref = vjp_fn((jnp.asarray(ubar), jnp.asarray(vbar)))
    assert relerr(abar, abar_ref) < 1e-7
    assert relerr(Tbar, Tbar_ref) < 1e-7
    lib.sb_destroy(ctypes.c_void_p(h))


@pytest.mark.parametrize("inertia", [0.0, 1.0])
def test_adjoint_identity(lib, case, inertia):
    """<J dx, y> must equal <dx, J^T y> for the shipped JVP and VJP."""
    rng = np.random.default_rng(5)
    d_alpha = np.ascontiguousarray(rng.normal(size=(N, N)) * 1e3)
    d_T = np.ascontiguousarray(rng.normal(size=(N, N)))
    ubar = np.ascontiguousarray(rng.normal(size=(N, N + 1)))
    vbar = np.ascontiguousarray(rng.normal(size=(N + 1, N)))

    h, u, v, _ = solve_cpp(lib, case, inertia)
    du = np.zeros((N, N + 1)); dv = np.zeros((N + 1, N))
    assert lib.sb_jvp(ctypes.c_void_p(h), P(u), P(v), P(d_alpha), P(d_T),
                      P(du), P(dv)) == 0
    abar = np.zeros((N, N)); Tbar = np.zeros((N, N))
    assert lib.sb_vjp(ctypes.c_void_p(h), P(u), P(v), P(ubar), P(vbar),
                      None, P(abar), P(Tbar)) == 0

    lhs = float(np.sum(du * ubar) + np.sum(dv * vbar))
    rhs = float(np.sum(d_alpha * abar) + np.sum(d_T * Tbar))
    assert abs(lhs - rhs) <= 1e-8 * max(abs(lhs), abs(rhs), 1e-30)
    lib.sb_destroy(ctypes.c_void_p(h))


def test_inertial_jacobian_differs_from_the_stokes_one(lib, case):
    """The derivative must actually go through the nonlinear term.

    If the JVP wrongly kept inverting A instead of J = A + dN/dw, every test
    above except the forward ones could still pass by accident on a weak flow.
    This pins that the two tangents genuinely differ.
    """
    rng = np.random.default_rng(6)
    d_alpha = np.ascontiguousarray(rng.normal(size=(N, N)) * 1e3)
    d_T = np.ascontiguousarray(rng.normal(size=(N, N)))

    tangents = []
    for inertia in (0.0, 1.0):
        h, u, v, _ = solve_cpp(lib, case, inertia)
        du = np.zeros((N, N + 1)); dv = np.zeros((N + 1, N))
        assert lib.sb_jvp(ctypes.c_void_p(h), P(u), P(v), P(d_alpha), P(d_T),
                          P(du), P(dv)) == 0
        tangents.append(du.copy())
        lib.sb_destroy(ctypes.c_void_p(h))

    diff = np.max(np.abs(tangents[1] - tangents[0])) / max(
        np.max(np.abs(tangents[0])), 1e-300)
    assert diff > 1e-3, f"tangents differ by only {diff:.2e}"
