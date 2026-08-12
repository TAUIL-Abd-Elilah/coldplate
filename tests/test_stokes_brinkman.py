# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""The C++/Eigen solver and its hand-derived adjoint, against JAX autodiff.

The C++ block computes derivatives by hand -- transpose solve plus an analytic
scatter -- while the reference uses autodiff. They share no code, so agreement
to machine precision is genuine evidence rather than a shared bug.
"""

from __future__ import annotations

import ctypes

import jax
import jax.numpy as jnp
import numpy as np
import pytest

N = 12
RA, PR = 3.0e4, 7.0


def relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


@pytest.fixture(scope="module")
def lib(stokes_lib):
    dp = ctypes.POINTER(ctypes.c_double)
    L = ctypes.CDLL(str(stokes_lib))
    L.sb_create.restype = ctypes.c_void_p
    L.sb_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, dp]
    L.sb_destroy.argtypes = [ctypes.c_void_p]
    L.sb_apply.restype = ctypes.c_int
    L.sb_apply.argtypes = [ctypes.c_void_p, dp, dp, dp, dp]
    L.sb_jvp.restype = ctypes.c_int
    L.sb_jvp.argtypes = [ctypes.c_void_p, dp, dp, dp, dp, dp, dp]
    L.sb_vjp.restype = ctypes.c_int
    L.sb_vjp.argtypes = [ctypes.c_void_p, dp, dp, dp, dp, dp, dp, dp]
    return L


@pytest.fixture(scope="module")
def case(lib):
    """A design with real contrast: near-solid cells alongside open fluid."""
    rng = np.random.default_rng(7)
    P = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))  # noqa: E731
    C = lambda a: np.ascontiguousarray(a, dtype=np.float64)  # noqa: E731

    alpha = C(10.0 ** rng.uniform(-2, 4, size=(N, N)))
    T = C(rng.normal(size=(N, N)))
    h = lib.sb_create(N, N, ctypes.c_double(PR), ctypes.c_double(RA), P(alpha))
    assert h, "sb_create returned null"
    h = ctypes.c_void_p(h)

    u = np.zeros((N, N + 1))
    v = np.zeros((N + 1, N))
    p = np.zeros((N, N))
    assert lib.sb_apply(h, P(T), P(u), P(v), P(p)) == 0
    yield dict(alpha=alpha, T=T, u=u, v=v, h=h, P=P, C=C, rng=rng)
    lib.sb_destroy(h)


@pytest.fixture(scope="module")
def cfg():
    import reference_jax as r

    return r, r.Config(Nx=N, Ny=N, Ra=RA, Pr=PR)


def test_forward_matches_reference(case, cfg):
    r, c = cfg
    u_ref, v_ref, _ = r.solve_fluid(jnp.asarray(case["T"]), jnp.asarray(case["alpha"]), c)
    assert relerr(case["u"], u_ref) < 1e-9
    assert relerr(case["v"], v_ref) < 1e-9


def test_jvp_matches_autodiff(lib, case, cfg):
    r, c = cfg
    P, C, rng = case["P"], case["C"], case["rng"]
    d_alpha = C(rng.normal(size=(N, N)) * case["alpha"])
    d_T = C(rng.normal(size=(N, N)))
    du = np.zeros((N, N + 1))
    dv = np.zeros((N + 1, N))
    assert lib.sb_jvp(case["h"], P(case["u"]), P(case["v"]), P(d_alpha), P(d_T),
                      P(du), P(dv)) == 0

    f = lambda a, t: r.solve_fluid(t, a, c)[:2]  # noqa: E731
    _, (du_ref, dv_ref) = jax.jvp(
        f, (jnp.asarray(case["alpha"]), jnp.asarray(case["T"])),
        (jnp.asarray(d_alpha), jnp.asarray(d_T)),
    )
    assert relerr(du, du_ref) < 1e-9
    assert relerr(dv, dv_ref) < 1e-9


def test_vjp_matches_autodiff(lib, case, cfg):
    r, c = cfg
    P, C, rng = case["P"], case["C"], case["rng"]
    ubar = C(rng.normal(size=(N, N + 1)))
    vbar = C(rng.normal(size=(N + 1, N)))
    abar = np.zeros((N, N))
    Tbar = np.zeros((N, N))
    assert lib.sb_vjp(case["h"], P(case["u"]), P(case["v"]), P(ubar), P(vbar),
                      None, P(abar), P(Tbar)) == 0

    f = lambda a, t: r.solve_fluid(t, a, c)[:2]  # noqa: E731
    _, vjp_fn = jax.vjp(f, jnp.asarray(case["alpha"]), jnp.asarray(case["T"]))
    abar_ref, Tbar_ref = vjp_fn((jnp.asarray(ubar), jnp.asarray(vbar)))
    assert relerr(abar, abar_ref) < 1e-9
    assert relerr(Tbar, Tbar_ref) < 1e-9


def test_adjoint_identity(lib, case):
    """<J dx, ybar> == <dx, J^T ybar>, entirely inside the C++ component."""
    P, C, rng = case["P"], case["C"], case["rng"]
    d_alpha = C(rng.normal(size=(N, N)) * case["alpha"])
    d_T = C(rng.normal(size=(N, N)))
    ubar = C(rng.normal(size=(N, N + 1)))
    vbar = C(rng.normal(size=(N + 1, N)))

    du = np.zeros((N, N + 1))
    dv = np.zeros((N + 1, N))
    abar = np.zeros((N, N))
    Tbar = np.zeros((N, N))
    lib.sb_jvp(case["h"], P(case["u"]), P(case["v"]), P(d_alpha), P(d_T), P(du), P(dv))
    lib.sb_vjp(case["h"], P(case["u"]), P(case["v"]), P(ubar), P(vbar), None,
               P(abar), P(Tbar))

    lhs = float(np.sum(du * ubar) + np.sum(dv * vbar))
    rhs = float(np.sum(d_alpha * abar) + np.sum(d_T * Tbar))
    assert abs(lhs - rhs) <= 1e-9 * max(abs(lhs), 1e-30)


def test_solid_domain_blocks_flow(lib, case):
    """Brinkman drag must suppress the flow when the domain is all solid."""
    P = case["P"]
    alpha_solid = np.ascontiguousarray(np.full((N, N), 1e5), dtype=np.float64)
    h = lib.sb_create(N, N, ctypes.c_double(PR), ctypes.c_double(RA), P(alpha_solid))
    assert h
    h = ctypes.c_void_p(h)
    u = np.zeros((N, N + 1))
    v = np.zeros((N + 1, N))
    p = np.zeros((N, N))
    assert lib.sb_apply(h, P(case["T"]), P(u), P(v), P(p)) == 0
    lib.sb_destroy(h)
    assert np.abs(u).max() < 1e-2 * np.abs(case["u"]).max()
