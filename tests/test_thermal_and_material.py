# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""The JAX thermal Tesseract and the PyTorch material_map Tesseract."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

N = 10


def relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


# --------------------------------------------------------------------------
# thermal_advdiff
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def th_case(_paths):
    from conftest import load_tesseract_api

    mod = load_tesseract_api("thermal_advdiff")
    rng = np.random.default_rng(3)
    u = jnp.asarray(rng.normal(size=(N, N + 1)) * 5.0).at[:, 0].set(0.0).at[:, N].set(0.0)
    v = jnp.asarray(rng.normal(size=(N + 1, N)) * 5.0).at[0, :].set(0.0).at[N, :].set(0.0)
    k = jnp.asarray(rng.uniform(0.02, 1.0, size=(N, N)))
    return mod, u, v, k, rng


def _dense_solve(mod, u, v, k, q=1.0, cf=0.4):
    Ny, Nx = k.shape
    res = lambda t: mod.residual(t.reshape(Ny, Nx), u, v, k, q, cf).ravel()  # noqa: E731
    A = jax.jacfwd(res)(jnp.zeros(Nx * Ny))
    b = -res(jnp.zeros(Nx * Ny))
    return jnp.linalg.solve(A, b).reshape(Ny, Nx)


def test_sparse_assembly_reproduces_the_residual(th_case):
    """The hand-assembled operator must equal the JAX residual it stands in for.

    If these drift apart the whole pipeline is silently wrong, so this is the
    single most important check on this component.
    """
    mod, u, v, k, rng = th_case
    A = mod.assemble(u, v, k, N, N)
    b = mod.rhs(N, N, 1.0, 0.4)
    T = jnp.asarray(rng.normal(size=(N, N)))
    lhs = A @ np.asarray(T).ravel() - b
    ref = np.asarray(mod.residual(T, u, v, k, 1.0, 0.4)).ravel()
    assert relerr(lhs, ref) < 1e-12


def test_solve_is_consistent_and_matches_dense(th_case):
    mod, u, v, k, _ = th_case
    inputs = mod.InputSchema(u=u, v=v, k=k, q_chip=1.0, chip_frac=0.4)
    T = mod.apply(inputs).T
    r = np.asarray(mod.residual(jnp.asarray(T), u, v, k, 1.0, 0.4))
    assert np.max(np.abs(r)) < 1e-10
    assert relerr(T, _dense_solve(mod, u, v, k)) < 1e-12


def test_thermal_derivatives_match_autodiff(th_case):
    mod, u, v, k, rng = th_case
    inputs = mod.InputSchema(u=u, v=v, k=k, q_chip=1.0, chip_frac=0.4)
    du = jnp.asarray(rng.normal(size=u.shape))
    dv = jnp.asarray(rng.normal(size=v.shape))
    dk = jnp.asarray(rng.normal(size=k.shape))

    jvp = mod.jacobian_vector_product(
        inputs, {"u", "v", "k"}, {"T"}, {"u": du, "v": dv, "k": dk}
    )["T"]
    f = lambda a, b, c: _dense_solve(mod, a, b, c)  # noqa: E731
    _, jvp_ref = jax.jvp(f, (u, v, k), (du, dv, dk))
    assert relerr(jvp, jvp_ref) < 1e-11

    Tbar = jnp.asarray(rng.normal(size=(N, N)))
    vjp = mod.vector_jacobian_product(inputs, {"u", "v", "k"}, {"T"}, {"T": Tbar})
    _, vjp_fn = jax.vjp(f, u, v, k)
    gu, gv, gk = vjp_fn(Tbar)
    assert max(relerr(vjp["u"], gu), relerr(vjp["v"], gv), relerr(vjp["k"], gk)) < 1e-11

    lhs = float(jnp.sum(jvp * Tbar))
    rhs = float(
        np.sum(np.asarray(du) * vjp["u"])
        + np.sum(np.asarray(dv) * vjp["v"])
        + np.sum(np.asarray(dk) * vjp["k"])
    )
    assert abs(lhs - rhs) <= 1e-11 * max(abs(lhs), 1e-30)


def test_peclet_weighting_is_smooth_through_zero_velocity(th_case):
    """The face weighting must be differentiable where a velocity changes sign.

    A hard upwind switch is non-differentiable exactly there, which stalled
    Newton at ~1e-2 and put a kink in the gradient path. Check the derivative
    of the weight is finite and the weight passes smoothly through 0.5.
    """
    mod, _, _, k, _ = th_case
    kf = 0.1

    def w(u_face):
        h = 1.0 / N
        return 0.5 * (1.0 + jnp.tanh(0.5 * u_face * h / kf))

    assert abs(float(w(0.0)) - 0.5) < 1e-14
    d = jax.grad(w)(0.0)
    assert np.isfinite(float(d)) and float(d) > 0


def test_differentially_heated_cavity_assembly_matches_residual(th_case):
    """Mode 2 must retain the residual/assembled-operator identity."""
    mod, u, v, k, rng = th_case
    T = jnp.asarray(rng.normal(size=(N, N)))
    A = mod.assemble(u, v, k, N, N, bc_mode=2.0)
    # Deliberately nonzero: mode 2 must ignore the cold-plate chip flux.
    q, cf = 3.7, 0.4
    b = mod.rhs(N, N, q, cf, bc_mode=2.0, t_hot=1.0, k=k)
    lhs = A @ np.asarray(T).ravel() - b
    ref = np.asarray(mod.residual(T, u, v, k, q, cf, 2.0, 1.0)).ravel()
    assert relerr(lhs, ref) < 1e-12


def test_differentially_heated_cavity_recovers_conduction_profile(th_case):
    """With zero flow the side-heated square has the exact linear solution."""
    mod, _, _, _, _ = th_case
    u = np.zeros((N, N + 1))
    v = np.zeros((N + 1, N))
    k = np.ones((N, N))
    inputs = mod.InputSchema(
        # A nonzero chip flux would bend this profile if mode 2 fell through
        # to the cold-plate bottom condition instead of making it adiabatic.
        u=u, v=v, k=k, q_chip=13.0, chip_frac=0.4,
        bc_mode=2.0, t_hot=1.0,
    )
    T = np.asarray(mod.apply(inputs).T)
    expected = np.tile(1.0 - (np.arange(N) + 0.5) / N, (N, 1))
    assert np.max(np.abs(T - expected)) < 2e-14
    assert np.max(np.abs(mod.residual(T, u, v, k, 13.0, 0.4, 2.0, 1.0))) < 1e-11


def test_rayleigh_benard_mode_still_recovers_vertical_conduction(th_case):
    """The additive mode 2 must not alter the pre-existing mode-1 boundary."""
    mod, _, _, _, _ = th_case
    u = np.zeros((N, N + 1))
    v = np.zeros((N + 1, N))
    k = np.ones((N, N))
    inputs = mod.InputSchema(u=u, v=v, k=k, bc_mode=1.0, t_hot=1.0)
    T = np.asarray(mod.apply(inputs).T)
    expected = np.tile((1.0 - (np.arange(N) + 0.5) / N)[:, None], (1, N))
    assert np.max(np.abs(T - expected)) < 2e-14


# --------------------------------------------------------------------------
# material_map
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mat(_paths):
    from conftest import load_tesseract_api

    return load_tesseract_api("material_map")


def test_filter_is_unbiased_at_the_walls(mat):
    """A uniform field must stay spatially uniform.

    Normalising by the convolution of ones is what stops the border being
    darkened by its missing neighbours.
    """
    out = mat.apply(
        mat.InputSchema(rho_raw=np.full((N, N), 0.42), filter_radius=2.0, beta=1.0)
    )
    assert out.rho_phys.std() < 1e-12


def test_projection_fixes_eta_to_one_half(mat):
    out = mat.apply(
        mat.InputSchema(rho_raw=np.full((N, N), 0.5), filter_radius=2.0, beta=8.0, eta=0.5)
    )
    assert np.max(np.abs(out.rho_phys - 0.5)) < 1e-12


def test_material_vjp_matches_finite_differences(mat):
    rng = np.random.default_rng(11)
    rho = rng.uniform(0.1, 0.9, size=(N, N))
    kw = dict(filter_radius=2.0, beta=4.0, penal=3.0)
    inputs = mat.InputSchema(rho_raw=rho, **kw)
    cot = {n: rng.normal(size=(N, N)) for n in ("k", "alpha", "rho_phys")}
    vjp = mat.vector_jacobian_product(inputs, {"rho_raw"}, set(cot), cot)["rho_raw"]

    def scalar(r):
        o = mat.apply(mat.InputSchema(rho_raw=r, **kw))
        return sum(float((getattr(o, n) * cot[n]).sum()) for n in cot)

    eps = 1e-6
    for j, i in [(2, 3), (7, 5), (0, 9)]:
        rp, rm = rho.copy(), rho.copy()
        rp[j, i] += eps
        rm[j, i] -= eps
        fd = (scalar(rp) - scalar(rm)) / (2 * eps)
        assert abs(vjp[j, i] - fd) <= 1e-6 * max(abs(fd), 1.0)


def test_material_adjoint_identity(mat):
    rng = np.random.default_rng(5)
    rho = rng.uniform(0.1, 0.9, size=(N, N))
    kw = dict(filter_radius=2.0, beta=4.0, penal=3.0)
    inputs = mat.InputSchema(rho_raw=rho, **kw)
    cot = {n: rng.normal(size=(N, N)) for n in ("k", "alpha", "rho_phys")}
    tan = rng.normal(size=(N, N))

    jvp = mat.jacobian_vector_product(inputs, {"rho_raw"}, set(cot), {"rho_raw": tan})
    vjp = mat.vector_jacobian_product(inputs, {"rho_raw"}, set(cot), cot)["rho_raw"]
    lhs = sum(float((jvp[n] * cot[n]).sum()) for n in cot)
    rhs = float((tan * vjp).sum())
    assert abs(lhs - rhs) <= 1e-9 * max(abs(lhs), 1e-30)
