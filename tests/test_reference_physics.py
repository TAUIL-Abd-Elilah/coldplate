# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Physics invariants of the reference implementation.

These are the checks that caught real bugs during development: a sign error in
the chip heat-flux boundary condition (which made every temperature negative),
and a test that wrongly expected zero flow under a purely vertical temperature
gradient. They are cheap and they pin the discretisation down.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def ref():
    import reference_jax as r

    return r


def test_hydrostatic_balance_produces_no_flow(ref):
    """A body force that varies only with height is conservative.

    Pressure absorbs it exactly, so the correct answer is zero velocity. This
    looks like a broken solver if you are not expecting it.
    """
    cfg = ref.Config(Nx=16, Ny=16)
    _, alpha = ref.material_maps(jnp.zeros((cfg.Ny, cfg.Nx)), cfg)
    y = (jnp.arange(cfg.Ny) + 0.5) / cfg.Ny
    T = jnp.tile((1.0 - y)[:, None], (1, cfg.Nx))
    u, v, _ = ref.solve_fluid(T, alpha, cfg)
    assert float(jnp.abs(v).max()) < 1e-10
    assert float(jnp.abs(u).max()) < 1e-10


def test_horizontal_gradient_drives_antisymmetric_circulation(ref):
    """Hot on the left rises, cold on the right sinks, symmetrically."""
    cfg = ref.Config(Nx=16, Ny=16)
    _, alpha = ref.material_maps(jnp.zeros((cfg.Ny, cfg.Nx)), cfg)
    x = (jnp.arange(cfg.Nx) + 0.5) / cfg.Nx
    T = jnp.tile((1.0 - x)[None, :], (cfg.Ny, 1))
    u, v, _ = ref.solve_fluid(T, alpha, cfg)

    assert float(jnp.abs(v).max()) > 1e-3, "buoyancy should drive a circulation"
    left = float(v[:, : cfg.Nx // 2].mean())
    right = float(v[:, cfg.Nx // 2 :].mean())
    assert left > 0 and right < 0
    assert abs(left + right) < 1e-6 * max(abs(left), 1.0)


def test_incompressibility(ref):
    cfg = ref.Config(Nx=16, Ny=16)
    _, alpha = ref.material_maps(jnp.zeros((cfg.Ny, cfg.Nx)), cfg)
    x = (jnp.arange(cfg.Nx) + 0.5) / cfg.Nx
    T = jnp.tile((1.0 - x)[None, :], (cfg.Ny, 1))
    u, v, _ = ref.solve_fluid(T, alpha, cfg)
    div = (u[:, 1:] - u[:, :-1]) / cfg.h + (v[1:, :] - v[:-1, :]) / cfg.h
    assert float(jnp.abs(div).max()) < 1e-9


def test_brinkman_penalisation_blocks_flow(ref):
    """Filling the domain with solid must suppress the flow by orders of magnitude."""
    cfg = ref.Config(Nx=16, Ny=16)
    x = (jnp.arange(cfg.Nx) + 0.5) / cfg.Nx
    T = jnp.tile((1.0 - x)[None, :], (cfg.Ny, 1))
    _, a_fluid = ref.material_maps(jnp.zeros((cfg.Ny, cfg.Nx)), cfg)
    _, a_solid = ref.material_maps(jnp.ones((cfg.Ny, cfg.Nx)), cfg)
    u_f, _, _ = ref.solve_fluid(T, a_fluid, cfg)
    u_s, _, _ = ref.solve_fluid(T, a_solid, cfg)
    assert float(jnp.abs(u_s).max()) < 1e-2 * float(jnp.abs(u_f).max())


def test_conduction_energy_balance(ref):
    """With no flow, heat in at the chip must equal heat out at the cold sink.

    This is the check that caught the chip-flux sign error: it fails loudly and
    unambiguously if the boundary condition pushes heat the wrong way.
    """
    cfg = ref.Config(Nx=16, Ny=16)
    k, _ = ref.material_maps(jnp.zeros((cfg.Ny, cfg.Nx)), cfg)
    T = ref.solve_thermal(
        jnp.zeros((cfg.Ny, cfg.Nx + 1)), jnp.zeros((cfg.Ny + 1, cfg.Nx)), k, cfg
    )
    assert float(T.min()) > 0, "heat enters the domain, so T must be positive"

    q_in = float(cfg.q_chip * cfg.chip_mask().sum() * cfg.h)
    q_out = float((k[cfg.Ny - 1, :] * T[cfg.Ny - 1, :] / (0.5 * cfg.h)).sum() * cfg.h)
    assert abs(q_in - q_out) / q_in < 1e-12


def test_stronger_buoyancy_cools_better(ref):
    """Raising Ra must lower the chip temperature: more convection, more cooling."""
    rho = jnp.zeros((16, 16))
    Js = []
    for Ra in (1e2, 1e3):
        cfg = ref.Config(Nx=16, Ny=16, Ra=Ra)
        T, info = ref.solve_coupled(rho, cfg, max_iter=400, relax=0.5)
        # Damped Picard converges linearly, so this tolerance reflects how far
        # 400 sweeps get, not the accuracy of the discretisation.
        assert info["residual"] < 1e-8
        Js.append(float(ref.objective(T, cfg)))
    assert Js[1] < Js[0]


def test_implicit_gradient_matches_unrolled_and_finite_differences(ref):
    """Three independent routes to dJ/drho must agree.

    Implicit differentiation of the fixed point, reverse-mode through the
    unrolled Picard loop, and central finite differences on the whole coupled
    solve. Agreement across all three is the core correctness claim.
    """
    import jax

    cfg = ref.Config(Nx=10, Ny=10, Ra=1.0e3)
    rng = np.random.default_rng(0)
    rho = jnp.asarray(rng.uniform(0.25, 0.75, size=(cfg.Ny, cfg.Nx)))

    T_star, info = ref.solve_coupled(rho, cfg, max_iter=400, relax=0.5)
    assert info["residual"] < 1e-11

    phi = lambda T, r: ref.coupled_step(T, cfg, r)  # noqa: E731
    _, vjp_fn = jax.vjp(phi, T_star, rho)
    g = jax.grad(lambda T: ref.objective(T, cfg))(T_star)
    lam, w = g, g
    for _ in range(2000):
        w = vjp_fn(w)[0]
        lam = lam + w
        if float(jnp.max(jnp.abs(w))) < 1e-14:
            break
    g_implicit = vjp_fn(lam)[1]

    g_unrolled = jax.grad(lambda r: ref.loss_unrolled(r, cfg, 300, 0.5))(rho)
    assert float(jnp.max(jnp.abs(g_implicit - g_unrolled))) < 1e-9

    eps = 1e-5
    for j, i in [(2, 3), (7, 5)]:
        Jp = ref.objective(ref.solve_coupled(rho.at[j, i].add(eps), cfg, 400, 0.5)[0], cfg)
        Jm = ref.objective(ref.solve_coupled(rho.at[j, i].add(-eps), cfg, 400, 0.5)[0], cfg)
        fd = float((Jp - Jm) / (2 * eps))
        assert abs(float(g_implicit[j, i]) - fd) <= 1e-5 * max(abs(fd), 1e-3)
