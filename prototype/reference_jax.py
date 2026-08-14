# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Dense-JAX reference implementation of the coupled Boussinesq cold-plate model.

This file is a separately written validation reference, not the submission. It exists to (a) pin down the
discretisation before any of it is split across Tesseracts, and (b) give the
sparse C++ solver an independent target to be checked against.

Physics (steady, non-dimensional, Boussinesq, Stokes flow with Brinkman
penalisation of solid regions):

    fluid    -Pr d^2u + grad p + Pr*alpha(rho)*u = Ra*Pr*T * e_y
             div u = 0
    thermal  div(u T) - div(k(rho) grad T) = 0

Coupling is genuinely two-way: buoyancy makes T drive u, advection makes u
drive T. So the steady state is a fixed point, not a feed-forward chain.

Both residuals are *linear* in their own state unknown, which makes their
state-adjoint solves cheap: the derivative of a linear solve is a transpose
solve reusing the same factorisation. The material map and parameter-to-solution
maps remain nonlinear; the feedback loop supplies the difficult state
nonlinearity.

Trick used throughout: each block is written as a residual R(x, params) that is
linear in x, so the system matrix is exactly jacfwd(R, argnums=0) evaluated at
x = 0, and the rhs is -R(0, params). This costs one dense Jacobian per solve
but removes every hand-assembly indexing bug, which is the point at this stage.

Grid: staggered MAC.
    p, T, rho : (Ny, Nx)      cell centres
    u         : (Ny, Nx+1)    vertical faces
    v         : (Ny+1, Nx)    horizontal faces
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------
# problem definition
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Non-dimensional cold-plate problem.

    The chip is a heated strip on part of the bottom wall; the top wall is the
    cold sink. Everything else is adiabatic and no-slip. The design field rho
    distributes a limited budget of solid (conductive, flow-blocking) material.
    """

    Nx: int = 24
    Ny: int = 24
    Ra: float = 1.0e4  # Rayleigh: buoyancy strength
    Pr: float = 7.0  # Prandtl: water
    k_solid: float = 1.0  # conductivity of solid
    k_fluid: float = 0.02  # conductivity of fluid (50x contrast)
    alpha_max: float = 1.0e5  # Brinkman penalty inside solid
    q_chip: float = 1.0  # heat flux into the chip strip
    chip_frac: float = 0.4  # chip covers middle 40% of the bottom wall
    # Weight on the convective acceleration (u.grad)u in the momentum equation.
    # 0 is the infinite-Prandtl (Stokes) limit and keeps the fluid block linear
    # in w, which is what the original solver assumed everywhere. 1 is the full
    # steady Navier-Stokes-Boussinesq problem. Intermediate values exist so the
    # Newton solve can be continued in from the Stokes solution at high Ra.
    inertia: float = 0.0

    @property
    def h(self) -> float:
        return 1.0 / self.Nx

    def chip_mask(self) -> jnp.ndarray:
        """Indicator over bottom-wall cells that carry the incoming heat flux."""
        i = jnp.arange(self.Nx)
        lo = 0.5 * (1.0 - self.chip_frac) * self.Nx
        hi = 0.5 * (1.0 + self.chip_frac) * self.Nx
        return ((i >= lo) & (i < hi)).astype(jnp.float64)


def material_maps(rho: jnp.ndarray, cfg: Config, penal: float = 3.0):
    """SIMP-style interpolation from density to physical properties.

    rho = 1 is solid (conducts well, blocks flow), rho = 0 is fluid.
    The Brinkman coefficient uses the standard RAMP-like convex form so that
    intermediate densities are strongly penalised for flow.
    """
    k = cfg.k_fluid + (cfg.k_solid - cfg.k_fluid) * rho**penal
    alpha = cfg.alpha_max * rho / (1.0 + 8.0 * (1.0 - rho))
    return k, alpha


# --------------------------------------------------------------------------
# fluid block: Stokes + Brinkman + Boussinesq buoyancy
# --------------------------------------------------------------------------


def _unpack_fluid(w: jnp.ndarray, cfg: Config):
    """Flat unknown vector -> (u, v, p) with no-slip walls already imposed."""
    Nx, Ny = cfg.Nx, cfg.Ny
    nu = Ny * (Nx - 1)
    nv = (Ny - 1) * Nx
    u_int = w[:nu].reshape(Ny, Nx - 1)
    v_int = w[nu : nu + nv].reshape(Ny - 1, Nx)
    p = w[nu + nv :].reshape(Ny, Nx)
    # u vanishes on the left/right walls, v on the bottom/top walls.
    u = jnp.zeros((Ny, Nx + 1)).at[:, 1:Nx].set(u_int)
    v = jnp.zeros((Ny + 1, Nx)).at[1:Ny, :].set(v_int)
    return u, v, p


def n_fluid_unknowns(cfg: Config) -> int:
    return cfg.Ny * (cfg.Nx - 1) + (cfg.Ny - 1) * cfg.Nx + cfg.Nx * cfg.Ny


def fluid_residual(w: jnp.ndarray, T: jnp.ndarray, alpha: jnp.ndarray, cfg: Config):
    """Residual of the Stokes-Brinkman-Boussinesq system. Linear in w."""
    Nx, Ny, h = cfg.Nx, cfg.Ny, cfg.h
    u, v, p = _unpack_fluid(w, cfg)

    # alpha lives at cell centres; interpolate onto the velocity faces.
    alpha_u = jnp.zeros((Ny, Nx + 1))
    alpha_u = alpha_u.at[:, 1:Nx].set(0.5 * (alpha[:, : Nx - 1] + alpha[:, 1:]))
    alpha_v = jnp.zeros((Ny + 1, Nx))
    alpha_v = alpha_v.at[1:Ny, :].set(0.5 * (alpha[: Ny - 1, :] + alpha[1:, :]))

    # --- x-momentum on interior vertical faces i = 1 .. Nx-1 ---
    uc = u[:, 1:Nx]
    u_xm = u[:, 0 : Nx - 1]
    u_xp = u[:, 2 : Nx + 1]
    # no-slip on bottom/top uses the reflected ghost value u_ghost = -u.
    u_ym = jnp.concatenate([-uc[0:1, :], uc[: Ny - 1, :]], axis=0)
    u_yp = jnp.concatenate([uc[1:Ny, :], -uc[Ny - 1 : Ny, :]], axis=0)
    lap_u = (u_xm - 2 * uc + u_xp) / h**2 + (u_ym - 2 * uc + u_yp) / h**2
    dpdx = (p[:, 1:Nx] - p[:, 0 : Nx - 1]) / h
    Ru = -cfg.Pr * lap_u + dpdx + cfg.Pr * alpha_u[:, 1:Nx] * uc
    if cfg.inertia:
        # (u.grad)u on the u-faces. Central differences reusing the same
        # reflected ghosts as the Laplacian, so the wall treatment is
        # consistent between the two terms. v is averaged from the four
        # surrounding v-faces.
        v_at_u = 0.25 * (
            v[0:Ny, 0 : Nx - 1] + v[0:Ny, 1:Nx]
            + v[1 : Ny + 1, 0 : Nx - 1] + v[1 : Ny + 1, 1:Nx]
        )
        Ru = Ru + cfg.inertia * (
            uc * (u_xp - u_xm) / (2 * h) + v_at_u * (u_yp - u_ym) / (2 * h)
        )

    # --- y-momentum on interior horizontal faces j = 1 .. Ny-1 ---
    vc = v[1:Ny, :]
    v_ym = v[0 : Ny - 1, :]
    v_yp = v[2 : Ny + 1, :]
    v_xm = jnp.concatenate([-vc[:, 0:1], vc[:, : Nx - 1]], axis=1)
    v_xp = jnp.concatenate([vc[:, 1:Nx], -vc[:, Nx - 1 : Nx]], axis=1)
    lap_v = (v_xm - 2 * vc + v_xp) / h**2 + (v_ym - 2 * vc + v_yp) / h**2
    dpdy = (p[1:Ny, :] - p[0 : Ny - 1, :]) / h
    T_face = 0.5 * (T[0 : Ny - 1, :] + T[1:Ny, :])  # T interpolated to v-faces
    Rv = (
        -cfg.Pr * lap_v
        + dpdy
        + cfg.Pr * alpha_v[1:Ny, :] * vc
        - cfg.Ra * cfg.Pr * T_face
    )
    if cfg.inertia:
        u_at_v = 0.25 * (
            u[0 : Ny - 1, 0:Nx] + u[0 : Ny - 1, 1 : Nx + 1]
            + u[1:Ny, 0:Nx] + u[1:Ny, 1 : Nx + 1]
        )
        Rv = Rv + cfg.inertia * (
            u_at_v * (v_xp - v_xm) / (2 * h) + vc * (v_yp - v_ym) / (2 * h)
        )

    # --- continuity at cell centres ---
    div = (u[:, 1 : Nx + 1] - u[:, 0:Nx]) / h + (v[1 : Ny + 1, :] - v[0:Ny, :]) / h
    # All-Dirichlet velocity leaves pressure defined only up to a constant;
    # pin one cell to make the system nonsingular.
    Rp = (-div).at[0, 0].set(p[0, 0])

    return jnp.concatenate([Ru.ravel(), Rv.ravel(), Rp.ravel()])


@functools.partial(jax.jit, static_argnums=1)
def assemble_fluid(alpha: jnp.ndarray, cfg: Config):
    """Assemble the fluid system matrix.

    Crucially this depends on the design only, *not* on temperature: buoyancy
    enters the rhs, not the operator. So it can be factorised once and reused
    across every Picard sweep, which is the single biggest cost saving here.
    """
    n = n_fluid_unknowns(cfg)
    T0 = jnp.zeros((cfg.Ny, cfg.Nx))
    res0 = lambda w: fluid_residual(w, T0, alpha, cfg)  # noqa: E731
    return jax.jacfwd(res0)(jnp.zeros(n))


@functools.partial(jax.jit, static_argnums=2)
def fluid_rhs(T: jnp.ndarray, alpha: jnp.ndarray, cfg: Config):
    """Buoyancy rhs. Affine in T, and zero when T is zero."""
    return -fluid_residual(jnp.zeros(n_fluid_unknowns(cfg)), T, alpha, cfg)


@functools.partial(jax.jit, static_argnums=2)
def solve_fluid(T: jnp.ndarray, alpha: jnp.ndarray, cfg: Config):
    """Solve the fluid block for the given temperature and design.

    Without inertia the block is linear in w and this is a single solve. With
    inertia it is the steady Navier-Stokes-Boussinesq problem, so we take the
    Stokes solution as the initial guess and run Newton on the full residual.
    The Stokes start is a good one precisely because the convective term is
    quadratic: it vanishes at w = 0 and is small wherever the flow is slow.
    """
    A = assemble_fluid(alpha, cfg)
    w = jnp.linalg.solve(A, fluid_rhs(T, alpha, cfg))
    if cfg.inertia:
        # Converge with derivatives switched off. `lax.while_loop` has no
        # reverse-mode rule, and even where it does, differentiating the
        # iteration history is the wrong thing: the derivative of a converged
        # fixed point depends on where it landed, not how it got there.
        w = jax.lax.stop_gradient(
            _newton_fluid(jax.lax.stop_gradient(w), jax.lax.stop_gradient(T),
                          jax.lax.stop_gradient(alpha), cfg)
        )
        # One differentiable Newton correction taken at the converged state.
        # Because R(w*) is at the solver floor, this leaves the value alone but
        # carries the exact implicit derivative dw = -J^-1 dR/dparams -- the
        # same quantity the C++ component obtains from its hand-derived
        # Jacobian, which is what the test suite compares.
        resid = lambda ww: fluid_residual(ww, T, alpha, cfg)  # noqa: E731
        w = w - jnp.linalg.solve(jax.jacfwd(resid)(w), resid(w))
    return _unpack_fluid(w, cfg)


def _newton_fluid(w0, T, alpha, cfg: Config, tol: float = 1e-12,
                  max_iter: int = 40):
    """Newton on the full nonlinear fluid residual, exact Jacobian by autodiff.

    This is the reference implementation, so the Jacobian is taken by
    `jax.jacfwd` rather than derived. That is the entire point: the C++
    component derives the same Jacobian by hand, and the test suite checks the
    two against each other to machine precision.
    """
    resid = lambda ww: fluid_residual(ww, T, alpha, cfg)  # noqa: E731

    def cond(state):
        _, it, r = state
        return (r > tol) & (it < max_iter)

    def body(state):
        w, it, _ = state
        R = resid(w)
        J = jax.jacfwd(resid)(w)
        w_new = w - jnp.linalg.solve(J, R)
        return w_new, it + 1, jnp.max(jnp.abs(resid(w_new)))

    w, _, _ = jax.lax.while_loop(cond, body, (w0, 0, jnp.inf))
    return w


# --------------------------------------------------------------------------
# thermal block: advection-diffusion
# --------------------------------------------------------------------------


def thermal_residual(
    T: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    k: jnp.ndarray,
    cfg: Config,
):
    """Residual of div(uT) - div(k grad T) - source. Linear in T."""
    Nx, Ny, h = cfg.Nx, cfg.Ny, cfg.h

    # --- advection, conservative form, smooth Peclet-weighted face values ---
    # A hard upwind switch makes this residual non-differentiable wherever a
    # face velocity crosses zero, which stalls Newton and puts a kink in the
    # gradient path. Blend upwind -> central as the local Peclet number falls;
    # where |Pe| is small diffusion dominates anyway, so it costs nothing.
    # Wall-normal velocities vanish, so the padded values never contribute.
    kfx = jnp.zeros((Ny, Nx + 1))
    kfx = kfx.at[:, 1:Nx].set(0.5 * (k[:, 0 : Nx - 1] + k[:, 1:Nx]))
    kfx = kfx.at[:, 0].set(k[:, 0]).at[:, Nx].set(k[:, Nx - 1])
    kfy = jnp.zeros((Ny + 1, Nx))
    kfy = kfy.at[1:Ny, :].set(0.5 * (k[0 : Ny - 1, :] + k[1:Ny, :]))
    kfy = kfy.at[0, :].set(k[0, :]).at[Ny, :].set(k[Ny - 1, :])

    w_x = 0.5 * (1.0 + jnp.tanh(0.5 * u * h / jnp.maximum(kfx, 1e-12)))
    w_y = 0.5 * (1.0 + jnp.tanh(0.5 * v * h / jnp.maximum(kfy, 1e-12)))

    Tp_x = jnp.concatenate([T[:, 0:1], T, T[:, Nx - 1 : Nx]], axis=1)
    Fx = u * (w_x * Tp_x[:, 0 : Nx + 1] + (1.0 - w_x) * Tp_x[:, 1 : Nx + 2])
    Tp_y = jnp.concatenate([T[0:1, :], T, T[Ny - 1 : Ny, :]], axis=0)
    Fy = v * (w_y * Tp_y[0 : Ny + 1, :] + (1.0 - w_y) * Tp_y[1 : Ny + 2, :])

    adv = (Fx[:, 1 : Nx + 1] - Fx[:, 0:Nx]) / h + (Fy[1 : Ny + 1, :] - Fy[0:Ny, :]) / h

    # --- diffusion with arithmetic-mean face conductivities ---
    kx = 0.5 * (k[:, 0 : Nx - 1] + k[:, 1:Nx])  # (Ny, Nx-1) interior x-faces
    ky = 0.5 * (k[0 : Ny - 1, :] + k[1:Ny, :])  # (Ny-1, Nx) interior y-faces

    qx = jnp.zeros((Ny, Nx + 1))
    qx = qx.at[:, 1:Nx].set(-kx * (T[:, 1:Nx] - T[:, 0 : Nx - 1]) / h)
    # left/right walls adiabatic -> flux stays zero

    qy = jnp.zeros((Ny + 1, Nx))
    qy = qy.at[1:Ny, :].set(-ky * (T[1:Ny, :] - T[0 : Ny - 1, :]) / h)
    # Bottom wall: chip heat flux entering the domain. The flux vector points
    # up (+y) into the cell, so this is +q_chip, not -q_chip: with div(q) in
    # the residual, a positive q_y at the bottom face *adds* energy.
    qy = qy.at[0, :].set(cfg.q_chip * cfg.chip_mask())
    # top wall: cold sink at T = 0, half-cell distance
    qy = qy.at[Ny, :].set(-k[Ny - 1, :] * (0.0 - T[Ny - 1, :]) / (0.5 * h))

    diff = (qx[:, 1 : Nx + 1] - qx[:, 0:Nx]) / h + (qy[1 : Ny + 1, :] - qy[0:Ny, :]) / h

    return (adv + diff).ravel()


@functools.partial(jax.jit, static_argnums=3)
def solve_thermal(u: jnp.ndarray, v: jnp.ndarray, k: jnp.ndarray, cfg: Config):
    """Solve the (linear) thermal block for the given velocity and design.

    Unlike the fluid operator this one *does* change every sweep, because the
    advection term carries the velocity, so it is rebuilt each time.
    """
    zero = jnp.zeros(cfg.Nx * cfg.Ny)
    res = lambda T_flat: thermal_residual(  # noqa: E731
        T_flat.reshape(cfg.Ny, cfg.Nx), u, v, k, cfg
    )
    A = jax.jacfwd(res)(zero)
    b = -res(zero)
    return jnp.linalg.solve(A, b).reshape(cfg.Ny, cfg.Nx)


# --------------------------------------------------------------------------
# coupled fixed point
# --------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=4)
def coupled_step_factored(T, lu_and_piv, alpha, k, cfg: Config):
    """One Picard sweep reusing a prefactorised fluid operator.

    T -> buoyancy rhs -> flow -> advection -> T. This is the map Phi whose
    fixed point is the steady coupled state and whose Jacobian drives the
    adjoint. Note that a single evaluation crosses *both* physics blocks.
    """
    lu, piv = lu_and_piv
    w = jax.scipy.linalg.lu_solve((lu, piv), fluid_rhs(T, alpha, cfg))
    u, v, _ = _unpack_fluid(w, cfg)
    return solve_thermal(u, v, k, cfg)


@functools.partial(jax.jit, static_argnums=1)
def coupled_step(T: jnp.ndarray, cfg: Config, rho: jnp.ndarray):
    """Unfactored convenience form of Phi, for gradient checks."""
    k, alpha = material_maps(rho, cfg)
    u, v, _ = solve_fluid(T, alpha, cfg)
    return solve_thermal(u, v, k, cfg)


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def _picard(rho, cfg: Config, max_iter: int, relax: float):
    """Damped Picard iteration, fully inside one jit."""
    k, alpha = material_maps(rho, cfg)
    lu_piv = jax.scipy.linalg.lu_factor(assemble_fluid(alpha, cfg))

    def body(carry, _):
        T, _prev = carry
        T_new = relax * coupled_step_factored(T, lu_piv, alpha, k, cfg) + (
            1.0 - relax
        ) * T
        return (T_new, jnp.max(jnp.abs(T_new - T))), jnp.max(jnp.abs(T_new - T))

    (T, _), hist = jax.lax.scan(
        body, (jnp.zeros((cfg.Ny, cfg.Nx)), jnp.inf), None, length=max_iter
    )
    return T, hist


def solve_coupled(
    rho: jnp.ndarray,
    cfg: Config,
    max_iter: int = 200,
    relax: float = 0.7,
):
    """Damped Picard iteration to the coupled steady state.

    Relaxation is needed because the buoyancy feedback is destabilising once Ra
    gets large. Returns the converged temperature plus diagnostics.
    """
    T, hist = _picard(rho, cfg, max_iter, relax)
    hist = list(map(float, hist))
    converged = [i for i, r in enumerate(hist) if r < 1e-10]
    return T, {
        "iters": (converged[0] + 1) if converged else max_iter,
        "residual": hist[-1],
        "history": hist,
    }


# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------


def objective(T: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Mean temperature over the chip strip -- the thing we want to minimise."""
    mask = cfg.chip_mask()
    return jnp.sum(T[0, :] * mask) / jnp.sum(mask)


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def loss_unrolled(rho: jnp.ndarray, cfg: Config, n_iter: int = 60, relax: float = 0.7):
    """Objective computed by unrolling the Picard loop.

    Differentiating *this* is the brute-force reference gradient: correct, but
    it tapes every sweep and so costs memory linear in n_iter. It is exactly
    what implicit differentiation replaces -- and a useful cross-check on it.
    """
    k, alpha = material_maps(rho, cfg)
    lu_piv = jax.scipy.linalg.lu_factor(assemble_fluid(alpha, cfg))

    def body(T, _):
        return relax * coupled_step_factored(T, lu_piv, alpha, k, cfg) + (
            1.0 - relax
        ) * T, None

    T, _ = jax.lax.scan(body, jnp.zeros((cfg.Ny, cfg.Nx)), None, length=n_iter)
    return objective(T, cfg)
