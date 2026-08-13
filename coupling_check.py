# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Can I differentiate my coupled components separately?

A standalone check, not specific to this project. If you have two solvers that
feed each other -- so the steady state is a fixed point rather than a chain --
this answers whether you can get away with differentiating them in isolation,
for the cost of a single vector-Jacobian product.

The reasoning. Let the coupling loop be a map Phi with fixed point x* =
Phi(x*, theta), and let J be an objective on x*. The exact adjoint solves

    (I - Phi_x)^T lambda = g,        g = dJ/dx

The component-wise approximation is lambda_0 = g. Its residual in the exact
adjoint equation is *exactly*

    r_0 = g - (I - Phi_x)^T lambda_0 = Phi_x^T g.

The natural relative residual is therefore

    gamma = || Phi_x^T g || / || g ||

which is ONE VJP through the loop. Whenever the adjoint system is invertible,
the actual adjoint error is

    lambda - lambda_0 = (I - Phi_x^T)^-1 r_0.

Thus gamma is a cheap warning signal, not a universal error bound: converting
residual to adjoint error also depends on the conditioning of the coupled
system, and converting that to design-gradient error also depends on the
parameter map.

If rho(Phi_x) < 1, the inverse may additionally be written as a convergent
Neumann series. That expansion is *not* valid when rho(Phi_x) >= 1, even if
Newton's method still reaches a perfectly valid fixed point.

Why not the spectral radius. rho(Phi_x) is the obvious candidate and it is the
wrong one: it is a worst case over all directions, and a large gain along
directions your objective never excites costs you nothing. Measured on 14
converged configurations drawn from four design families and five attempted
Rayleigh levels, log(gamma) correlates with log(relative error) at 0.995 while
rho(Phi_x) manages 0.825 -- and rho orders some pairs backwards.

On the benchmark shipped here, relative error is approximately gamma while
gamma is small. The thresholds below are benchmark-calibrated guidance, not
universal guarantees.

Usage with any JAX-traceable loop::

    from coupling_check import coupling_gamma
    report = coupling_gamma(phi, x_star, g)
    print(report)

`phi` maps the coupled state to itself (one trip round the loop) and may call
served Tesseracts inside; `x_star` is the converged state; `g` is dJ/dx there.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

# Benchmark-calibrated defaults, not theory or universal guarantees. Callers
# with their own validation data should pass thresholds appropriate to it.
DEFAULT_SAFE = 0.01
DEFAULT_RISKY = 0.10


@dataclass
class CouplingReport:
    gamma: float
    verdict: str
    detail: str
    repeated_vjp_norms: list[float]

    @property
    def neumann_terms(self) -> list[float]:
        """Backward-compatible alias; these are not always a convergent series."""
        return self.repeated_vjp_norms

    def __str__(self) -> str:
        terms = ", ".join(f"{t:.3e}" for t in self.repeated_vjp_norms)
        return (
            f"coupling gamma = {self.gamma:.4g}  [{self.verdict}]\n"
            f"  {self.detail}\n"
            f"  repeated VJP norms ||(Phi_x^T)^k g||/||g||: {terms}"
        )


def coupling_gamma(
    phi,
    x_star,
    g,
    n_terms: int = 4,
    safe_threshold: float = DEFAULT_SAFE,
    risky_threshold: float = DEFAULT_RISKY,
) -> CouplingReport:
    """Measure how much of the adjoint a component-wise gradient would miss.

    Parameters
    ----------
    phi
        One trip around the coupling loop, x -> Phi(x). Must be traceable by
        JAX; it may call served Tesseracts internally.
    x_star
        The converged coupled state. gamma is a local quantity, so this should
        be an actual fixed point, not an arbitrary iterate.
    g
        dJ/dx at x_star -- the objective's own sensitivity to the coupled
        state. gamma depends on this direction, which is the whole point.
    n_terms
        How many repeated VJP norms to report. Only the first is needed for
        gamma. The rest are diagnostics; they are not terms of a convergent
        Neumann series unless rho(Phi_x) < 1.
    safe_threshold, risky_threshold
        Benchmark-calibrated guidance thresholds. They must satisfy
        ``0 <= safe_threshold < risky_threshold``.

    Returns
    -------
    CouplingReport
    """
    if not 0 <= safe_threshold < risky_threshold:
        raise ValueError("thresholds must satisfy 0 <= safe < risky")

    _, vjp_fn = jax.vjp(phi, x_star)
    g = jnp.asarray(g)
    g_norm = float(jnp.linalg.norm(g))
    if g_norm == 0.0:
        raise ValueError("dJ/dx is identically zero; gamma is undefined")

    terms, w = [], g
    for _ in range(max(1, n_terms)):
        (w,) = vjp_fn(w)
        terms.append(float(jnp.linalg.norm(w)) / g_norm)

    gamma = terms[0]
    if gamma < safe_threshold:
        verdict = "SAFE"
        detail = (f"normalized adjoint residual is {100*gamma:.2g}% -- below the "
                  f"benchmark-calibrated {100*safe_threshold:.0f}% threshold")
    elif gamma < risky_threshold:
        verdict = "MARGINAL"
        detail = (f"normalized adjoint residual is {100*gamma:.0f}%; validate the "
                  "component-wise gradient before using it as a sensitivity")
    else:
        verdict = "UNSAFE"
        detail = (f"normalized adjoint residual is {100*gamma:.0f}%, above the "
                  f"benchmark-calibrated {100*risky_threshold:.0f}% risk threshold")
    return CouplingReport(gamma=gamma, verdict=verdict, detail=detail,
                          repeated_vjp_norms=terms)


def spectral_radius(phi, x_star, n_power: int = 40, seed: int = 0) -> float:
    """rho(Phi_x) by power iteration, for comparison.

    Provided so the two can be measured side by side. It is *not* the
    recommended statistic -- see the module docstring.
    """
    v = jnp.asarray(np.random.default_rng(seed).normal(size=jnp.shape(x_star)))
    v = v / jnp.linalg.norm(v)
    lam = 0.0
    for _ in range(n_power):
        w = jax.jvp(phi, (x_star,), (v,))[1]
        nrm = float(jnp.linalg.norm(w))
        if nrm < 1e-300:
            return 0.0
        lam, v = nrm, w / nrm
    return lam
