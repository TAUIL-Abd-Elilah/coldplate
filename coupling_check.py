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

so, expanding,

    lambda = g + Phi_x^T g + (Phi_x^T)^2 g + ...

Differentiating the components separately -- treating the pipeline as
feed-forward -- keeps only the first term. The leading error is therefore
Phi_x^T g, and the natural relative measure is

    gamma = || Phi_x^T g || / || g ||

which is ONE VJP through the loop.

Why not the spectral radius. rho(Phi_x) is the obvious candidate and it is the
wrong one: it is a worst case over all directions, and a large gain along
directions your objective never excites costs you nothing. Measured on a
coupled Boussinesq problem across four design families and five Rayleigh
numbers, log(gamma) correlates with log(relative error) at 0.995 while
rho(Phi_x) manages 0.825 -- and rho orders some pairs backwards, calling a
configuration safe when it has twice the error of one it calls dangerous.

Empirically the relative error of the component-wise gradient is approximately
gamma while gamma is small, and grows faster once the higher terms bite.

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

# Thresholds are guidance, not theory. They come from the measured relationship
# between gamma and the observed error on the problem in this repository.
SAFE = 0.01  # component-wise gradient good to roughly a percent
RISKY = 0.10  # above this it is not worth having


@dataclass
class CouplingReport:
    gamma: float
    verdict: str
    detail: str
    neumann_terms: list[float]

    def __str__(self) -> str:
        terms = ", ".join(f"{t:.3e}" for t in self.neumann_terms)
        return (
            f"coupling gamma = {self.gamma:.4g}  [{self.verdict}]\n"
            f"  {self.detail}\n"
            f"  Neumann terms ||(Phi_x^T)^k g||/||g||: {terms}"
        )


def coupling_gamma(phi, x_star, g, n_terms: int = 4) -> CouplingReport:
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
        How many Neumann terms to report. Only the first is needed for gamma;
        the rest show whether the series is decaying quickly or grinding down.

    Returns
    -------
    CouplingReport
    """
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
    if gamma < SAFE:
        verdict = "SAFE"
        detail = (f"differentiating the components separately should cost about "
                  f"{100*gamma:.2g}% -- below the {100*SAFE:.0f}% guidance threshold")
    elif gamma < RISKY:
        verdict = "MARGINAL"
        detail = (f"expect a component-wise gradient to be roughly {100*gamma:.0f}% "
                  f"wrong; usable as a search direction, not as a sensitivity")
    else:
        verdict = "UNSAFE"
        detail = (f"a component-wise gradient will be badly wrong (>= {100*gamma:.0f}%), "
                  f"and individual sensitivities may have the wrong sign")
    return CouplingReport(gamma=gamma, verdict=verdict, detail=detail,
                          neumann_terms=terms)


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
