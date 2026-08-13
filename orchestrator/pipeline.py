# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Compose three Tesseracts into one end-to-end differentiable cold-plate model.

    rho_raw  --[material_map : PyTorch]-->  (k, alpha, rho_phys)
                                                |
                          +---------------------+
                          |                     |
                          v                     v
        T --[stokes_brinkman : C++]--> (u,v) --[thermal_advdiff : JAX]--> T
                          ^                                               |
                          +-----------------------------------------------+
                                     buoyancy / advection feedback

The inner loop is a genuine fixed point: buoyancy makes temperature drive the
flow, advection makes the flow drive temperature. So the steady state is not a
feed-forward chain and its sensitivity cannot be assembled block by block.

The gradient uses the implicit function theorem at T* = Phi(T*, theta):

    dJ/dtheta = (dPhi/dtheta)^T lambda ,   (I - dPhi/dT)^T lambda = dJ/dT

The adjoint system is solved with GMRES. Every matvec applies (I - Phi_T)^T,
which runs a VJP back through the JAX thermal Tesseract and then a VJP back
through the C++ fluid Tesseract -- i.e. the Krylov iteration bounces across the
component boundary on every single step. There is no factoring of that work
into per-component pieces, which is precisely the claim this project makes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as spla
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)


@dataclass
class Params:
    """Physical and numerical settings for one cold-plate problem."""

    Nx: int = 48
    Ny: int = 48
    Ra: float = 3.0e4  # operating point: strong enough coupling that naive
    Pr: float = 7.0  # component-wise gradients point the wrong way
    q_chip: float = 1.0
    chip_frac: float = 0.4
    # 0 = cold-plate (chip heat flux on part of the bottom wall, the design
    # problem); 1 = Rayleigh-Benard (isothermal hot bottom wall, used to check
    # the coupled physics against the classical critical Rayleigh number).
    bc_mode: float = 0.0
    t_hot: float = 1.0
    # Which functional of the coupled state is being minimised. Changing this
    # changes dJ/dT, and therefore the directional gain, without touching the
    # physics or rho(Phi_T).
    objective: str = "chip_mean"
    filter_radius: float = 2.0
    beta: float = 1.0
    eta: float = 0.5
    penal: float = 3.0
    k_solid: float = 1.0
    k_fluid: float = 0.02
    alpha_max: float = 1.0e5
    volume_fraction: float = 0.35  # budget of solid material

    def chip_mask(self) -> np.ndarray:
        i = np.arange(self.Nx)
        lo = 0.5 * (1.0 - self.chip_frac) * self.Nx
        hi = 0.5 * (1.0 + self.chip_frac) * self.Nx
        return ((i >= lo) & (i < hi)).astype(np.float64)


@dataclass
class ColdPlate:
    """Holds the three served Tesseracts and drives the coupled solve."""

    params: Params = field(default_factory=Params)
    images: dict[str, str] = field(
        default_factory=lambda: {
            "material": "material_map",
            "fluid": "stokes_brinkman",
            "thermal": "thermal_advdiff",
        }
    )
    verbose: bool = False
    _t: dict = field(default_factory=dict, init=False)
    _T_warm: Any = field(default=None, init=False)  # warm start for the next solve
    stats: dict = field(
        default_factory=lambda: {
            "phi_calls": 0,
            "vjp_matvecs": 0,
            "jvp_matvecs": 0,
            # gamma-gated mode only: how many decisions were taken and which way
            "gamma_vjps": 0,
            "gate_cheap": 0,
            "gate_exact": 0,
        },
        init=False,
    )

    def __enter__(self):
        for name, image in self.images.items():
            t = Tesseract.from_image(image)
            t.serve()
            self._t[name] = t
        return self

    def __exit__(self, *exc):
        for t in self._t.values():
            try:
                t.teardown()
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass

    # -- individual components -------------------------------------------

    def material(self, rho_raw):
        p = self.params
        return apply_tesseract(
            self._t["material"],
            {
                "rho_raw": rho_raw,
                "filter_radius": p.filter_radius,
                "beta": p.beta,
                "eta": p.eta,
                "penal": p.penal,
                "k_solid": p.k_solid,
                "k_fluid": p.k_fluid,
                "alpha_max": p.alpha_max,
            },
        )

    def phi(self, T, alpha, k):
        """One trip around the coupling loop: T -> (u,v) -> T. Crosses C++/JAX."""
        p = self.params
        flow = apply_tesseract(
            self._t["fluid"], {"alpha": alpha, "T": T, "Ra": p.Ra, "Pr": p.Pr}
        )
        out = apply_tesseract(
            self._t["thermal"],
            {
                "u": flow["u"],
                "v": flow["v"],
                "k": k,
                "q_chip": p.q_chip,
                "chip_frac": p.chip_frac,
                "bc_mode": p.bc_mode,
                "t_hot": p.t_hot,
            },
        )
        return out["T"]

    # -- coupled solve ----------------------------------------------------

    def solve_coupled(self, alpha, k, tol=1e-10, max_newton=30, gmres_tol=1e-10):
        """Newton-Krylov solve of the fixed point F(T) = Phi(T) - T = 0.

        Picard and Anderson both stall once the coupling loop gain approaches 1,
        and the filtered designs this optimiser actually visits sit right there:
        a near-uniform permeable domain lets convection cells span the cavity,
        which is exactly when the feedback is strongest. Newton does not care --
        it only needs the linearisation to be invertible, not contractive.

        Each GMRES matvec applies (Phi_T - I) through jax.jvp, running forward
        through the C++ fluid block and then the JAX thermal block. So the
        forward solve crosses the component boundary once per matvec, mirroring
        the adjoint solve which crosses it once per VJP.
        """
        import time as _time

        shape = (self.params.Ny, self.params.Nx)
        n = shape[0] * shape[1]
        # Warm start from the previous design's steady state. Successive designs
        # differ only slightly, so this typically cuts the Newton count from
        # ~11 to ~3 -- and since every Newton step costs ~26 container
        # round-trips, that is most of the wall-clock cost of an optimisation.
        T = self._T_warm if self._T_warm is not None and self._T_warm.shape == shape \
            else jnp.zeros(shape)
        res = np.inf

        for it in range(1, max_newton + 1):
            t_it = _time.time()
            phi_val, jvp_lin = jax.linearize(lambda t: self.phi(t, alpha, k), T)
            self.stats["phi_calls"] += 1
            F = phi_val - T
            res = float(jnp.max(jnp.abs(F)))
            if res < tol:
                self._T_warm = T
                return T, {"iters": it, "residual": res, "ok": True}

            nmv = [0]

            def matvec(x, _jvp=jvp_lin, _n=nmv):
                _n[0] += 1
                d = jnp.asarray(x.reshape(shape))
                return np.array(_jvp(d) - d, dtype=np.float64).ravel()

            op = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float64)
            # Inexact Newton with an Eisenstat-Walker style forcing term: solve
            # the Newton system only as tightly as the current residual
            # warrants. Far from the solution an exact step is wasted work, and
            # here every matvec is two container round-trips, so a fixed tight
            # tolerance made GMRES hit its iteration cap on every step.
            eta = float(min(0.1, max(gmres_tol, 0.5 * np.sqrt(min(res, 1.0)))))
            delta, _ = spla.gmres(
                op, np.array(-F, dtype=np.float64).ravel(),
                rtol=eta, atol=gmres_tol, restart=30, maxiter=60,
            )
            self.stats["jvp_matvecs"] += nmv[0]
            if self.verbose:
                print(
                    f"      newton {it:2d}  |F|={res:.3e}  eta={eta:.1e}  "
                    f"gmres_matvecs={nmv[0]:3d}  ({_time.time()-t_it:.1f}s)",
                    flush=True,
                )
            d = jnp.asarray(delta.reshape(shape))

            # Backtracking line search. Only ever accept a step that actually
            # reduces the residual -- an earlier version fell through and took
            # an untested, further-halved step when the search failed, which
            # let |F| jump back up by an order of magnitude and made the whole
            # Newton loop thrash instead of converge.
            # Merit function is the 2-norm, while convergence is tested on the
            # inf-norm. Using inf-norm for both is too harsh: a good step often
            # reduces the overall residual while raising a single worst cell,
            # which rejects the step and stalls Newton on designs that are
            # perfectly solvable.
            merit = float(jnp.linalg.norm(F))
            step, accepted = 1.0, False
            for _ in range(6):
                T_try = T + step * d
                F_try = self.phi(T_try, alpha, k) - T_try
                self.stats["phi_calls"] += 1
                if float(jnp.linalg.norm(F_try)) < merit:
                    T = T_try
                    res = float(jnp.max(jnp.abs(F_try)))
                    accepted = True
                    break
                step *= 0.5

            if not accepted:
                # No step along this direction improves the residual. Either we
                # are at the accuracy floor, or (for near-uniform designs at
                # high Ra) there is no reachable steady state to converge to.
                # Either way, further iterations only burn container calls.
                ok = res < tol
                self._T_warm = T if res < 1e-6 else None
                return T, {"iters": it, "residual": res, "ok": ok,
                           "stalled": not ok}

        self._T_warm = T if res < 1e-6 else None  # don't poison the next solve
        return T, {"iters": max_newton, "residual": res, "ok": res < tol}

    def objective(self, T):
        """The quantity being minimised.

        Selectable because the directional gain gamma depends on dJ/dT, so
        changing the objective at a *fixed* physical state changes gamma while
        leaving rho(Phi_T) untouched by construction. That is the cleanest way
        to test whether the error of a component-wise gradient is governed by
        the direction or by the spectral radius -- see objective_sweep.py.
        """
        p = self.params
        mask = jnp.asarray(p.chip_mask())
        kind = p.objective

        if kind == "chip_mean":
            return jnp.sum(T[0, :] * mask) / jnp.sum(mask)
        if kind == "chip_peak":
            # smooth max over the chip strip (p-norm)
            t = T[0, :] * mask
            return (jnp.sum(t**8) / jnp.sum(mask)) ** (1.0 / 8.0)
        if kind == "domain_mean":
            return jnp.mean(T)
        if kind == "top_half_mean":
            return jnp.mean(T[p.Ny // 2 :, :])
        if kind == "left_column_mean":
            return jnp.mean(T[:, : max(1, p.Nx // 4)])
        if kind == "outlet_mean":
            # mean over the top row: what a downstream heat exchanger would see
            return jnp.mean(T[p.Ny - 1, :])
        raise ValueError(f"unknown objective {kind!r}")

    # -- end-to-end value and gradient ------------------------------------

    def value_and_grad(self, rho_raw, gmres_tol=1e-10, mode="composed",
                       gamma_gate=0.01):
        """J(rho_raw) and dJ/drho_raw through the whole coupled pipeline.

        mode selects how the gradient is obtained, so the optimiser can be
        driven by a deliberately wrong one for comparison without paying for
        the exact adjoint it is not going to use:
            "composed"    -- implicit differentiation of the fixed point (correct)
            "one_way"     -- full chain, feedback loop cut       (naive, strong)
            "frozen"      -- velocity held constant              (naive, weak)
            "gamma_gated" -- measure the directional gain first, then decide

        The last one is what the diagnostic is *for*. gamma = ||Phi_T^T g||/||g||
        is the normalized residual of the loop-cut adjoint lambda_0 = g. It is
        therefore an inexpensive, objective-aware warning signal: one VJP,
        against the tens of VJPs the GMRES adjoint needs. It is not a universal
        error bound; the configured gate is benchmark-calibrated. Measuring it
        before deciding turns a claim about accuracy into a budget decision --
        pay for the exact adjoint on the iterations that need it, take the cheap
        gradient on the ones that do not.

        Note the accounting is honest about the gate not being free: the VJP
        that produces gamma is counted, and on iterations where the gate opens
        it is pure overhead on top of the full adjoint.
        """
        rho_raw = jnp.asarray(rho_raw)
        n = rho_raw.size
        shape = rho_raw.shape

        # 1. design -> properties (PyTorch Tesseract), keeping its VJP around
        mat, mat_vjp = jax.vjp(self.material, rho_raw)
        alpha, k, rho_phys = mat["alpha"], mat["k"], mat["rho_phys"]

        # 2. converge the coupled state
        T_star, info = self.solve_coupled(alpha, k)

        if mode in ("one_way", "frozen"):
            naive = self.one_way_grad if mode == "one_way" else self.frozen_flow_grad
            p = self.params
            flow = apply_tesseract(
                self._t["fluid"], {"alpha": alpha, "T": T_star, "Ra": p.Ra, "Pr": p.Pr}
            )
            return {
                "J": float(self.objective(T_star)),
                "grad": naive(rho_raw, _state=(mat, mat_vjp, alpha, k, T_star)),
                "T": np.asarray(T_star),
                "u": np.asarray(flow["u"]),
                "v": np.asarray(flow["v"]),
                "rho_phys": np.asarray(rho_phys),
                "info": info,
            }

        # 3. adjoint of the fixed point, solved across the component boundary
        _, phi_vjp = jax.vjp(lambda T, a, kk: self.phi(T, a, kk), T_star, alpha, k)
        g = jax.grad(self.objective)(T_star)

        gamma = None
        if mode == "gamma_gated":
            # One VJP through the loop gives the loop-cut adjoint residual.
            # This is the entire cost of the calibrated decision.
            self.stats["gamma_vjps"] = self.stats.get("gamma_vjps", 0) + 1
            w = phi_vjp(g)[0]
            gnorm = float(jnp.linalg.norm(g))
            gamma = float(jnp.linalg.norm(w)) / gnorm if gnorm > 0 else 0.0
            if gamma < gamma_gate:
                # Cheap path: the loop contributes less than the gate, so the
                # component-wise gradient is within tolerance and the GMRES
                # adjoint would be money spent for nothing.
                self.stats["gate_cheap"] = self.stats.get("gate_cheap", 0) + 1
                p = self.params
                flow = apply_tesseract(
                    self._t["fluid"],
                    {"alpha": alpha, "T": T_star, "Ra": p.Ra, "Pr": p.Pr},
                )
                return {
                    "J": float(self.objective(T_star)),
                    "grad": self.one_way_grad(
                        rho_raw, _state=(mat, mat_vjp, alpha, k, T_star)
                    ),
                    "T": np.asarray(T_star),
                    "u": np.asarray(flow["u"]),
                    "v": np.asarray(flow["v"]),
                    "rho_phys": np.asarray(rho_phys),
                    "info": info,
                    "gamma": gamma,
                    "gate": "cheap",
                }
            self.stats["gate_exact"] = self.stats.get("gate_exact", 0) + 1

        def matvec(x):
            self.stats["vjp_matvecs"] += 1
            lam = jnp.asarray(x.reshape(shape))
            # np.array (not asarray): JAX buffers convert to read-only views and
            # scipy's GMRES updates its Krylov vectors in place.
            return np.array(lam - phi_vjp(lam)[0], dtype=np.float64).ravel()

        op = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float64)
        rhs_vec = np.array(g, dtype=np.float64).ravel()
        lam, code = spla.gmres(op, rhs_vec, rtol=gmres_tol, restart=40, maxiter=200)
        if code != 0:
            raise RuntimeError(f"adjoint GMRES failed to converge (code {code})")

        # 4. push the adjoint back through the properties and into the design
        _, dalpha, dk = phi_vjp(jnp.asarray(lam.reshape(shape)))
        (drho,) = mat_vjp({"alpha": dalpha, "k": dk, "rho_phys": jnp.zeros_like(rho_phys)})

        # one extra forward call for the velocity field (the C++ side reuses its
        # cached factorisation, so this is a back-substitution, not a re-solve)
        p = self.params
        flow = apply_tesseract(
            self._t["fluid"], {"alpha": alpha, "T": T_star, "Ra": p.Ra, "Pr": p.Pr}
        )

        return {
            "J": float(self.objective(T_star)),
            "grad": np.asarray(drho),
            "T": np.asarray(T_star),
            "u": np.asarray(flow["u"]),
            "v": np.asarray(flow["v"]),
            "rho_phys": np.asarray(rho_phys),
            "info": info,
        }

    def one_way_grad(self, rho_raw, _state=None):
        """Naive gradient that keeps the chain but cuts the feedback loop.

        This is the *strong* strawman, and the one a competent engineer would
        actually write. It uses the C++ solver's derivatives in full -- the path
        rho -> alpha -> (u,v) -> T is differentiated properly -- but it treats
        the temperature entering the buoyancy term as a constant. In other
        words it pretends the pipeline is feed-forward and ignores that the
        steady state is a fixed point.

        Everything here is exact except that one omission, which isolates the
        cost of not differentiating *through the coupling*.
        """
        rho_raw = jnp.asarray(rho_raw)
        if _state is None:
            mat, mat_vjp = jax.vjp(self.material, rho_raw)
            alpha, k = mat["alpha"], mat["k"]
            T_star, _ = self.solve_coupled(alpha, k)
        else:
            mat, mat_vjp, alpha, k, T_star = _state
        T_frozen = jax.lax.stop_gradient(T_star)
        p = self.params

        def J_of(a, kk):
            flow = apply_tesseract(
                self._t["fluid"], {"alpha": a, "T": T_frozen, "Ra": p.Ra, "Pr": p.Pr}
            )
            out = apply_tesseract(
                self._t["thermal"],
                {
                    "u": flow["u"],
                    "v": flow["v"],
                    "k": kk,
                    "q_chip": p.q_chip,
                    "chip_frac": p.chip_frac,
                    "bc_mode": p.bc_mode,
                    "t_hot": p.t_hot,
                },
            )
            return self.objective(out["T"])

        da, dk = jax.grad(J_of, argnums=(0, 1))(alpha, k)
        (drho,) = mat_vjp(
            {"alpha": da, "k": dk, "rho_phys": jnp.zeros_like(mat["rho_phys"])}
        )
        return np.asarray(drho)

    def frozen_flow_grad(self, rho_raw, _state=None):
        """The weakest naive gradient: differentiate with the flow held constant.

        This is what you are forced to compute when the fluid component cannot
        hand you derivatives at all. Compare with one_way_grad, which does use
        the C++ solver's derivatives and only cuts the feedback loop.
        """
        rho_raw = jnp.asarray(rho_raw)
        if _state is None:
            mat, mat_vjp = jax.vjp(self.material, rho_raw)
            alpha, k = mat["alpha"], mat["k"]
            T_star, _ = self.solve_coupled(alpha, k)
        else:
            mat, mat_vjp, alpha, k, T_star = _state

        p = self.params
        flow = apply_tesseract(
            self._t["fluid"], {"alpha": alpha, "T": T_star, "Ra": p.Ra, "Pr": p.Pr}
        )
        u_f = jax.lax.stop_gradient(flow["u"])
        v_f = jax.lax.stop_gradient(flow["v"])

        def J_of_k(kk):
            out = apply_tesseract(
                self._t["thermal"],
                {"u": u_f, "v": v_f, "k": kk, "q_chip": p.q_chip,
                 "chip_frac": p.chip_frac, "bc_mode": p.bc_mode, "t_hot": p.t_hot},
            )
            return self.objective(out["T"])

        dk = jax.grad(J_of_k)(k)
        (drho,) = mat_vjp(
            {"alpha": jnp.zeros_like(alpha), "k": dk, "rho_phys": jnp.zeros_like(mat["rho_phys"])}
        )
        return np.asarray(drho)
