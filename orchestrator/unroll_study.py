# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Where you may differentiate an unrolled loop, and where you may not.

"Two-way coupled" covers three different things, and the README separates them:
a *chain*, an *unrolled loop*, and a *solved loop*. Only the third needs the
machinery here. That distinction has so far been argued rather than measured,
and it is the one that decides whether this project's Newton-Krylov forward and
GMRES adjoint are load-bearing or ornamental. So measure it.

The unrolled-loop gradient is the honest competitor. You iterate
`T <- Phi(T, theta)` a fixed number of times from a fixed start and let reverse
mode differentiate the whole unrolled computation. It needs no adjoint solve
and no implicit function theorem, it is exact for the iterate it actually
computes, and where the iteration contracts it lands on the fixed-point
gradient as the sweeps grow. It is a perfectly good engineering choice, and
under a contraction it is the choice we would make too.

What decides it is the loop gain rho(Phi_T):

  * **rho < 1, contracting.** Picard converges geometrically, the unrolled
    gradient converges with it, and a short unroll is exact enough.
  * **rho > 1, repelling.** There is no converged iterate to unroll toward.
    Picard diverges, and so does anything differentiated through it. The steady
    state still exists and is still differentiable -- Newton reaches it without
    requiring Phi to contract -- but its sensitivity is then a *solve*, not a
    sweep.

This script measures both regimes on the same physics, the same components and
the same containers, changing only the Rayleigh number. Everything is compared
against a central difference of the true coupled solve, which is the only
referee that does not share a method with any of the candidates.

Usage:
    python orchestrator/unroll_study.py            # audited grid, records
    python orchestrator/unroll_study.py --N 12     # smoke, records nothing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from pipeline import ColdPlate, Params

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]

#: The grid `validate_pipeline.py` audits, so the repelling case here is the
#: same state the headline gradient comparison uses.
AUDITED_N = 20

#: Sweep counts to unroll. Kept modest because every sweep is another pair of
#: container round trips in both directions.
UNROLL_STEPS = (1, 2, 4, 8, 16)

#: Two operating points on identical physics. The first is where this
#: repository's gradient study lives; the second is where the optimiser
#: actually runs, and where the README already concedes the cheap gradient is
#: serviceable.
REGIMES = (
    {"name": "repelling", "Ra": 3.0e4},
    {"name": "contracting", "Ra": 1.0e3},
)


def _loop_gain(cp: ColdPlate, T_star, alpha, k, probes: int = 12) -> float:
    """rho(Phi_T) by power iteration on the linearised loop map.

    Each matvec is a JVP through the thermal block and the C++ fluid solver, so
    this crosses the container boundary `probes` times.
    """
    _, jvp_lin = jax.linearize(lambda T: cp.phi(T, alpha, k), T_star)
    v = jnp.asarray(np.random.default_rng(3).normal(size=np.asarray(T_star).shape))
    v = v / jnp.linalg.norm(v)
    gain = 0.0
    for _ in range(probes):
        w = jvp_lin(v)
        gain = float(jnp.linalg.norm(w))
        if gain == 0.0:
            return 0.0
        v = w / gain
    return gain


def _picard(cp: ColdPlate, T0, alpha, k, sweeps: int) -> dict:
    """Residual ||Phi(T)-T|| per sweep, and whether the iteration converges.

    Classified on whether the residual actually reaches zero, not on
    first-versus-last. A repelling iteration still falls steeply out of a cold
    start before it stalls, so comparing the ends calls a stalled sequence
    "contracting" -- which it is not, and which would quietly turn this study
    into an advertisement for its own conclusion.
    """
    T = T0
    history = []
    for _ in range(sweeps):
        T_next = cp.phi(T, alpha, k)
        residual = float(jnp.linalg.norm(T_next - T))
        history.append(residual)
        if not np.isfinite(residual) or residual > 1e12:
            break
        T = T_next
    tolerance = 1e-6
    # Ratios only mean something while the residual is above the floor. Once a
    # converged sequence reaches ~1e-12 the successive ratios are noise
    # bouncing either side of 1, and reading those as a failure to contract
    # calls an obviously converged iteration divergent -- which it did, before
    # this guard.
    ratios = [
        history[i + 1] / history[i]
        for i in range(len(history) - 1)
        if history[i] > tolerance and np.isfinite(history[i + 1])
    ]
    converged = bool(history) and np.isfinite(history[-1]) and history[-1] < tolerance
    return {
        "residuals": history,
        "contraction_ratios": ratios,
        "converged": converged,
        "tolerance": tolerance,
        "criterion": (
            "the plain iteration converges when its residual reaches 1e-6; the "
            "contraction ratios are reported for the sweeps above that floor, "
            "where they still carry information"
        ),
    }


def study(cp: ColdPlate, rho, T0, direction) -> dict:
    """Unrolled-loop gradients against the truth, at one operating point."""
    # The referee: a central difference of the fully solved coupled problem.
    def J_solved(r):
        mat = cp.material(r)
        T, info = cp.solve_coupled(mat["alpha"], mat["k"])
        if not info["ok"]:
            raise SystemExit("the coupled solve did not converge; no referee available")
        return float(cp.objective(T))

    eps = 3e-4
    truth = (J_solved(rho + eps * direction) - J_solved(rho - eps * direction)) / (2 * eps)

    # The implicit adjoint, which is what this repository builds.
    res = cp.value_and_grad(rho)
    implicit = float(np.sum(np.asarray(res["grad"]) * direction))

    # The unrolled competitor, at several sweep counts.
    def loss_unrolled(r, steps):
        mat = cp.material(r)
        T = jnp.asarray(T0)
        for _ in range(steps):
            T = cp.phi(T, mat["alpha"], mat["k"])
        return cp.objective(T)

    rows = []
    for steps in UNROLL_STEPS:
        started = time.time()
        try:
            grad = jax.grad(lambda r: loss_unrolled(r, steps))(jnp.asarray(rho))
            value = float(np.sum(np.asarray(grad) * direction))
            finite = bool(np.isfinite(value))
        except Exception as exc:  # noqa: BLE001 - a diverged unroll is a result
            value, finite = float("nan"), False
            print(f"    unroll {steps:>3}: raised {type(exc).__name__}", flush=True)
        error = (abs(value - truth) / max(abs(truth), 1e-30)) if finite else float("inf")
        rows.append({
            "sweeps": int(steps),
            "directional_derivative": value if finite else None,
            "relative_error": error if np.isfinite(error) else None,
            "finite": finite,
            "seconds": round(time.time() - started, 1),
        })
        shown = f"{error:.3e}" if np.isfinite(error) else "not finite"
        print(f"    unroll {steps:>3} sweeps: rel err {shown}", flush=True)

    return {
        "finite_difference_truth": truth,
        "eps": eps,
        "implicit_adjoint": implicit,
        "implicit_relative_error": abs(implicit - truth) / max(abs(truth), 1e-30),
        "unrolled": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure when an unrolled loop is legitimate.")
    ap.add_argument("--N", type=int, default=AUDITED_N, help="grid size")
    args = ap.parse_args(argv)
    N = args.N

    rng = np.random.default_rng(0)
    rho = rng.uniform(0.25, 0.75, size=(N, N))
    direction = rng.normal(size=(N, N))
    direction /= np.linalg.norm(direction)

    record: dict = {
        "schema_version": 1,
        "N": int(N),
        "unroll_steps": list(UNROLL_STEPS),
        "design_draw": "validate_pipeline.py's: default_rng(0).uniform(0.25, 0.75)",
        "referee": "central difference of the fully solved coupled problem",
        "regimes": {},
    }

    for regime in REGIMES:
        name, Ra = regime["name"], regime["Ra"]
        print(f"\n=== {name}: Ra = {Ra:.0e} ===", flush=True)
        params = Params(Nx=N, Ny=N, Ra=Ra)
        with ColdPlate(params=params) as cp:
            mat = cp.material(rho)
            alpha, k = mat["alpha"], mat["k"]

            T_star, info = cp.solve_coupled(alpha, k)
            if not info["ok"]:
                raise SystemExit(f"{name}: the coupled solve did not converge")
            gain = _loop_gain(cp, T_star, alpha, k)
            print(f"  loop gain rho(Phi_T) = {gain:.4f}", flush=True)

            # Picard from the same cold start the unroll uses.
            T0 = jnp.zeros_like(jnp.asarray(T_star))
            picard = _picard(cp, T0, alpha, k, sweeps=max(UNROLL_STEPS))
            residuals = picard["residuals"]
            print(f"  Picard residual: {residuals[0]:.3e} -> {residuals[-1]:.3e}"
                  f"  ({'converges' if picard['converged'] else 'does not converge'})",
                  flush=True)
            if picard["contraction_ratios"]:
                shown = ", ".join(f"{r:.2f}" for r in picard["contraction_ratios"][-4:])
                print(f"  last contraction ratios: {shown}", flush=True)

            outcome = study(cp, rho, T0, direction)

        outcome.update({
            "Ra": float(Ra),
            "loop_gain": gain,
            "picard": picard,
        })
        record["regimes"][name] = outcome

    # The claim this study exists to test, stated as a pass condition.
    repelling = record["regimes"]["repelling"]
    contracting = record["regimes"]["contracting"]
    best_unroll_repelling = min(
        (row["relative_error"] for row in repelling["unrolled"]
         if row["relative_error"] is not None),
        default=None,
    )
    best_unroll_contracting = min(
        (row["relative_error"] for row in contracting["unrolled"]
         if row["relative_error"] is not None),
        default=None,
    )
    record["summary"] = {
        "repelling_loop_gain": repelling["loop_gain"],
        "contracting_loop_gain": contracting["loop_gain"],
        "implicit_error_repelling": repelling["implicit_relative_error"],
        "implicit_error_contracting": contracting["implicit_relative_error"],
        "best_unroll_error_repelling": best_unroll_repelling,
        "best_unroll_error_contracting": best_unroll_contracting,
    }

    print("\n=== summary ===")
    print(f"  repelling   rho={repelling['loop_gain']:.3f}  "
          f"implicit {repelling['implicit_relative_error']:.2e}  "
          f"best unroll {best_unroll_repelling if best_unroll_repelling is None else format(best_unroll_repelling, '.2e')}")
    print(f"  contracting rho={contracting['loop_gain']:.3f}  "
          f"implicit {contracting['implicit_relative_error']:.2e}  "
          f"best unroll {best_unroll_contracting if best_unroll_contracting is None else format(best_unroll_contracting, '.2e')}")

    target = ROOT / "orchestrator" / "results" / "unroll_study.json"
    if N == AUDITED_N:
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", newline="\n")
        print(f"\nwrote {target}")
    else:
        print(f"\ngrid {N} is not the audited {AUDITED_N}; {target.name} left untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
