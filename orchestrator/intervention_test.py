# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""Turn each gradient into the same action and re-solve the true physics.

Gradient validation proves that the composed adjoint is correct, but a judge can
still ask whether correctness changes a downstream decision. This experiment
turns each gradient into the same volume-neutral intervention:

* add material to the ``k`` cells it calls most beneficial;
* remove the same amount from the ``k`` cells it calls least beneficial;
* use identical counts and amplitudes, with exactly zero net material;
* re-solve the true coupled forward problem and measure the realised change.

The exact gradient chooses the optimal intervention of this form to first
order. The one-way gradient sees every component derivative but cuts the
feedback loop. Forward re-solves, not either gradient's prediction, decide
which action actually works.

Usage:  python intervention_test.py [--N 20] [--Ra 3e4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np

from pipeline import ColdPlate, Params


def balanced_topk_direction(gradient, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a {-1,0,+1} direction with ``k`` additions and removals."""
    g = np.asarray(gradient, dtype=np.float64)
    n = g.size
    if not 1 <= k <= n // 2:
        raise ValueError(f"k must be between 1 and {n // 2}, got {k}")
    order = np.argsort(g.ravel(), kind="mergesort")
    add = order[:k]
    remove = order[-k:]
    direction = np.zeros(n, dtype=np.float64)
    direction[add] = 1.0
    direction[remove] = -1.0
    return direction.reshape(g.shape), add, remove


def main(
    N: int = 20,
    Ra: float = 3.0e4,
    seed: int = 0,
    fraction: float = 0.05,
    amplitudes: tuple[float, ...] = (0.01, 0.025, 0.05),
    out: str = "results/intervention_test.json",
) -> int:
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.25, 0.75, size=(N, N))
    k = max(1, int(round(fraction * rho.size)))
    params = Params(Nx=N, Ny=N, Ra=Ra)

    print(f"forward-validated intervention, grid {N}x{N}, Ra={Ra:.0e}, "
          f"seed={seed}, k={k}")
    print("each action adds material to k cells and removes it from k cells")
    print("with identical amplitudes and exactly zero net material.\n")

    with ColdPlate(params=params) as cp:
        # Establish that this design HAS a steady state before paying for its
        # adjoint. value_and_grad would otherwise run the full GMRES adjoint --
        # up to 200 matvecs, each a pair of container round-trips -- against a
        # state that is about to be rejected anyway. On a sweep over random
        # designs, where non-convergence is a normal outcome rather than an
        # error, that dominated the entire runtime.
        probe = cp.material(rho)
        _, probe_info = cp.solve_coupled(probe["alpha"], probe["k"])
        if not probe_info["ok"]:
            raise RuntimeError(
                f"base coupled state did not converge "
                f"(residual {probe_info['residual']:.2e})"
            )

        # Warm-started by the probe, so this re-solve is nearly free.
        base = cp.value_and_grad(rho)
        if not base["info"]["ok"]:
            raise RuntimeError("base coupled state did not converge")
        g_exact = np.asarray(base["grad"])
        g_naive = np.asarray(cp.one_way_grad(rho))

        d_exact, add_exact, remove_exact = balanced_topk_direction(g_exact, k)
        d_naive, add_naive, remove_naive = balanced_topk_direction(g_naive, k)

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from coupling_check import coupling_gamma  # noqa: E402

        mat = cp.material(rho)
        T_star = base["T"]
        g_state = jax.grad(cp.objective)(T_star)
        report = coupling_gamma(
            lambda T: cp.phi(T, mat["alpha"], mat["k"]), T_star, g_state
        )

        base_T = np.asarray(base["T"])

        def true_objective(candidate):
            """Evaluate one action without preventing the other from running.

            A failed nonlinear solve is an inconclusive measurement, not
            evidence that no equilibrium exists.  Return it as data so the
            caller can still evaluate the competing action from the identical
            warm start.
            """
            cp._T_warm = base_T
            try:
                props = cp.material(candidate)
                T, info = cp.solve_coupled(props["alpha"], props["k"])
            except Exception as exc:  # noqa: BLE001 - solver failures are data
                return None, None, f"{type(exc).__name__}: {str(exc)[:160]}"
            if not info["ok"]:
                residual = float(info["residual"])
                return None, info, (
                    "solver did not converge within its budget "
                    f"(residual {residual:.2e})"
                )
            return float(cp.objective(T)), info, None

        rows = []
        print(f"base J = {base['J']:.8f}; coupling residual gamma = {report.gamma:.4f}")
        print(f"{'amplitude':>10} {'exact delta J':>15} {'naive delta J':>15} "
              f"{'exact advantage':>17}")
        for amplitude in amplitudes:
            if amplitude <= 0:
                raise ValueError("intervention amplitudes must be positive")
            r_exact = rho + amplitude * d_exact
            r_naive = rho + amplitude * d_naive
            if np.any((r_exact < 0) | (r_exact > 1) | (r_naive < 0) | (r_naive > 1)):
                raise ValueError("amplitude moves the design outside [0, 1]")

            J_exact, info_exact, reason_exact = true_objective(r_exact)
            J_naive, info_naive, reason_naive = true_objective(r_naive)
            exact_ok = J_exact is not None
            naive_ok = J_naive is not None
            delta_exact = J_exact - base["J"] if exact_ok else None
            delta_naive = J_naive - base["J"] if naive_ok else None
            advantage = (
                delta_naive - delta_exact if exact_ok and naive_ok else None
            )
            if advantage is None:
                outcome = "inconclusive"
            elif advantage > 0:
                outcome = "exact_wins"
            elif advantage < 0:
                outcome = "shortcut_wins"
            else:
                outcome = "tie"
            rows.append(
                {
                    "amplitude": amplitude,
                    "outcome": outcome,
                    "exact_action_ok": exact_ok,
                    "naive_action_ok": naive_ok,
                    "exact_action_reason": reason_exact,
                    "naive_action_reason": reason_naive,
                    "J_exact_action": J_exact,
                    "J_naive_action": J_naive,
                    "delta_J_exact_action": delta_exact,
                    "delta_J_naive_action": delta_naive,
                    "exact_advantage": advantage,
                    "predicted_delta_exact_action": float(
                        amplitude * np.sum(g_exact * d_exact)
                    ),
                    "predicted_delta_naive_action_by_true_gradient": float(
                        amplitude * np.sum(g_exact * d_naive)
                    ),
                    "residual_exact_action": (
                        float(info_exact["residual"]) if info_exact else None
                    ),
                    "residual_naive_action": (
                        float(info_naive["residual"]) if info_naive else None
                    ),
                }
            )
            exact_text = f"{delta_exact:15.7f}" if exact_ok else f"{'FAILED':>15}"
            naive_text = f"{delta_naive:15.7f}" if naive_ok else f"{'FAILED':>15}"
            adv_text = f"{advantage:17.7f}" if advantage is not None else f"{'inconclusive':>17}"
            print(f"{amplitude:10.3f} {exact_text} {naive_text} {adv_text}")

    overlap_add = len(set(add_exact.tolist()) & set(add_naive.tolist()))
    overlap_remove = len(set(remove_exact.tolist()) & set(remove_naive.tolist()))
    exact_wins = sum(r["outcome"] == "exact_wins" for r in rows)
    comparable = [r for r in rows if r["outcome"] != "inconclusive"]
    result = {
        "N": N,
        "Ra": Ra,
        "seed": seed,
        "base_J": base["J"],
        "base_residual": base["info"]["residual"],
        "k_each_way": k,
        "fraction_each_way": k / rho.size,
        "gamma": report.gamma,
        "gamma_verdict": report.verdict,
        "naive_relative_error": float(
            np.linalg.norm(g_naive - g_exact) / np.linalg.norm(g_exact)
        ),
        "naive_cosine": float(
            np.vdot(g_naive.ravel(), g_exact.ravel())
            / (np.linalg.norm(g_naive) * np.linalg.norm(g_exact))
        ),
        "add_set_overlap": overlap_add / k,
        "remove_set_overlap": overlap_remove / k,
        "exact_wins": exact_wins,
        "n_amplitudes": len(rows),
        "n_comparable_amplitudes": len(comparable),
        "rows": rows,
    }
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print(f"\nexact-gradient action wins at {exact_wins}/{len(comparable)} "
          "comparable amplitudes")
    print(f"wrote {target}")
    ok = (
        len(comparable) == len(rows)
        and exact_wins == len(comparable)
        and all(r["delta_J_exact_action"] < 0 for r in comparable)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--Ra", type=float, default=3.0e4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.01, 0.025, 0.05])
    parser.add_argument("--out", default="results/intervention_test.json")
    args = vars(parser.parse_args())
    args["amplitudes"] = tuple(args["amplitudes"])
    raise SystemExit(main(**args))
