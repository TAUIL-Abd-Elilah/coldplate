# Demo video script (rendered 00:04:50.600)

This narration is generated from committed result JSON by
`scripts/build_demo_video.py`; the timestamps below match the rendered audio.

## 00:00:00–00:00:15 — The decision, not just the derivative

*On screen: `orchestrator/results/fig10_intervention.png`.*

> What if every component derivative is correct, yet the engineering decision is wrong?
>
> Coldplate asks where a limited amount of metal should go in a buoyancy-cooled chip.
>
> At strong coupling, cutting one feedback loop changes which cells we choose and how much cooling we realize.
>

## 00:00:15–00:00:43 — A real heterogeneous fixed point

*On screen: `orchestrator/results/fig5_architecture.png`.*

> The pipeline serves a PyTorch material map, a C plus plus Eigen flow solver, and a thermal solver.
>
> Temperature drives buoyancy; velocity advects heat, so the converged state is a two-way fixed point.
>
> Newton Krylov crosses the component boundary with J V P's; the implicit adjoint crosses it again with V J P's.
>
> The thermal slot swaps JAX autodiff for independent Fortran differentiated by Enzyme at LLVM I R, and the full coupled pipeline is checked again after the swap.
>

## 00:00:44–00:01:04 — Cutting one loop corrupts the sensitivity

*On screen: `orchestrator/results/fig2_gradient_validation.png`.*

> Finite-difference component samples confirm the composed gradient to 3.7 parts per million.
>
> The strongest shortcut still differentiates every component, but freezes temperature inside buoyancy.
>
> At this strong setting it has 86 percent relative error and wrong signs in 33 percent of the design field, while the forward temperature gives no warning.
>

## 00:01:04–00:01:23 — A fresh forward solve decides

*On screen: `orchestrator/results/fig10_intervention.png`.*

> A norm is not an outcome, so each gradient proposes the same count and amplitude of positive and negative raw-design changes and is judged by a fresh coupled solve.
>
> The composed choice wins all 3 tested action sizes.
>
> At the largest step it delivers 58 percent more realized cooling under the same zero-sum raw-design rule.
>

## 00:01:23–00:02:03 — Frozen eight-step showdown: incomplete

*On screen: `orchestrator/results/fig12_showdown.png`.*

> We froze the repeated procedure before storing these trajectories, but the same operating point already had favorable one-step evidence, so this is follow-up rather than an untouched independent confirmation.
>
> The protocol gives every branch the same start, projected-volume target, eight update opportunities, and true candidate-solve budget.
>
> The frozen eight-step endpoint has no winner because the composed step-six candidate did not converge. Over the shared first 5 accepted decisions, the descriptive reductions are 11.83, 5.16, and 4.71 percent; that common prefix was examined after the failure and is not the frozen endpoint.
>
> The failure is retained without parameter tuning or selective rerun.
>

## 00:02:04–00:02:31 — 48 attempts, with overlap disclosed

*On screen: `orchestrator/results/fig13_robustness_matrix.png`.*

> This retrospective frozen extension retains every failure; 13 attempts overlap the earlier pilot and 35 cells had no stored result when the design was frozen.
>
> Among 39 comparable cases, the exact action wins 35, the shortcut wins 1, and 3 are ties.
>
> The post-freeze descriptive 95 percent seed-cluster bootstrap interval has a lower endpoint of 81.1 percent; the other 9 attempts remain visible as noncomparable, not deleted.
>

## 00:02:31–00:03:20 — Physics outside the original design point

*On screen: `orchestrator/results/fig14_physics_validation.png`.*

> The de Vahl Davis reference activates full nonlinear Navier Stokes inertia, hot and cold side walls, and insulated horizontal walls.
>
> At Rayleigh 1000 and 10000, all 6 Nusselt and centerline-velocity metrics are within 1.2 percent of the published reference.
>
> A separate 5 by 5 by 2 millimeter sealed-water example maps every nondimensional group back to S I units and preserves exactly 1 watt on the discretized chip.
>
> Only 3 of 6 planned layout and mesh solves converged; the N equals 32 finned solve stalled, so its apparent reduction is withheld rather than promoted as evidence.
>
> Even the converged baseline predicts a temperature above water's liquid range, outside the constant-property model used for this scaling exercise.
>
> The retained failure is a boundary on the dimensional illustration, not a performance or equal-material optimization claim.
>

## 00:03:20–00:03:45 — One VJP tells us when to worry

*On screen: `orchestrator/results/fig8_predictor.png`.*

> The loop-cut adjoint's exact equation residual is Phi transpose g; normalizing it costs one V J P and retains the objective direction spectral radius discards.
>
> Across 14 converged physical configurations, its log correlation with measured error is 0.995.
>
> Across 2,377 synthetic fixed points, it is 0.989, versus 0.691 for spectral radius.
>

## 00:03:46–00:04:03 — A diagnostic with an honest boundary

*On screen: `orchestrator/results/fig11_generalization.png`.*

> The limit is explicit: correlation falls to 0.36 when the loop repels, so the reusable PyTree utility provides no universal threshold and never calls that regime safe.
>
> An upstream-ready Tesseract JAX issue and test plan are prepared, but nothing will be submitted before publication review.
>

## 00:04:04–00:04:23 — The optimized artefact

*On screen: `orchestrator/results/fig1_final.png`.*

> At the weaker-coupling topology-optimization start both gradients can descend, which is precisely why the strong-setting decision studies matter.
>
> The full composed run over 120 iterations lowers the chip objective by 84.6 percent.
>
> It forms a branching conductor toward the cold sink while preserving channels for buoyant coolant flow.
>

## 00:04:23–00:04:49 — Auditable in one judge path

*On screen: `orchestrator/results/fig5_architecture.png`.*

> Linux C I runs the tests and claim audit, while a separate job rebuilds all four component images and serves three at a time across the real derivative boundary.
>
> The August twenty-ninth release workflow records exact O C I digests, checksums the paper and video, and refuses to publish until anonymous pulls succeed.
>
> Coldplate is a two-way equilibrium whose composition changes a measured engineering decision, with the evidence and the failure modes attached.
>
