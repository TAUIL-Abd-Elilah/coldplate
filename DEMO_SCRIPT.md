# Demo video script (target 4:55, hard cap 5:00)

Storyboard for the ≤5 minute submission video. Every figure referenced is
rendered in `orchestrator/results/`; spoken numbers are covered by
`scripts/audit_claims.py`.

The story leads with a measurable component swap, establishes why the loop
requires composition, and then shows the decisive engineering result: two
equal-budget actions chosen by different gradients and checked with fresh
coupled forward solves.

---

## 0:00–0:35 — Components that should not fit together

*On screen: `fig5_architecture.png`.*

> A cold plate cools a chip. Coolant carries heat away, solid metal conducts it
> away, and the design question is where to spend a limited material budget.
>
> Each run serves three Tesseracts: a PyTorch material map, a C++/Eigen fluid
> solver with a hand-derived adjoint, and one thermal backend. That thermal slot
> can be either JAX autodiff or the same equation in Fortran, differentiated by
> Enzyme as a compiler pass over LLVM IR.
>
> Across the repository that is three implementation languages and four
> genuinely different derivative stacks.

## 0:35–1:10 — The backends are interchangeable

*On screen: highlight the two alternatives in the thermal slot.*

> A component contract matters only if implementations are swappable, so we
> swapped them and measured every level.
>
> The temperature field agrees to seven times ten to the minus sixteen; JVP and
> VJP to about ten to the minus fifteen; the converged coupled state to five
> times ten to the minus twelve. Most importantly, the end-to-end design
> gradient agrees to five times ten to the minus twelve, with cosine one.
>
> The composed gradient does not care whether a Python tracer or a compiler
> pass over Fortran produced the thermal derivative.

## 1:10–1:40 — Why this is a loop, not a chain

*On screen: trace the two-way fluid–thermal arrows.*

> Buoyancy makes temperature drive the flow; advection makes flow drive
> temperature. The steady state is a fixed point, not a feed-forward pass.
>
> Newton–Krylov finds that state with JVPs through both components. The implicit
> adjoint is a second Krylov solve using their VJPs in reverse. At our strong
> operating point the loop gain exceeds one, so Picard iteration cannot
> converge; the cross-component derivatives are what make the solve possible.

## 1:40–2:15 — What it costs to cut the loop

*On screen: `fig2_gradient_validation.png`, then `fig7_regime_maps.png`.*

> The composed gradient agrees with finite differences to eight parts in a
> million—the differencing noise floor.
>
> Now take the charitable shortcut: keep each component's derivative, but treat
> the temperature entering buoyancy as fixed. At strong coupling it has
> eighty-six percent relative error and a third of design variables have the
> wrong sign. Spatially, it entirely misses the coherent region where adding
> metal blocks convection and makes the chip hotter. The forward solution gives
> no warning.

## 2:15–2:45 — A one-VJP warning signal

*On screen: `fig8_predictor.png`.*

> Spectral radius is the obvious warning statistic, but it is objective-blind.
> We hold the state and its Jacobian fixed, change only the objective, and the
> shortcut error moves one hundred and thirty-six fold while spectral radius
> cannot move at all.
>
> If the loop-cut adjoint is lambda-zero equals g, its exact equation residual
> is Phi-transpose-g. Normalize that residual and it costs one VJP. Across
> fourteen converged configurations its log correlation with measured error is
> nought point nine nine five.

## 2:45–3:10 — Does that hold anywhere else?

*On screen: `fig11_generalization.png`.*

> A predictor validated on one physical system is a predictor validated on one
> physical system. So we removed the physics: two thousand three hundred and
> seventy-seven randomly generated coupled fixed points, four operator
> families, linear and nonlinear loops, everything in closed form.
>
> Gamma tracks the error at nought point nine eight nine. Spectral radius
> manages nought point six nine. Nothing gamma called safe exceeded one point
> four percent error, and everything it called unsafe genuinely exceeded five.
>
> The same study found the limit, and we would rather report it than have a
> reviewer find it. That agreement is carried by attracting fixed points. When
> the fixed point repels, gamma's verdict is still right but its magnitude is
> not — and that is exactly when you should stop screening and pay for the
> adjoint.

## 3:10–3:35 — Right signs, worthless ranking

*On screen: `fig9_attribution.png`.*

> Here is why a wrong gradient can still optimise. Ask it which design cells
> matter most — the question behind tolerancing and sensor placement. On the
> fifty most influential cells the shortcut gets every sign right, which is why
> descent works.
>
> Its ranking of those same cells correlates with the truth at minus nought
> point zero one one: chance. It misses nearly half the true top fifty, and
> promotes one truly ranked one thousand and sixteenth out of one thousand and
> twenty-four. Serviceable as a direction. Worthless as a sensitivity.

## 3:35–4:05 — The gradient changes a realised decision

*On screen: `fig10_intervention.png`; animate exact and shortcut bars together.*

> But prediction is not the finish line. We give each gradient the same action:
> add material to twenty cells, remove the same amount from twenty others, then
> discard both predictions and re-solve the true coupled physics.
>
> The gradients agree on only forty percent of the add set and ten percent of
> the remove set. At three perturbation sizes, both actions cool the chip—but the
> composed gradient wins all three fresh forward solves. At the largest step it
> reduces the objective by nought point zero eight eight, versus nought point
> zero five six: fifty-eight percent more realised cooling for exactly the same
> material budget.
>
> Hold a second strong setting fixed and repeat across three converged random
> designs: the composed choice wins three out of three again. In one, the
> shortcut gradient actually has negative cosine, and the correct choice cools
> nearly four times as much.
>
> That is the engineering decision the expensive composed gradient changes.

## 4:05–4:30 — The design artefact

*On screen: `fig1_optimisation.gif`, played through.*

> At the weakly coupled topology-optimisation start, both gradients are good
> search directions—an important limitation, not something we hide. Over one
> hundred and twenty iterations at ninety-six squared, the exact run cuts chip
> temperature by eighty-four point six percent. It finds a branching conductor
> toward the cold sink while leaving coolant channels open for buoyant flow.

## 4:30–4:55 — Close

> Three served components, a selectable JAX-or-Fortran thermal backend, four
> derivative stacks, one two-way differentiable equilibrium—and a one-VJP check
> for when the shortcut is risky.
>
> The README starts with a safe judge command. Sixty-one component tests run
> in CI without Docker, while a scheduled container job exercises the real
> boundary. Source, four-page paper, evidence, and every command are public.

---

## Recording notes

- The strongest live command is `scripts/judge_demo.sh --no-build --grid 16`.
  It performs integrity checks and the JAX-versus-Enzyme backend swap while
  cleaning up only Coldplate images.
- If showing a terminal, `nm -D --defined-only libthermal_ad.so | grep cosh`
  is a useful beat: `cosh` appears in no source file. It is Enzyme's generated
  derivative of `tanh`, visible in the linked symbols.
- Keep the admission that both long optimisations worked. It motivates the
  equal-budget intervention, which is the stronger outcome test.
- Do not call the calibrated γ bands universal thresholds, do not call the
  repeated VJPs a convergent Neumann series at ρ ≥ 1, and do not claim that all
  four implementations are served simultaneously.
- Replace “public” in the closing line if the repository has not yet been made
  public at recording time. The submission itself must use the public URL.
