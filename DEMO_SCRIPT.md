# Demo video script (target 4:30)

Storyboard for the optional ≤5 minute submission video. Every figure referenced
is already rendered in `orchestrator/results/`.

---

## 0:00–0:35 — The problem

*On screen: `fig5_architecture.png`.*

> A cold plate cools a chip. Cool fluid carries heat away, solid metal conducts
> it away, and you have a limited budget of metal — so the design problem is
> where to put it.
>
> The physics is two solvers that don't belong together. A Stokes–Brinkman flow
> solver written in C++ with Eigen, where the derivatives are hand-derived — no
> AD tool anywhere. And an advection–diffusion solver in JAX, differentiated by
> autodiff. A PyTorch material model sits in front of both. Three languages,
> three completely different ways of producing a derivative.

## 0:35–1:15 — Why it isn't a pipeline

*On screen: architecture diagram, highlight the loop between the two solvers.*

> These don't form a chain. Buoyancy means temperature drives the flow;
> advection means the flow drives temperature. The steady state is a fixed
> point, not a feed-forward pass.
>
> So the gradient needs implicit differentiation. And both halves of it are
> Krylov solves whose every matvec crosses the boundary between the C++
> container and the JAX one — JVPs going forward, VJPs coming back. There's no
> way to do that work per-component and staple it together afterwards.

## 1:15–2:00 — It works, and it's exact

*On screen: `fig2_gradient_validation.png`.*

> Left panel: the composed gradient against central finite differences. It sits
> on the diagonal — agreement to 8 parts in a million, which is the
> finite-difference noise floor, not our error.
>
> And the composed pipeline reproduces an independently written monolithic
> reference to one part in 10¹² — including the converged coupled state, which
> the two codes reach by different nonlinear solvers.

## 2:00–3:00 — What you lose without it

*On screen: `fig2` right panel, then `fig3_coupling_strength.png`.*

> Right panel is the gradient you get if you don't compose — and this is the
> *good* version of wrong: it uses the C++ solver's adjoint in full and gets
> everything right except the feedback loop. It scatters. The circled points
> have the wrong sign.
>
> And the damage is predictable. One number governs it: the gain of a single
> trip around the coupling loop. Sort by that and the error is perfectly
> monotonic across five and a half orders of magnitude — while the *forward*
> solution barely moves. There is no warning sign in the answer, only in the
> gradient.

## 3:00–3:50 — The honest part

*On screen: `fig4_opt_comparison.png`, then `fig6_trajectory_error.png`.*

> Here's the result we didn't want. We ran the full optimisation twice, once
> with each gradient — and both worked. The naive one even ended a hair lower.
>
> But look at what the naive gradient is actually doing. It's wrong by 40 to
> 150 percent the entire run, and at the starting design it has the wrong sign
> on 74 percent of the design variables. It never becomes correct. What happens
> is that its *direction* recovers, and Adam normalises per coordinate, so it
> only ever consumes direction.
>
> So: a wrong gradient can still be a usable search direction and still be a
> useless sensitivity. If you only feed it to a normalised optimiser, this hides
> forever. If you use it as a quantity — sensitivity, uncertainty, which
> variables matter — it's off by a factor of two and inverted on most of them.

## 3:50–4:30 — The design

*On screen: `fig1_optimisation.gif`, played through.*

> And the artefact itself: 120 design iterations, chip temperature down 84
> percent. The optimiser finds a branching tree that conducts heat up toward the
> sink while leaving channels open for buoyancy to carry the rest — which is
> what the natural-convection topology optimisation literature reports.
>
> Three containers, three languages, three differentiation strategies, one
> differentiable function. Everything reproducible from the README.

---

## Recording notes

- Screen-record `fig1_optimisation.gif` playing at full size for the closing
  shot; it is 6 fps and holds on the final frame.
- If showing a terminal, `python validate_pipeline.py 16` is the single most
  convincing live command — it prints the Newton convergence, the GMRES matvec
  count, and the three-way gradient comparison in about two minutes.
- Keep the naive-vs-composed honesty beat in. It is the most defensible part of
  the submission and reviewers will trust the rest more for it.
