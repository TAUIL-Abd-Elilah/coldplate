# Demo video script (target 4:40)

Storyboard for the ≤5 minute submission video. Every figure referenced is
already rendered in `orchestrator/results/`.

The ordering is deliberate: lead with the thing that is unambiguously true and
easy to verify (components are interchangeable), then the thing that is
surprising (how much the naive gradient costs, and when), then the thing that
is useful (a cheap test for whether you can skip composing). Save the
optimisation for the end — it is the prettiest but the least load-bearing.

---

## 0:00–0:40 — Four components that should not fit together

*On screen: `fig5_architecture.png`.*

> A cold plate cools a chip. Coolant carries heat away, solid metal conducts it
> away, and you have a limited budget of metal — so the design question is where
> to put it.
>
> The physics here is four separate pieces of software that have no business
> working together. A Stokes–Brinkman flow solver in C++ with Eigen, where the
> adjoint is derived by hand — no AD tool anywhere. An advection–diffusion
> solver in JAX, differentiated by autodiff. The *same* advection–diffusion
> equation again, this time in Fortran, differentiated by Enzyme as a compiler
> pass over LLVM IR. And a PyTorch material model in front of all of them.
>
> Four languages. Four completely different ways of producing a derivative.

## 0:40–1:30 — They're interchangeable, and that's measurable

*On screen: highlight the two thermal boxes and the purple double arrow.*

> The two thermal solvers implement the same equation behind the same schema.
> If the component contract means anything, they should be swappable — so we
> swapped them and measured what changed.
>
> The component temperature field: agrees to seven times ten to the minus
> sixteen. The JVP and the VJP: ten to the minus fifteen. The converged coupled
> state: five times ten to the minus twelve.
>
> And the end-to-end gradient — through the C++ fluid solver, through the
> PyTorch material map, through a two-way coupled fixed point — agrees to five
> times ten to the minus twelve, with a cosine of one point zero to twelve
> decimal places.
>
> The gradient does not care whether the thermal block was differentiated by a
> Python tracer or by a compiler pass over Fortran. That is what a real
> component boundary buys you.

## 1:30–2:10 — Why it isn't a pipeline

*On screen: architecture diagram, trace the loop between the two solvers.*

> These do not form a chain. Buoyancy means temperature drives the flow;
> advection means the flow drives temperature. The steady state is a fixed
> point, not a feed-forward pass.
>
> So the gradient needs implicit differentiation, and both halves of it are
> Krylov solves whose every matvec crosses the container boundary — JVPs going
> forward through Newton, VJPs coming back through the adjoint. There is no way
> to do that work per-component and staple it together afterwards.
>
> At our operating point the loop gain exceeds one, which means Picard
> iteration provably cannot converge. Newton only needs the linearisation to be
> invertible, not contractive — so the JVP endpoints aren't a nicety, they're
> what makes the forward problem solvable at all.

## 2:10–2:50 — What it costs to skip that

*On screen: `fig2_gradient_validation.png`, then `fig7_regime_maps.png`.*

> Our gradient sits on the diagonal against finite differences — agreement to
> eight parts in a million, which is the differencing noise floor, not our
> error.
>
> Now the gradient you get if you *don't* compose. And this is the charitable
> version: it uses the C++ solver's adjoint in full and gets everything right
> except the feedback loop. Eighty-six percent error. A third of the design
> variables have the wrong sign.
>
> Here's what that looks like in space. Same design, rising coupling. At high
> Rayleigh number the exact gradient grows a region of *positive* sensitivity in
> the lower left — put metal there and the chip gets hotter, because you block
> the convection cell carrying the heat out. The naive gradient has no such
> region anywhere. The disagreement isn't scattered noise, it's one contiguous
> blob: the entire coupling term, missing.

## 2:50–3:40 — When does it actually matter?

*On screen: `fig8_predictor.png`.*

> But here's the uncomfortable part. We ran the full optimisation twice, once
> with each gradient, and both worked. The naive one even finished a hair lower.
>
> So the damage is regime-dependent — and the obvious way to predict it is
> wrong. The coupling loop gain, the spectral radius of the fixed-point
> Jacobian, is the natural candidate. It scatters. These two circled points sit
> at the same Rayleigh number: the one with *half* the spectral radius has
> *twice* the error. It orders them backwards.
>
> The implicit function theorem says why. The exact adjoint is g plus Phi
> transpose g plus higher terms; cutting the loop keeps only the first. So the
> leading error is Phi-transpose-g — it depends on the *direction* g, the
> objective's own sensitivity. A spectral radius is a worst case over all
> directions and is blind to that.
>
> Use that directional gain instead and everything collapses onto the diagonal
> across four orders of magnitude. Correlation of nought point nine nine five,
> against nought point eight two five for the spectral radius. And it costs one
> VJP — far less than the gradient it's judging.
>
> So there's a usable answer to "can I get away with differentiating my
> components separately?" Compute that number. Below about a percent, you're
> fine. Above ten percent, don't bother.

## 3:40–4:20 — The artefact

*On screen: `fig1_optimisation.gif`, played through.*

> And the design itself: 120 iterations at ninety-six squared, chip temperature
> down eighty-four point six percent. The optimiser finds a branching tree that
> conducts heat up toward the sink while leaving channels open for buoyancy to
> carry the rest — which is what the natural-convection topology optimisation
> literature reports.

## 4:20–4:40 — Close

> Four containers, four languages, four differentiation strategies, one
> differentiable function — and a one-VJP test for when you need it.
>
> Everything reproducible from the README. Twenty tests run in CI without
> Docker; the container-level composition checks need it.

---

## Recording notes

- The single most convincing live command is
  `python compare_thermal_backends.py 16` — it prints the JAX-vs-Enzyme
  agreement at all three levels in about a minute.
- If showing a terminal, `nm -D --defined-only libthermal_ad.so | grep cosh`
  is a nice beat: `cosh` appears in no source file. It is Enzyme's generated
  derivative of `tanh`, visible in the linked symbols.
- Keep the "both optimisations worked" admission in. It is the most defensible
  moment in the video and it sets up the γ result, which is the actual
  contribution. Reviewers trust the rest more for it.
- Do **not** claim the naive gradient is an ascent direction. It is not, at any
  operating point we measured.
