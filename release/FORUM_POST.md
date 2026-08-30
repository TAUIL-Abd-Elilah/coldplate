# Discourse showcase post

Post to
<https://si-tesseract.discourse.group/c/hackathons-events/tesseract-summer-hackathon-2026/>
(the hackathon category, where `tesseract-caprock` and `ferrumizer` posted).
Discourse renders GitHub-flavoured Markdown, so this can be pasted as-is.

**Title**

```text
Coldplate: one jax.grad across a two-way physics loop, four derivative stacks and three languages
```

**Images to attach**, in the order they are referenced below. Drag each into
the composer at the marked point; do not hotlink from GitHub, because a
raw.githubusercontent URL renders as a bare link for logged-out readers.

1. `orchestrator/results/fig5_architecture.png`
2. `orchestrator/results/fig2_gradient_validation.png`
3. `orchestrator/results/fig1_optimisation.gif`
4. `orchestrator/results/fig11_generalization.png`

---

Track 02, multi-physics and coupled systems.

Most differentiable-simulation demos are chains: one component feeds the next,
one sweep of the chain rule, done. Coldplate is a loop, and the loop is the
part that gets dropped.

A cold plate cooled by natural convection. Temperature drives the flow through
buoyancy; the flow drives temperature through advection. The steady state is
therefore a fixed point rather than a pipeline, and its gradient is an
implicit-function-theorem adjoint whose every matrix-vector product has to
cross a container boundary — in both directions, because the forward solve is
Newton–Krylov (JVPs) and the adjoint is GMRES against the transpose (VJPs).

## What I composed

*[attach fig5_architecture.png]*

Four Tesseracts, three implementation languages, four different ways of
getting a derivative:

- **`material_map`** — PyTorch, `torch.autograd`. Filter, project, SIMP.
- **`stokes_brinkman`** — C++/Eigen, **hand-derived discrete adjoint**. Also
  takes a nonlinear Navier–Stokes inertia term, in which case its adjoint is a
  solve against the Jacobian at the converged state.
- **`thermal_advdiff`** — JAX, autodiff.
- **`thermal_fortran`** — the same equation written independently in Fortran
  and differentiated by **Enzyme at the LLVM IR level**, with the toolchain and
  a hash-pinned `LLVMEnzyme-19.so` vendored so the build does not depend on a
  mutable nightly URL.

The two thermal blocks are *interchangeable*, not merely composable. Swap JAX
autodiff for compiler-differentiated Fortran and the full end-to-end gradient
moves by **5.3 × 10⁻¹²**, cosine 1.000000000000. `scripts/judge_demo.sh` serves
both and swaps them in 1–3 minutes warm.

## The result

*[attach fig2_gradient_validation.png]*

The interesting quantity is not the gradient's accuracy, it is what it costs
to get it wrong.

Cutting the feedback loop — the shortcut anyone writes when one solver hands
out no derivatives — leaves a gradient that is **86% wrong with a third of the
design variables carrying the wrong sign**. The version that uses the C++
solver's adjoint *in full* and gets everything right except the loop is just as
bad. And nothing in the forward solution says so: J, the residual and the
convergence history all look healthy.

Asked to place the same fixed zero-sum design action, the coupling-complete
gradient buys **58% more realised cooling** when the true coupled solver
re-scores both choices. A retrospectively frozen 48-attempt extension keeps
every contrary outcome: 35 exact wins, **1 shortcut win**, 3 ties, 9
noncomparable.

*[attach fig1_optimisation.gif]*

## Can you get away with the shortcut? One VJP tells you

*[attach fig11_generalization.png]*

The obvious diagnostic is the loop gain ρ(Φ_T), and it is **not sufficient**,
because it is objective-blind. Holding the design and the operating point fixed
so ρ cannot move, and changing only *what is being measured*, the loop-cut
error moved by a factor of 136.

The residual of the loop-cut adjoint is exactly Φ_Tᵀg — **one VJP**, far less
than the adjoint solve it would let you skip. Its normalised norm
γ = ‖Φ_Tᵀg‖/‖g‖ predicts the damage: log-correlation 0.995 across the physics,
and **0.989 across 2,377 random coupled systems with no physics in them at
all**, against 0.691 for the spectral radius. It has a boundary I would rather
document than hide: for repelling fixed points (ρ ≥ 1) the correlation collapses
to 0.36, and a known-repelling map can never be labelled safe.

It ships as `fixed_point_adjoint.py` — arbitrary JAX PyTrees, knows nothing
about cold plates, and refuses to return a verdict until you give it thresholds
calibrated on your own application. I have opened
[tesseract-jax#247](https://github.com/pasteurlabs/tesseract-jax/issues/247)
asking whether it belongs in the library; happy to be told it belongs in user
code.

## Why Tesseract is load-bearing, not a costume

At the operating point ρ(Φ_T) ≈ **1.19 > 1**: the fixed point is *repelling*, so
there is no converging Picard iteration to unroll and no way to fake this with
a `for` loop under `jax.grad`. Newton–Krylov reaches it and the adjoint is a
second transposed solve, and every matvec of both crosses the container
boundary. The `jax.custom_vjp` objection gets an explicit section in the README
rather than being ignored.

Two claims about component isolation are **enforced by the build** rather than
asserted: `thermal_fortran`'s config fails the image build if any AD framework
is importable inside it, or if Enzyme's generated `cosh` is absent from the
compiled object.

## Two things found in the platform itself

Composing four components in three languages, with four different derivative
stacks, pushes on corners a single-framework pipeline never reaches. Both of these are open, and I am not
describing either as accepted:

- **[tesseract-core#706](https://github.com/pasteurlabs/tesseract-core/issues/706)**
  — `check-gradients` documents `--eps` as a step "as a fraction of the maximum
  absolute value of each input" and applies it as an absolute one. Our Brinkman
  drag is of order 10⁴, so the documented reading of `--eps=1e-6` asks for a
  step of 0.02 and takes 10⁻⁶, and the checker then reports **400 failures out
  of 400** against a derivative that matches an independent JAX
  reimplementation to better than 1e-9. Holding everything else fixed and
  changing only the step on that one input, 120/120 failures become 4/120. Six-line reproducer, no container
  needed. It also notes that `--show-progress` has no `--no-` form, so a CI job
  cannot turn the progress bar off.
- **[tesseract-core#713](https://github.com/pasteurlabs/tesseract-core/pull/713)**
  — a pull request fixing that one: `eps` also takes a step per input path, a
  scalar behaves exactly as before, the docstring is made true, and
  `--no-show-progress` exists. Green locally on the 422 runtime tests; the CLA
  and public PR checks are green, and maintainer review is pending.
- **[tesseract-jax#247](https://github.com/pasteurlabs/tesseract-jax/issues/247)**
  — the γ diagnostic above, proposed as a library call rather than something
  every user rediscovers.

I mention the first one because its failure mode is the worst kind: the tool
accuses your code, confidently, and nothing in the output suggests the step size
was the problem.

## And then its checker found a bug in ours

Once the step was scaled properly, I ran Tesseract's checker against all four
components at the converged coupled state. Two results, and the second is the
one I would not have got any other way.

Across 156 comparisons the worst disagreement between any derivative
endpoint and a central difference through `apply` is **8.4e-07** relative —
four derivative implementations sharing no machinery, graded by somebody else's
tool.

It also found **20 phantom sensitivities** in our JAX thermal block:
comparisons where the finite difference is *exactly* zero, because the forward
map never reads the perturbed input, while the endpoint reports a real number.
They are the wall-face velocities. The assembly `apply` uses sums fluxes over
interior faces only; the JAX residual the derivative path differentiates
includes wall terms. At a no-slip wall `u = 0`, so the two agree on every value
and differ only in the derivative with respect to an input that is always zero —
which is exactly why none of our own checks caught it.

The independently written Fortran block has none. That localises it, and it also
bounds it: swapping the affected block for the clean one moves the end-to-end
gradient by 5.3 × 10⁻¹² (a measurement that was committed long before I went
looking), so the phantom cotangents cannot be reaching the composed gradient.
The defect is real, it is ours, it changes no number in the repository, and it
is written up there rather than quietly patched on the last day.

If you are composing Tesseracts and have not run `check-gradients` against your
components at the state your adjoint actually differentiates at, it is worth an
afternoon.

## What the coupled adjoint replaces

One composed gradient costs 4.0 s
(13 JVPs and
15 VJPs across the container
boundary). A central-difference gradient over the same 2,304 design variables is
4,608 coupled solves, about
1.3 h — extrapolated from a measured
per-probe time, and labelled as such. The whole 120-iteration optimisation took
7.8 minutes; the same schedule on finite
differences extrapolates to
6.5 days.

## The parts that went against us, kept

- The long optimisation runs to the same place on the cheap gradient. Both
  reached 84.6%; the naive run ended a hair *lower*. An optimiser converging is
  not evidence that a gradient is right, and we say so under
  *What we do not claim*.
- A frozen eight-step protocol stopped when a solve failed, so it has **no
  verdict** and is retained as incomplete rather than trimmed to the prefix
  that would have won.
- The dimensional SI case failed its own audit — 3 of 6 solves converged, at
  temperatures outside its constant-property regime — so **no fin-performance
  number is claimed** anywhere.
- An earlier diagnostic reported 50–150% error and 79% sign flips. It was
  comparing a volume-projected gradient against a raw one. The correction is in
  the README, not in the git history only.

`scripts/audit_claims.py` re-derives every quoted number from the stored
measurements and additionally **refuses a list of overclaims we previously made
and retracted**, so they cannot come back by copy-paste.

## Links

- Repository, four-page paper, and every number beside the file that produced
  it: <https://github.com/TAUIL-Abd-Elilah/coldplate>
- Results page (every headline beside its source JSON, plus a γ gate you can
  move yourself): <https://tauil-abd-elilah.github.io/coldplate/docs/>
- 4:51 narrated demo, in the repository under `demo/` — with a second render
  narrated locally, so no service's terms govern the audio.

Apache-2.0. Happy to answer questions about the cross-boundary Krylov plumbing,
or about the Fortran/Enzyme toolchain, which was the part that cost the most
real hours.

Built on Tesseract, from Pasteur Labs & ISI.
