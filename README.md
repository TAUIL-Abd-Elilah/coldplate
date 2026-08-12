# Coldplate

**Track: Multi-physics & coupled systems**

End-to-end differentiable topology optimisation of a natural-convection cold
plate, composed from three Tesseracts that genuinely disagree about how they
compute — a C++/Eigen solver with a hand-derived adjoint, a JAX solver with
autodiff, and a PyTorch material model.

The headline result is not that the composition works. It is a measurement of
what you lose without it: differentiating the components separately — even
doing everything else exactly right — leaves a gradient that is **wrong by
40–150%, with the wrong sign on up to 74% of the design variables**, while
still looking healthy to an optimiser.

---

## The claim

The two physics blocks are coupled in both directions. Buoyancy makes
temperature drive the flow; advection makes the flow drive temperature. The
steady state is therefore a fixed point, not a feed-forward chain:

```
T* = Phi(T*, theta),    Phi = thermal( fluid(T) )
```

If the fluid component cannot hand you derivatives — the normal situation for a
legacy or closed-source solver — the best you can do is freeze the velocity
field and differentiate the thermal block alone. We measured what that costs.

We compare three gradients against central finite differences on the fully
converged coupled solve. Two are naive, and the second is the one a competent
engineer would actually write:

- **composed adjoint** — implicit differentiation of the fixed point (this work);
- **one-way** — uses the C++ solver's derivatives in full and differentiates the
  entire chain `rho → alpha → (u,v) → T`, but treats the temperature entering
  buoyancy as constant. Everything is exact *except* that it ignores the loop;
- **frozen-flow** — the velocity field held constant, i.e. what you are forced
  to do when the fluid component exposes no derivatives at all.

Measured through the real Tesseracts at Ra = 3×10⁴ (see `validate_pipeline.py`;
numbers reproduced by the script, not quoted from a notebook):

| gradient | rel. error | cosine vs true | design variables with the wrong sign |
| --- | --- | --- | --- |
| **composed adjoint** | **8.3 × 10⁻⁶** | 1.0000 | 0% |
| one-way (naive) | 0.856 | +0.53 | **33%** |
| frozen-flow (naive) | 0.831 | +0.56 | **27%** |

The composed figure is a directional derivative along a random unit vector,
agreeing with central differences to 8.3 × 10⁻⁶ and holding at ~10⁻⁵ across
step sizes from 10⁻³ to 10⁻⁴ — that plateau *is* the finite-difference noise
floor, since the fixed point is only converged to ~10⁻¹⁰. The adjoint is exact;
the difference scheme is the inaccurate one.

The naive gradients carry **~85% relative error and point the wrong way on a
third of all design variables** — despite the one-way version using the C++
solver's adjoint in full and getting everything right except the feedback loop.
Cutting that loop is not a small approximation: it is most of the gradient.

### What we do *not* claim

We ran the full topology optimisation twice — once driven by the composed
gradient, once by the naive one, same seed and schedule — and **both succeeded**:

| driven by | final J | reduction |
| --- | --- | --- |
| composed gradient (exact) | 1.2677 | 84.1% |
| one-way gradient (naive) | 1.2374 | 84.5% |

The naive run ended marginally *lower*. We are not going to dress that up. An
optimiser is not evidence that a gradient is right.

The resolution is measurable, and it is the most interesting thing we found.
Tracking the naive gradient's error along the trajectory (`--diagnose`):

| iteration | J | loop gain | naive error | cosine | wrong sign |
| --- | --- | --- | --- | --- | --- |
| 1 | 7.99 | 0.76 | **153%** | 0.54 | **74%** |
| 6 | 4.78 | 0.61 | 105% | 0.76 | 35% |
| 12 | 2.63 | 0.56 | 41% | 0.95 | 6% |
| 60 | 1.28 | 0.39 | 63% | 0.85 | 8% |
| 120 | 1.27 | 0.50 | 76% | 0.81 | 7% |

The naive gradient is wrong by **40–150% for the entire run** and, at the
uniform starting design, has the **wrong sign on 74% of all design variables**.
It never becomes correct. What happens is that after a few iterations its
*direction* recovers (cosine 0.81–0.96) — because as the design solidifies,
solid material blocks the flow and the coupling weakens — and Adam normalises
per coordinate, so it consumes only direction.

So the honest statement is: **a wrong gradient can still be a usable search
direction, and is still a useless sensitivity.** If the gradient is only ever
fed to a normalised optimiser, this failure can hide indefinitely. If it is
used as a quantity — sensitivity analysis, uncertainty propagation, deciding
which design variables matter, anything with a magnitude in it — it is wrong by
a factor of two and inverted on most variables.

That is a much easier mistake to make than "my optimiser diverged", and it is
exactly the kind of thing composing across the boundary buys you protection
from.

### The failure is predictable

One number governs it: ρ(Φ_T), the spectral radius of the fixed-point Jacobian
— the gain of a single trip around the coupling loop. Measured through the
Tesseracts by power iteration on JVPs (`sweep_coupling.py`):

| Ra | loop gain ρ(Φ_T) | naive rel. error | cosine | wrong sign |
| --- | --- | --- | --- | --- |
| 10² | 0.0055 | 0.0000 | 1.0000 | 0% |
| 10³ | 0.0553 | 0.0003 | 1.0000 | 0% |
| 3×10³ | 0.1649 | 0.0036 | 1.0000 | 0% |
| 10⁴ | 0.5219 | 0.1333 | 0.9933 | 0% |
| 3×10⁵ | 0.7686 | 0.6031 | 0.8530 | 4% |
| **3×10⁴** | **1.1919** | **0.8558** | **0.5335** | **33%** |

The rows are sorted by loop gain, not by Ra, and that is the point. Ra = 3×10⁵
has a *lower* loop gain than Ra = 3×10⁴ and correspondingly less error, so the
damage is **not** monotonic in the physical parameter — but it is perfectly
monotonic in ρ(Φ_T), across 5.5 orders of magnitude of error. The loop gain is
the controlling variable, which is what makes this transferable to other
coupled systems: measure ρ(Φ_T) and you know whether you can get away with
differentiating your components separately.

Weak coupling hides the problem completely — at Ra = 10² the naive gradient is
correct to six digits. That is exactly why it is easy to ship a component-wise
pipeline and never discover it is wrong. And there is no warning sign in the
forward solution: J moves by only 13% between Ra = 10² and 3×10⁴ while the
gradient error goes from 3 × 10⁻⁶ to 0.86.

### Why Newton, not Picard

Note the operating point has **ρ(Φ_T) ≈ 1.19 > 1**. The fixed point is
*repelling*, so Picard iteration cannot converge to it — that is not an
observation but a consequence of the spectral radius exceeding one. Anderson
acceleration did not rescue it either in our earlier runs. The steady state is
nonetheless perfectly well defined and perfectly differentiable; it simply
cannot be reached by iterating the map.

Newton–Krylov does not care, because it only needs `(Φ_T − I)` to be
invertible, not contractive. This is the practical reason the composition needs
JVPs as well as VJPs: the forward solve and the adjoint solve are *both* Krylov
iterations across the component boundary.

---

## Architecture

```
                  rho_raw  (design variables)
                     |
        [ material_map · PyTorch · torch.autograd ]
                     |
              k, alpha, rho_phys
                     |
      +--------------+---------------+
      |                              |
      v                              v
  [ stokes_brinkman ]            [ thermal_advdiff ]
  C++ / Eigen SparseLU           JAX / sparse LU
  hand-derived adjoint           JAX autodiff
      |                              ^
      |   u, v ---------------------→|
      |←--------------------------- T |
      +------------------------------+
            two-way coupled fixed point
```

Three components, three languages, three differentiation strategies:

| Tesseract | Language | How derivatives are obtained |
| --- | --- | --- |
| `stokes_brinkman` | C++ / Eigen | **Hand-derived discrete adjoint.** No AD tool. The system is linear in `w = (u,v,p)`, so the JVP and VJP are extra solves against the *same* sparse LU: `lam = A⁻ᵀ wbar`, then an analytic scatter against `dA/dalpha` and `db/dT`. |
| `thermal_advdiff` | JAX | **JAX autodiff.** The 5-point operator (~0.8% dense) is solved with a sparse LU, but every parameter derivative — through the Péclet-weighted face values and the face-averaged conductivity — comes from `jax.jvp` / `jax.vjp` of the residual. |
| `material_map` | PyTorch | **torch.autograd.** Cone filter → Heaviside projection → SIMP/RAMP property maps. |

Nothing about these three is naturally compatible. They disagree on language,
on memory layout, and on how a derivative is even produced. Tesseract is what
makes them a single differentiable function.

---

## Why this needs Tesseract

The gradient uses the implicit function theorem at the fixed point:

```
dJ/dtheta = (dPhi/dtheta)^T lambda,     (I - dPhi/dT)^T lambda = dJ/dT
```

Both solves are **Krylov iterations whose matvecs cross the component
boundary**:

- **Forward** — Newton–Krylov on `F(T) = Phi(T) - T`. Each GMRES matvec applies
  `(Phi_T - I)` via `jax.jvp`, running *forward* through the C++ block then the
  JAX block.
- **Adjoint** — GMRES on `(I - Phi_T)^T`. Each matvec runs a VJP *backward*
  through the JAX block then the C++ block.

There is no way to factor that work into per-component pieces and reassemble
it afterwards. The adjoint of the coupled system only exists as a conversation
between the two solvers, in both directions. That conversation is exactly what
Tesseract's `jacobian_vector_product` / `vector_jacobian_product` endpoints
make possible across a container and language boundary.

A practical consequence worth noting: Newton–Krylov matters here. Picard cannot
converge at all once ρ(Φ_T) exceeds 1, which the gradient-study operating point
does (1.19), and Anderson acceleration did not rescue it. Newton only needs the
linearisation to be invertible, not contractive — so the JVP endpoints are not
a nicety, they are what makes the forward problem solvable at all in this
regime.

---

## The design it produces

120 design iterations at 48×48, minimising mean chip temperature subject to a
35% solid-material budget, with filtering and Heaviside continuation:

**J: 7.99 → 1.27, an 84% reduction in chip temperature.**

The optimiser discovers a branching tree that conducts heat from the chip up
toward the cold sink while leaving open channels for buoyancy-driven flow to
carry the rest away — the same qualitative structure the natural-convection
topology optimisation literature reports. See `fig1_optimisation.gif`.

A caveat worth stating: the optimisation runs at Ra = 10³, not the 3×10⁴ used
for the gradient study. That is a measured constraint, not a preference
(`probe_startpoint.py`). A near-uniform intermediate density — where topology
optimisation is obliged to start — is the worst case for coupling: low Brinkman
drag means nothing damps the flow and low conductivity means a high Péclet
number, simultaneously. Such designs have no reachable steady state above
Ra ≈ 10³, because open-cavity natural convection there is genuinely unsteady
and there is nothing for a steady solver to converge to. The heterogeneous
design used for the gradient study stays solvable to Ra = 3×10⁵. Even at
Ra = 10³ the starting design has a loop gain of 0.76.

## Physics

Steady, non-dimensional, Boussinesq, Stokes flow with Brinkman penalisation of
solid regions, on a staggered MAC grid:

```
fluid     -Pr ∇²u + ∇p + Pr·alpha(rho)·u = Ra·Pr·T ê_y
          ∇·u = 0
thermal   ∇·(uT) - ∇·(k(rho) ∇T) = 0
```

A chip dissipates a fixed heat flux into part of the bottom wall; the top wall
is a cold sink; the sides are adiabatic and all walls are no-slip. The design
field `rho` distributes a limited budget of solid material — conductive but
flow-blocking — to minimise the mean chip temperature. Solid conducts heat
away; open channels let buoyancy carry it away. The optimum has to trade the
two off, which is why the coupled gradient matters.

Both blocks are *linear in their own unknown*; all the nonlinearity lives in
the composition. That is what makes the per-block adjoints exact and cheap, and
it isolates the hard part — the coupling — as the only place a gradient can go
wrong.

---

## Validation

Every component is checked against an independent implementation, and the
composition is checked against a monolithic reference.

| check | result |
| --- | --- |
| C++ solver & hand-derived adjoint vs JAX autodiff (forward, JVP, VJP, adjoint identity) | 1.1 × 10⁻¹² |
| JAX thermal: sparse assembly vs residual, JVP/VJP vs autodiff, adjoint identity | 3.2 × 10⁻¹⁵ |
| PyTorch material map vs finite differences, adjoint identity | 4.4 × 10⁻¹⁰ |
| Composed Tesseracts vs monolithic JAX reference (per block and full loop) | 2.4 × 10⁻¹² |
| Conduction-only energy balance, flux in vs flux out | 5.3 × 10⁻¹⁵ |
| Composed end-to-end gradient vs finite differences (directional derivative) | 8.3 × 10⁻⁶ |
| Sparse thermal assembly vs the JAX residual it must reproduce | 1.4 × 10⁻¹⁶ |

The reference implementation (`prototype/reference_jax.py`) is deliberately
written a different way — dense operators assembled by `jacfwd` of a linear
residual — so that agreement is evidence rather than a shared bug.

---

## Reproduce

Requires Docker and Python 3.10+.

```bash
pip install "tesseract-core[runtime]" tesseract-jax "jax[cpu]" numpy scipy matplotlib
```

Build the three Tesseracts:

```bash
tesseract build tesseracts/stokes_brinkman
tesseract build tesseracts/thermal_advdiff
tesseract build tesseracts/material_map
```

Reproduce the headline claim — the composed gradient matches finite differences
to the noise floor, the component-wise ones do not:

```bash
cd orchestrator && python validate_pipeline.py 16
```

Measure where component-wise differentiation breaks down:

```bash
cd orchestrator && python sweep_coupling.py
```

Run the optimisation and render every figure:

```bash
cd orchestrator && python optimize.py --N 48 --iters 120 && python make_figures.py --N 48
```

Drive the same optimisation with the naive gradient, for comparison:

```bash
cd orchestrator && python optimize.py --N 48 --iters 120 --mode one_way
```

Track how wrong the naive gradient is at each design along the way — this is
the measurement behind the trajectory table above:

```bash
cd orchestrator && python optimize.py --N 48 --iters 120 --diagnose 6 --outdir results_diag
```

Check which Rayleigh numbers keep the optimiser's designs well posed:

```bash
cd orchestrator && python probe_startpoint.py
```

Verify the composed pipeline against the independent monolithic reference:

```bash
cd orchestrator && python compare_to_reference.py 16
```

Component-level tests, which need no containers:

```bash
python tesseracts/stokes_brinkman/test_against_reference.py
python tesseracts/thermal_advdiff/test_thermal.py
python tesseracts/material_map/test_material.py
```

---

## Layout

```
tesseracts/
  stokes_brinkman/    C++/Eigen fluid solver, hand-derived adjoint
  thermal_advdiff/    JAX advection-diffusion, sparse LU
  material_map/       PyTorch filter + projection + property maps
orchestrator/
  pipeline.py             composes the three; Newton-Krylov forward, GMRES adjoint
  validate_pipeline.py    headline gradient validation
  sweep_coupling.py       loop gain vs naive-gradient failure
  optimize.py             topology optimisation driver (--mode, --diagnose)
  probe_startpoint.py     which Ra keeps the optimiser's designs well posed
  compare_to_reference.py differential test against the monolithic reference
  show_trajectory.py      naive-gradient error along the optimisation
  make_figures.py         figures and animation
prototype/
  reference_jax.py    independent monolithic reference implementation
```

## Figures

| file | what it shows |
| --- | --- |
| `fig1_optimisation.gif` | the design evolving: layout, temperature + streamlines, convergence |
| `fig2_gradient_validation.png` | composed vs naive against finite differences |
| `fig3_coupling_strength.png` | naive-gradient error vs coupling loop gain |
| `fig4_opt_comparison.png` | optimisation driven by each gradient |
| `fig5_architecture.png` | the three components and the adjoint between them |
| `fig6_trajectory_error.png` | the naive gradient stays wrong but stays pointed downhill |

## License

Apache 2.0. See [LICENSE](LICENSE).
