# Coldplate

**Track: Multi-physics & coupled systems**

End-to-end differentiable topology optimisation of a natural-convection cold
plate, composed from four Tesseracts that genuinely disagree about how they
compute — a C++/Eigen solver with a hand-derived adjoint, a JAX solver with
autodiff, a PyTorch material model, and a Fortran solver differentiated by
**Enzyme at the LLVM IR level**.

Three results, in order of how much we trust them.

**1. The components are interchangeable.** Swapping the JAX thermal solver for
the Fortran/Enzyme one leaves the end-to-end gradient unchanged to
**5.3 × 10⁻¹²**, cosine 1.000000000000 — the same physics, reached through a
completely different derivative technology. That is what a component boundary
is supposed to buy you, measured rather than asserted.

**2. Not composing across that boundary costs a lot, but only sometimes.** In a
strongly coupled state the component-wise gradient carries **86% error and
inverts the sign on a third of the design variables**. In the regime this
optimiser actually runs in, the same approximation is 4–20% off and works fine.
Same code, same physics, two orders of magnitude difference in how wrong you
are — and nothing in the forward solution tells you which case you are in.

**3. There is a cheap way to tell.** The implicit function theorem says the
leading error term is `Φ_Tᵀg`, so the predictor is the directional gain
`γ = ‖Φ_Tᵀg‖/‖g‖` — **one VJP**. Across four design families and five Rayleigh
numbers, log₁₀(γ) correlates with log₁₀(error) at **0.995**, while the more
obvious candidate, the coupling loop gain ρ(Φ_T), manages 0.825 and orders some
pairs backwards.

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

**Every number in that table reproduces through either thermal backend**
(`scripts/validate_both_backends.sh`), which is worth stating because the two
share no derivative machinery:

| | JAX autodiff | Fortran + Enzyme |
| --- | --- | --- |
| J | 2.626343 | 2.626343 |
| analytic ⟨g, d⟩ | 3.5227189491 × 10⁻² | 3.5227189484 × 10⁻² |
| directional rel. error | 7.45 × 10⁻⁶ | 7.10 × 10⁻⁶ |
| one-way naive | 0.856, cos 0.5335, 33% | 0.856, cos 0.5335, 33% |
| frozen-flow naive | 0.831, cos 0.5604, 27% | 0.831, cos 0.5604, 27% |

Agreement to nine or ten significant figures, and the compiler-differentiated
path is 3.7× faster here (2.9 s vs 10.7 s) — the sparse operator costs nine
Enzyme JVPs, against a JAX trace per parameter derivative.

The naive gradients carry **~85% relative error and point the wrong way on a
third of all design variables** — despite the one-way version using the C++
solver's adjoint in full and getting everything right except the feedback loop.
Cutting that loop is not a small approximation: it is most of the gradient.

### What we do *not* claim

We ran the full topology optimisation twice — once driven by the composed
gradient, once by the naive one, same seed and schedule — and **both succeeded**:

| driven by | final J | reduction |
| --- | --- | --- |
| composed gradient (exact) | 1.2588 | 84.6% |
| one-way gradient (naive) | 1.2576 | 84.6% |

The naive run ended a hair *lower*. We are not going to dress that up. An
optimiser is not evidence that a gradient is right. (This replicates: at 48×48
the same pair came out 1.2677 vs 1.2374.)

The reason is measurable. Tracking the naive gradient's error at the designs
the optimiser actually visits (`--diagnose`, 96×96):

| iteration | J | loop gain | naive error | cosine | wrong sign |
| --- | --- | --- | --- | --- | --- |
| 1 | 8.18 | 0.759 | 6.3% | 0.9983 | 0% |
| 12 | 2.31 | 0.428 | 20.1% | 0.9867 | 2% |
| 60 | 1.19 | 0.640 | 7.6% | 0.9973 | 1% |
| 120 | 1.26 | 0.587 | 4.1% | 0.9996 | 1% |

Along this trajectory the naive gradient is only **4–20% off, with a cosine
above 0.98 and almost no sign errors**. It is a perfectly serviceable search
direction here, and the optimisation result above is exactly what you would
expect from that. No mystery.

The interesting part is that this is *not* what the same comparison gives at
the strongly coupled operating point in the section above, where it is 86%
wrong with a third of the signs inverted. Two things differ: the Rayleigh
number, and the fact that filtered, projected designs are smooth while the
random design used for the gradient study is not.

> **A correction, kept in the open.** An earlier version of this table reported
> 50–150% error and up to 79% sign flips. That was an artifact: the diagnostic
> compared the optimiser's *volume-projected* gradient against a *raw* naive
> one, so it was measuring the mean-removal, not the coupling. The numbers
> above come from comparing raw against raw. The component-level,
> substitutability and Rayleigh-sweep results were never affected — none of
> them pass through that code path.

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

The rows are sorted by loop gain, not by Ra, and within this sweep that
ordering is perfect: Ra = 3×10⁵ has a *lower* loop gain than Ra = 3×10⁴ and
correspondingly less error, so the damage is not monotonic in the physical
parameter but is monotonic in ρ(Φ_T), across 5.5 orders of magnitude.

**That ordering does not survive contact with other designs.** Iteration 1 of
the optimisation has ρ(Φ_T) = 0.759 and only 6% gradient error, while the
sweep's Ra = 3×10⁵ row has essentially the same loop gain, 0.769, and 60%
error. Worse, ρ can order two states *backwards* — see the next section. So
the loop gain is a control parameter along a one-parameter family and not a
sufficient statistic. Fortunately there is a statistic that is.

What does survive is the practical warning: **there is no sign of any of this
in the forward solution.** J moves by 13% between Ra = 10² and 3×10⁴ while the
gradient error goes from 3 × 10⁻⁶ to 0.86. A pipeline can look completely
healthy and be handing you a gradient that is 86% wrong.

### What actually predicts it: one VJP

The implicit function theorem says what the error should be. The exact adjoint
solves `(I − Φ_T)ᵀ λ = g`, so

```
lambda = g + Phi_T^T g + (Phi_T^T)^2 g + ...
```

and cutting the feedback loop keeps only the first term. The leading error is
therefore **`Φ_Tᵀ g`** — which depends on the *direction* `g`, the objective's
own sensitivity to the coupled state. ρ(Φ_T) is a worst case over all
directions and cannot see this: a large gain along directions the objective
never excites costs nothing.

That suggests the **directional gain**

```
gamma  =  || Phi_T^T g ||  /  || g ||
```

which costs exactly one VJP — far less than the gradient it is judging.
Measured across four design families × five Rayleigh numbers
(`predict_error.py`, figure 8):

| predictor | correlation with log₁₀(naive error) |
| --- | --- |
| ρ(Φ_T) | +0.825 |
| **log₁₀(γ)** | **+0.995** |

And the case that settles it — two states at the *same* Rayleigh number:

| design | ρ(Φ_T) | γ | naive error |
| --- | --- | --- | --- |
| rough, Ra = 10⁴ | **0.682** | 0.111 | **0.400** |
| smooth, Ra = 10⁴ | **0.377** | 0.253 | **0.792** |

The smooth design has *half* the spectral radius and *twice* the error. ρ
orders these backwards; γ orders them correctly, and does so across the whole
dataset. Empirically `error ≈ γ` for γ ≲ 0.05, growing superlinearly beyond as
the higher Neumann terms start to matter.

So there is a usable answer to "can I get away with differentiating my
components separately?": **compute γ with a single VJP.** Below ~0.01 the
component-wise gradient is good to about a percent; above ~0.1 it is not worth
having. This is cheap enough to run as an assertion inside a pipeline, and it
is a statement about the composition rather than about this particular problem.
It ships as a standalone module, [`coupling_check.py`](coupling_check.py),
which needs only a JAX-traceable loop and does not import anything from here.

### The decisive test: hold the physics fixed, change the objective

The evidence above varies designs and Rayleigh numbers, so ρ and γ move
together and a sceptic could say the comparison is confounded. This removes the
confound entirely.

At one design and one Rayleigh number there is a single coupled state, a single
Φ, and therefore **a single ρ(Φ_T)**. Changing only *what is being measured*
changes `g = dJ/dT`, and so changes γ — but ρ cannot move. So if the error
varies, ρ is structurally incapable of explaining it (`objective_sweep.py`):

| objective | γ | naive rel. error |
| --- | --- | --- |
| outlet mean | 0.0027 | 0.0117 |
| top-half mean | 0.0076 | 0.0026 |
| chip peak | 0.0261 | 0.0385 |
| chip mean | 0.0300 | 0.0399 |
| domain mean | 0.0967 | 0.0211 |
| left-column mean | 0.3879 | 0.3535 |

**ρ(Φ_T) = 0.5481 on every one of those rows, while the error varies by a
factor of 136.** That settles it: the spectral radius is not a sufficient
statistic for whether component-wise differentiation is safe, and no amount of
additional data can rescue it. A constant cannot explain a 136-fold spread.

γ does move with the error, and it gets the dangerous case right — 0.388
against a measured 0.354. But it is a **useful approximation, not a formula**:
across objectives it correlates at 0.80 (versus 0.995 across designs and
Rayleigh numbers), and it over-predicts on the domain-mean row and
under-predicts on the outlet row, each by about 4×. That is expected — γ is the
leading Neumann term of the error in λ, whereas the quantity you finally care
about also depends on how `Φ_θᵀ` maps that error into design space. Treat it as
an order-of-magnitude guide with a real theoretical basis, not as a prediction.

### What the error looks like in space

Holding *one* design fixed and raising only the Rayleigh number
(`gradient_map_sweep.py`, figure 7):

| Ra | loop gain | naive rel. error | cosine | wrong sign |
| --- | --- | --- | --- | --- |
| 10³ | 0.056 | 0.0001 | 1.0000 | 0% |
| 10⁴ | 0.549 | 0.062 | 0.9983 | 0% |
| 3×10⁴ | 1.357 | 0.811 | 0.684 | 22% |

The picture is more informative than the numbers. At Ra = 3×10⁴ the exact
gradient develops a coherent region of *positive* sensitivity in the lower left
— adding material there makes the chip hotter, because it obstructs the
convection cell that was carrying heat away. The naive gradient has no such
region anywhere; it stays negative across the whole domain. The disagreement is
not scattered noise, it is one contiguous blob, and it is exactly the part of
the sensitivity that exists only because the flow responds to the design.

That is the clearest statement of what composing across the boundary buys: not
a small accuracy improvement, but an entire term that is otherwise absent.

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

Four components, four languages, four differentiation strategies:

| Tesseract | Language | How derivatives are obtained |
| --- | --- | --- |
| `stokes_brinkman` | C++ / Eigen | **Hand-derived discrete adjoint.** No AD tool. The system is linear in `w = (u,v,p)`, so the JVP and VJP are extra solves against the *same* sparse LU: `lam = A⁻ᵀ wbar`, then an analytic scatter against `dA/dalpha` and `db/dT`. |
| `thermal_advdiff` | JAX | **JAX autodiff.** The 5-point operator (~0.8% dense) is solved with a sparse LU, but every parameter derivative — through the Péclet-weighted face values and the face-averaged conductivity — comes from `jax.jvp` / `jax.vjp` of the residual. |
| `thermal_fortran` | Fortran | **Enzyme, compiler AD.** Same equation as above, written independently in Fortran. flang emits LLVM IR, an Enzyme pass differentiates it, and the result is a `.so` with JVP/VJP entry points. Nothing is hand-derived and no AD library is linked. |
| `material_map` | PyTorch | **torch.autograd.** Cone filter → Heaviside projection → SIMP/RAMP property maps. |

Nothing about these is naturally compatible. They disagree on language, on
memory layout, and on how a derivative is even produced. Tesseract is what
makes them a single differentiable function.

### Interchangeable, not merely composable

`thermal_advdiff` and `thermal_fortran` implement the same equation behind the
same schema. If the contract means anything, they must be swappable — and they
are (`compare_thermal_backends.py`, at a converged steady state):

| level | JAX vs Fortran/Enzyme |
| --- | --- |
| component `T` | 7.1 × 10⁻¹⁶ |
| component JVP | 1.4 × 10⁻¹⁵ |
| component VJP | 4.3 × 10⁻¹⁵ |
| converged coupled state `T*` | 4.8 × 10⁻¹² |
| **end-to-end `dJ/drho`** | **5.3 × 10⁻¹²**, cosine 1.000000000000 |

The gradient that comes out of the whole composition — through the C++ fluid
solver and the PyTorch material map — does not care whether the thermal block
was differentiated by a Python tracer or by a compiler pass over Fortran.

Two details from building it, since both cost real time:

* flang applies Fortran name mangling, so the subroutine needs
  `bind(C, name="...")`. Without it the linked module contains a *declaration*
  with no body and Enzyme reports "failed to find fn to differentiate".
* LFortran lowers `tanh` into its own runtime as `_lfortran_dtanh`, which
  Enzyme treats as opaque and refuses to differentiate. Binding straight to
  libm's `tanh` via `ISO_C_BINDING` fixes it, because Enzyme carries a rule for
  that. You can see the pass worked in the linked symbols: the library imports
  `cosh`, which appears nowhere in the source — it is the generated derivative
  of `tanh`.

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

### The honest objection: why not `jax.custom_vjp`?

Any two-container differentiable pipeline invites the same challenge — attach a
hand-written backward pass to a function JAX cannot differentiate, keep
everything in one process, delete the containers. **That objection is correct
whenever the components are Python.** Three things make it not apply here.

*The components are not Python.* Collapsing this means embedding an Eigen C++
solver and a flang/Enzyme-compiled Fortran library into the JAX process and
keeping their build toolchains alive in the same environment. The Enzyme
component needs LLVM 19 and flang at build time; the JAX component needs a
completely different Python stack. They are containerised because that is the
honest way to ship them, not to manufacture a boundary.

*The boundary is crossed thousands of times per solve, in both directions.*
`custom_vjp` gives you one backward pass through a component. What this needs is
a Krylov iteration whose every matvec re-enters the other component — forward
for the Newton solve, backward for the adjoint. That is not a wrapped function
call, it is a conversation.

*Interchangeability is the payoff.* Because the contract is a schema rather than
a Python signature, the JAX thermal block and the Fortran/Enzyme one are
substitutable with no change to the caller and no change to the gradient
(5.3 × 10⁻¹²). A `custom_vjp` wrapper is written against one implementation.

Two claims are enforced by the build rather than asserted here: neither the C++
nor the Fortran image can import `jax`, `torch`, `tensorflow`, `autograd` or
`casadi` — the build fails if they can — and the Fortran library must contain
`cosh` in its linked symbols, a function that appears in no source file. It is
Enzyme's generated derivative of `tanh`. If the pass had silently no-opped, that
symbol would be missing and the build would stop.

### Chain versus loop

Worth being precise about what kind of composition this is, because the two are
often conflated. A *chain* — component A feeds component B feeds an objective —
is differentiable by one sweep of the chain rule, and a single `jax.grad` over
wrapped components suffices. Nothing needs to be solved.

This is a *loop*. The fluid solver's output is the thermal solver's input and
the thermal solver's output is the fluid solver's input, so there is no ordering
in which one sweep suffices. The steady state must be solved for, and its
sensitivity requires solving a second, transposed system — the adjoint of
equation (1) — whose operator is available only as an action, only by calling
both components. That is the sense in which Tesseract is load-bearing here
rather than convenient.

### Why Newton, not Picard

A practical consequence worth noting: Newton–Krylov matters here. Picard cannot
converge at all once ρ(Φ_T) exceeds 1, which the gradient-study operating point
does (1.19), and Anderson acceleration did not rescue it. Newton only needs the
linearisation to be invertible, not contractive — so the JVP endpoints are not
a nicety, they are what makes the forward problem solvable at all in this
regime.

---

## The design it produces

120 design iterations at 96×96, minimising mean chip temperature subject to a
35% solid-material budget, with filtering and Heaviside continuation:

**J: 8.18 → 1.26, an 84.6% reduction in chip temperature.**

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

### The coupled physics reproduces the classical critical Rayleigh number

A fluid layer heated from below stays motionless until buoyancy overcomes
viscous and thermal diffusion. For rigid walls the onset is a precisely known
number, **Ra_c = 1707.762**. That is the right benchmark for *this* solver:
Stokes flow is the infinite-Prandtl limit, which is the regime the classical
result is derived in. (A Navier–Stokes benchmark such as de Vahl Davis would
not be — it runs at Pr = 0.71, where the inertia term we omit is not small, so
disagreement there would prove nothing.)

Onset is also exactly where our own machinery puts it. At the conduction state
the coupling loop *is* the linear stability operator, so convection begins
precisely when the loop gain reaches one — `ρ(Φ_T) = 1 ⟺ Ra = Ra_c`. Measured
by bisecting on Ra (`benchmark_critical_rayleigh.py`):

| aspect ratio | Ra_c measured | excess over 1707.762 |
| --- | --- | --- |
| 1 | 2519.07 | +47.5% |
| 2 | 1970.84 | +15.4% |
| 4 | 1779.02 | +4.2% |
| **8** | **1707.97** | **+0.012%** |

The classical value is for an unbounded layer; our no-slip side walls stabilise
a confined box, so Ra_c must sit *above* it and fall towards it as the box
widens. It does, and at aspect ratio 8 it agrees to **four significant
figures** — a relative error of 1.2 × 10⁻⁴.

One number checks the C++ Stokes solver, the thermal solver, the buoyancy
coupling between them, and the loop-gain machinery, all at once.

### The discretisation converges at second order

Verification rather than validation — does the code solve its own equations at
the rate the scheme implies? Smooth analytic material properties, the density
filter bypassed (it is grid-dependent regularisation, not physics), and
Richardson extrapolation on two independent grid trios
(`grid_convergence.py`):

| N | J | | |
| --- | --- | --- | --- |
| 16 | 3.750196 | **observed order** | **finest-grid error** |
| 24 | 3.762598 | | |
| 32 | 3.767089 | p = 1.871 (16/32/64) | 0.046% |
| 48 | 3.770452 | | |
| 64 | 3.771708 | p = 1.829 (24/48/96) | 0.023% |
| 96 | 3.772662 | | |

Monotone, and the two trios agree — second order, which is what central
differencing for diffusion plus Péclet-weighted advection should give at
moderate Péclet number.

Getting an honest number here took two corrections worth recording, because
both produced plausible-looking wrong answers first. Extrapolating through a
stalled Newton solve measures the solver rather than the scheme, so
non-converged trios are now refused outright. And the chip heat load is applied
through a binary mask (`0.3N ≤ i < 0.7N`) whose width snaps to cell edges, so
the total imposed heat wobbled ±5% non-monotonically with N — that alone made J
non-monotone and destroyed the asymptotic range. The study heats the full wall
instead. Neither issue affects the design problem; both would have silently
produced a meaningless "order of accuracy".

### Component and composition checks

Every component is checked against an independent implementation, and the
composition is checked against a monolithic reference.

| check | result |
| --- | --- |
| C++ solver & hand-derived adjoint vs JAX autodiff (forward, JVP, VJP, adjoint identity) | 1.1 × 10⁻¹² |
| JAX thermal: sparse assembly vs residual, JVP/VJP vs autodiff, adjoint identity | 3.2 × 10⁻¹⁵ |
| PyTorch material map vs finite differences, adjoint identity | 4.4 × 10⁻¹⁰ |
| Composed Tesseracts vs monolithic JAX reference (per block and full loop) | 2.5 × 10⁻¹² |
| Converged **coupled state** T\* vs the reference's, reached by a different nonlinear solver (5 Newton iterations vs 21 Picard) | 1.5 × 10⁻¹² |
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
pip install -r requirements-orchestrator.txt
```

Versions are pinned to what the published numbers were produced with. The
orchestrator and the Tesseracts deliberately pin *different* versions — the
driver runs jax 0.11 while the JAX Tesseract runs 0.10.2 inside its image,
which is the isolation working as intended rather than an oversight. Nothing
depends on the exact versions; relaxing a pin to `>=` is safe if one becomes
unavailable.

As built: LLVM/flang 19.1.7 with the Enzyme nightly plugin, Eigen 3.4,
tesseract-core 1.11.0, tesseract-jax 0.4.1. Regenerate this list with
`scripts/capture_versions.sh`.

Build the Tesseracts. The Fortran one needs its compiler toolchain image first
(flang + LLVM 19 + the Enzyme plugin); it is split out so its ~200 MB of
downloads are paid once rather than on every rebuild:

```bash
scripts/build_toolchain.sh          # or: docker build -t coldplate-enzyme-toolchain:1.0 \
                                    #       tesseracts/thermal_fortran/toolchain
```

```bash
tesseract build tesseracts/stokes_brinkman
tesseract build tesseracts/thermal_advdiff
tesseract build tesseracts/thermal_fortran
tesseract build tesseracts/material_map
```

Show that the JAX and Fortran/Enzyme thermal blocks are interchangeable:

```bash
cd orchestrator && python compare_thermal_backends.py 16
```

Reproduce the whole gradient validation through *both* backends:

```bash
scripts/validate_both_backends.sh 16
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

Test what predicts the error of a component-wise gradient (this is the result
in figure 8, and it costs one VJP per configuration):

```bash
cd orchestrator && python predict_error.py --N 20
```

Render the spatial maps of where the two gradients disagree:

```bash
cd orchestrator && python gradient_map_sweep.py --N 24
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
  thermal_fortran/    Fortran advection-diffusion, Enzyme compiler AD
    toolchain/        base image: flang + LLVM 19 + Enzyme plugin
  material_map/       PyTorch filter + projection + property maps
orchestrator/
  pipeline.py             composes the three; Newton-Krylov forward, GMRES adjoint
  validate_pipeline.py    headline gradient validation
  sweep_coupling.py       loop gain vs naive-gradient failure
  optimize.py             topology optimisation driver (--mode, --diagnose)
  probe_startpoint.py     which Ra keeps the optimiser's designs well posed
  compare_to_reference.py differential test against the monolithic reference
  predict_error.py        what predicts component-wise gradient error (one VJP)
  gradient_map_sweep.py   spatial maps of gradient disagreement vs coupling
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
| `fig6_trajectory_error.png` | naive-gradient error at the designs the optimiser visits |
| `fig7_regime_maps.png` | one design, rising coupling: where the two gradients disagree |
| `fig8_predictor.png` | the directional gain γ predicts the error; the loop gain does not |

## License

Apache 2.0. See [LICENSE](LICENSE).
