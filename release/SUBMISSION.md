# Submission pack

Everything the two outside steps need, written down so neither is composed under
time pressure. The repository must be public before either one, because both
carry links a judge will click.

## 1. The LinkedIn post

Required by the rules: it must carry the repository link and tag **Pasteur Labs
& ISI** and **Tesseract**. Post from the entrant's own account, then copy its URL
into the form.

---

Most differentiable-simulation demos are chains: one component feeds the next,
one sweep of the chain rule, done.

Coldplate is a loop. A cold plate cooled by natural convection — temperature
drives the flow through buoyancy, the flow drives temperature through advection.
The steady state is a fixed point, not a pipeline, and its gradient is an
implicit-function-theorem adjoint whose every matrix-vector product has to cross
a container boundary in both directions.

Four Tesseracts, three languages, four ways of getting a derivative:

• a C++/Eigen Stokes–Brinkman solver with a hand-derived discrete adjoint
• a JAX advection–diffusion solver
• the same equation written independently in Fortran and differentiated by
  Enzyme at the LLVM IR level
• a PyTorch material map

The two thermal blocks are interchangeable, not merely composable: swapping JAX
autodiff for compiler-differentiated Fortran moves the end-to-end gradient by
5.3 × 10⁻¹², cosine 1.000000000000.

The result that matters is not the gradient's accuracy but what it costs to get
it wrong. Cutting the feedback loop — the shortcut anyone writes when a solver
hands out no derivatives — leaves a gradient that is 86% wrong with a third of
the signs inverted, and nothing in the forward solution says so. Asked to place
the same fixed design action, the coupling-complete gradient buys 58% more
realised cooling when the true coupled solver re-scores both choices.

So: can you get away with differentiating your components separately? There is a
cheap answer. The residual of the loop-cut adjoint is exactly Φᵀg, one VJP, and
its normalised norm predicts the damage — log-correlation 0.995 across the
physics, and 0.989 across 2,377 random coupled systems with no physics in them
at all. It ships as a module that knows nothing about cold plates.

The repository also keeps what did not work: a frozen protocol that stopped when
a solve failed, a dimensional case that failed its own audit, and the long
optimisation where the cheap gradient did just as well. Those are in the README
under "What we do not claim", because a study that cannot record a negative is
not evidence.

Code, four-page paper, narrated demo and every number beside the file that
produced it:
https://github.com/TAUIL-Abd-Elilah/coldplate

Track 02 — multi-physics & coupled systems. Apache-2.0.
Built on Tesseract, from @Pasteur Labs & ISI. #Tesseract

---

## 1b. The forum showcase post

Not required by the rules, and worth doing anyway: the organisers read
[the hackathon category](https://si-tesseract.discourse.group/c/hackathons-events/tesseract-summer-hackathon-2026/)
and have replied to entries posted there. Two of the strongest public entries
posted one; this repository had no presence there at all.

The post is written and ready in [`FORUM_POST.md`](FORUM_POST.md), sized for
Discourse, with the four figures to attach named in the order they are
referenced. It needs the entrant's own forum account.

## 2. The form

<https://tally.so/r/KYNZMg> — four required fields, one submission per team.

| field | answer |
| --- | --- |
| Your full name | *(the registered entrant's legal name)* |
| Your email | *(the address used at registration)* |
| Link to your LinkedIn post | *(URL of the post above, after publishing)* |
| Link to your GitHub repository | `https://github.com/TAUIL-Abd-Elilah/coldplate` |

Submit **once**. The repository README names the track in its first three lines,
which is what the form's instruction asks for.

## 3. Before either step

Run through [`AUGUST_29_CHECKLIST.md`](AUGUST_29_CHECKLIST.md). The parts no
script can do:

- confirm registration, and the eligibility declarations the terms require:
  age/guardian consent, residence and sanctions, and no disqualifying Pasteur
  Labs employment, contract or immediate-family relationship;
- confirm the work was created during the 3–31 August period, does not infringe
  third-party IP, and that every entrant agrees to Apache-2.0;
- confirm one submission only, at most four people, and that every actual team
  member was named at registration;
- confirm the Git author identities `Coldplate`, `pixgenx` and
  `TAUIL-Abd-Elilah` are aliases of declared entrants, and accept that the
  commit history's email becomes public with the repository;
- acknowledge the marketing/display licence the terms grant the host, and that a
  winner must return tax forms within 30 days.

## 4. What a judge is most likely to open, in order

1. the repository front page — the claim, four entry points, one hero image
2. [the results page](../docs/index.html) — every headline beside the file that
   produced it, and a gate they can move themselves
3. [the four-page paper](../PAPER.pdf)
4. [the 4:51 demo](../demo/coldplate_submission.mp4), or the
   [locally narrated one](../demo/coldplate_submission_local_voice.mp4)
5. `bash scripts/judge_demo.sh` — 1–3 minutes warm, serves both thermal
   backends and swaps them

All five were checked from a clean Linux clone, with the real containers, on
2026-08-29.
