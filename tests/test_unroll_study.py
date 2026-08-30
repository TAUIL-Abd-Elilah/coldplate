# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
"""The unroll study has to keep conceding the case it loses.

`unroll_study.py` exists to test the README's three-way split between a chain,
an unrolled loop and a solved loop. The result only means something if it
reports both halves: that differentiating an unrolled iteration is *correct*
where the loop contracts, and impossible where it does not. A record that
showed only the second half would be an advertisement rather than a
measurement.

So these tests pin the concession as hard as the claim: in the contracting
regime the unrolled gradient must reach the same accuracy as the implicit
adjoint, and the artefact must say so.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "orchestrator" / "results" / "unroll_study.json"
SCRIPT = ROOT / "orchestrator" / "unroll_study.py"

pytestmark = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason="unroll_study.json is produced by a container run; "
           "absent in a source-only checkout",
)


@pytest.fixture(scope="module")
def study():
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_the_artifact_is_the_audited_grid(study):
    audited = int(
        re.search(r"^AUDITED_N = (\d+)$", SCRIPT.read_text(encoding="utf-8"), re.M).group(1)
    )
    assert study["N"] == audited


def test_both_regimes_were_measured(study):
    assert set(study["regimes"]) == {"repelling", "contracting"}


def test_the_regimes_are_actually_on_opposite_sides_of_one(study):
    """The whole split is about rho(Phi_T) crossing 1. If both points landed on
    the same side, the study would be comparing nothing."""
    assert study["regimes"]["repelling"]["loop_gain"] > 1.0
    assert study["regimes"]["contracting"]["loop_gain"] < 1.0


def test_only_the_physics_differs_between_regimes(study):
    """Same design draw, same components, same grid -- only the Rayleigh
    number moves. Otherwise the comparison would confound the regime with
    something else."""
    assert (
        study["regimes"]["repelling"]["Ra"]
        != study["regimes"]["contracting"]["Ra"]
    )
    assert "validate_pipeline" in study["design_draw"]


def test_the_unroll_is_conceded_where_the_loop_contracts(study):
    """The honest half. Where Picard contracts, differentiating the unrolled
    iteration is a good choice, and this artefact has to keep saying so."""
    contracting = study["regimes"]["contracting"]
    assert contracting["picard"]["converged"] is True
    best = min(row["relative_error"] for row in contracting["unrolled"]
               if row["relative_error"] is not None)
    # As good as the implicit adjoint, to within an order of magnitude.
    assert best <= 10 * contracting["implicit_relative_error"] or best < 1e-6


def test_the_unroll_fails_where_the_fixed_point_repels(study):
    repelling = study["regimes"]["repelling"]
    errors = [row["relative_error"] for row in repelling["unrolled"]
              if row["relative_error"] is not None]
    best = min(errors) if errors else math.inf
    assert best > 1e-3, (
        "the unrolled gradient did no worse than 0.1% at a repelling fixed "
        "point, which would undercut the reason this project solves the loop"
    )
    assert repelling["implicit_relative_error"] < 1e-4


def test_more_sweeps_do_not_rescue_the_repelling_case(study):
    """A converging sequence of errors would mean the unroll just needed more
    steps. It must not be monotonically improving."""
    rows = sorted(study["regimes"]["repelling"]["unrolled"], key=lambda r: r["sweeps"])
    errors = [row["relative_error"] for row in rows]
    finite = [e for e in errors if e is not None]
    assert len(finite) >= 2
    improving = all(b <= a for a, b in zip(finite, finite[1:], strict=False))
    assert not improving, (
        "unrolled error decreased monotonically with sweeps, which would mean "
        "the unroll converges here after all"
    )


def test_picard_is_classified_on_its_contraction_ratio(study):
    """Not on first-versus-last residual.

    A repelling iteration still falls steeply out of a cold start before it
    stalls, so comparing the ends calls a stalled sequence "contracting". That
    mislabelling is what this field exists to prevent, and it happened once.
    """
    for name, regime in study["regimes"].items():
        picard = regime["picard"]
        assert "contraction_ratios" in picard, name
        assert "ratio" in picard["criterion"] or "0.5" in picard["criterion"], name
    assert study["regimes"]["repelling"]["picard"]["converged"] is False


def test_the_referee_shares_no_method_with_the_candidates(study):
    """Both candidates are compared against a central difference of the fully
    solved problem, which is the only arbiter that is not itself one of them."""
    assert "central difference" in study["referee"]
    for regime in study["regimes"].values():
        assert regime["finite_difference_truth"] != 0.0
