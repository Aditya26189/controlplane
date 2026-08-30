"""What can be derived about cost and feasibility, and what cannot.

**Read this before quoting anything from here.**

``DECISIONS.md`` 096 recorded that Phase 6 economics — ``sizing.py``, the price
list, the stratified estimator, ``test_no_scenario_mixing`` — was specified in
five contract documents and never built. That is still true. This package is
**not** that module and does not close that gap.

What it does contain is the subset that needs no new measurement, because it is
arithmetic over numbers already in ``results/``:

- :mod:`controlplane.economics.feasibility` — the abstention floor. Given a
  measured base error rate and a declared target risk, the minimum fraction of
  traffic that *any* selector must abstain on, however good. An impossibility
  result, not an estimate.
- :mod:`controlplane.economics.review` — review volume and the sample size
  needed to estimate recall to a declared margin, from the measured operating
  points and one declared workload.

**Still not built, and still cited by five contracts:** ``sizing.py``, the
computed price list, the Neyman allocation schedule, the blinded label queue,
Cohen's κ, and ``test_no_scenario_mixing``. Any cost, headcount or saving figure
remains a hand-derived declared estimate and must be labelled one. See
``DECISIONS.md`` 099 for exactly which line moved and which did not.

The distinction this package keeps: a **measured** input comes from an artifact
in ``results/``; a **declared** input comes from ``config.yaml`` and is a choice
someone made. Every function here labels which of its inputs are which, and
every output carries that labelling through.
"""

from .feasibility import (
    AbstentionFloor,
    AchievedRisk,
    abstention_floor,
    achieved_risk,
    feasibility_curve,
)
from .review import (
    ReviewSizing,
    recall_sample_size,
    review_volume,
)

__all__ = [
    "AbstentionFloor",
    "AchievedRisk",
    "ReviewSizing",
    "abstention_floor",
    "achieved_risk",
    "feasibility_curve",
    "recall_sample_size",
    "review_volume",
]
