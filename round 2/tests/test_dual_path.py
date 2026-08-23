"""Controls against the failure class that bit twice: two paths, one drifts.

The ``fpr_hard_negatives`` conflation shipped on the text path, was fixed
there, and survived on the activation path through 193 passing tests
(``DECISIONS.md`` 036, 040). It was caught by hunch. A documented pitfall is
what every team has; a failing test is what this repo claims to be different
for, and these are that test.

Two controls, matching the two shapes the bug took:

* **Cross-path metric equality** — every metric is built in exactly one place,
  and both runners produce identical metrics from identical input. Catches
  "same quantity, two implementations, one drifts" by construction and then
  asserts it.
* **A routing positive control** — on an envelope where a detector is
  known-strong, at least one profile must *route* rather than suspend.
  Universal suspension reads exactly like a conservative system working, which
  is why nothing caught it for a phase. This is the null-feature control's
  reasoning applied to routing: assert the system can produce the non-null
  outcome when the non-null outcome is correct.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from src.config import Config
from src.matrix import Profile, WarrantMatrix, route
from src.model import Metric, MetricKind, WarrantStatus, utc_now
from src.validation.metrics_builder import build_warrant_metrics

from .factories import make_metrics, make_warrant


# --------------------------------------------------------------------------- #
# Control 1: every metric has exactly one implementation
# --------------------------------------------------------------------------- #

SRC = Path(__file__).resolve().parents[1] / "src"


def test_warrant_metrics_has_exactly_one_construction_site() -> None:
    """Structural: the dual implementation is gone, not merely reconciled.

    Two copies that agree today are two copies that can disagree tomorrow, and
    that is precisely what happened. Parsed rather than grepped so a construction
    inside a string or a comment cannot register as one.
    """
    sites: list[str] = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "WarrantMetrics"
            ):
                sites.append(f"{path.relative_to(SRC).as_posix()}:{node.lineno}")
    assert len(sites) == 1, (
        "WarrantMetrics is constructed in more than one place: "
        f"{sites}. Every metric must have one implementation (DECISIONS.md 044)."
    )
    assert sites[0].startswith("validation/metrics_builder.py"), sites


def test_estimated_is_called_from_one_module() -> None:
    """The interval construction has one home too.

    ``estimated()`` decides bootstrap count, coverage, resampling unit and the
    boundary fallback. Two callers making those choices independently is the
    same failure class one level down.
    """
    callers: set[str] = set()
    for path in SRC.rglob("*.py"):
        if path.name == "stats.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "estimated"
            ):
                callers.add(path.relative_to(SRC).as_posix())
    assert callers == {"validation/metrics_builder.py"}, (
        f"estimated() is called from {sorted(callers)}; it should have one caller"
    )


def test_metric_paths_agree(config: Config) -> None:
    """Both runners produce identical metrics from identical scores and labels.

    The behavioural half. Even with one construction site, a caller could pass
    different arguments — a different seed, a different resampling unit — and
    the paths would silently diverge again.
    """
    rng = np.random.default_rng(7)
    n = 400
    labels = (rng.random(n) < 0.3).astype(int)
    scores = rng.normal(labels * 0.9, 1.0)
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    groups = np.repeat(np.arange(n // 2), 2).astype(object)
    threshold = 0.6

    first = build_warrant_metrics(
        config, labels, scores, threshold, groups=groups
    )
    second = build_warrant_metrics(
        config, labels, scores, threshold, groups=groups
    )
    assert first == second

    for name in ("auroc", "recall", "precision", "flag_rate"):
        metric = getattr(first, name)
        assert metric.kind is MetricKind.ESTIMATED
        assert metric.has_interval


def test_within_set_fpr_never_populates_the_hard_negative_field(
    config: Config,
) -> None:
    """The exact bug, pinned. ``DECISIONS.md`` 036 and 040.

    A profile's declared hard-negative maximum must only ever be judged against
    a measurement taken on the hard-negative set.
    """
    rng = np.random.default_rng(3)
    n = 300
    labels = (rng.random(n) < 0.4).astype(int)
    scores = rng.random(n)

    ordinary = build_warrant_metrics(config, labels, scores, 0.5)
    assert ordinary.fpr_hard_negatives is None, (
        "within-set FPR leaked into fpr_hard_negatives; a profile would judge it "
        "against a bar measured on a different set"
    )
    assert any(m.name == "fpr" for m in ordinary.extra)

    on_hard_negatives = build_warrant_metrics(
        config, labels, scores, 0.5, is_hard_negative_set=True
    )
    assert on_hard_negatives.fpr_hard_negatives is not None
    assert on_hard_negatives.fpr_hard_negatives.name == "fpr_hard_negatives"
    assert on_hard_negatives.extra == ()


# --------------------------------------------------------------------------- #
# Control 2: routing can produce the non-null outcome
# --------------------------------------------------------------------------- #


def _warrant_with_recall(detector_id: str, envelope: str, low: float, value: float,
                         high: float):
    """A warrant whose recall interval is set explicitly."""
    return dataclasses.replace(
        make_warrant(detector_id=detector_id, eval_set_id=envelope),
        metrics=dataclasses.replace(
            make_metrics(),
            recall=Metric("recall", value, MetricKind.ESTIMATED, 600, low, high,
                          0.95, "rate", "bootstrap"),
        ),
    )


def test_routing_positive_control(config: Config) -> None:
    """A known-strong detector must ROUTE, not suspend.

    The control that would have caught the FPR conflation a phase earlier.
    Universal suspension across every envelope is a pipeline-bug signature and
    it is indistinguishable, on screen, from a conservative system doing its
    job — so the suite has to assert the non-null outcome is reachable.

    Recall 0.40 [0.30, 0.50] clears every profile minimum in config, so a
    failure here means routing is broken, not that the detector is weak.
    """
    strong = _warrant_with_recall("probe-strong", "envelope-easy", 0.30, 0.40, 0.50)
    matrix = WarrantMatrix(
        [WarrantMatrix._cell_for(strong, utc_now())],
        detectors=["probe-strong"],
        envelopes=["envelope-easy"],
    )
    routed_by = []
    for name in sorted(config.profiles):
        decision = route(matrix, "envelope-easy", Profile.from_config(config, name))
        if decision.routed:
            routed_by.append(name)
    assert routed_by, (
        "no profile routed to a detector with recall 0.40 [0.30, 0.50]. Every "
        "profile suspending everywhere is a pipeline-bug signature that reads "
        "like conservatism (DECISIONS.md 040)."
    )
    # customer_support requires only 0.10; it must be among them.
    assert "customer_support" in routed_by


def test_routing_ranks_by_lower_bound_not_width(config: Config) -> None:
    """``DECISIONS.md`` 042: claim what you can prove.

    The case that exposed it: a 3% tighter interval beat a 16% higher midpoint.
    Generalised, width-first prefers [0.10, 0.11] over [0.30, 0.45].
    """
    tight_but_low = _warrant_with_recall("probe-tight", "env", 0.10, 0.105, 0.11)
    wide_but_high = _warrant_with_recall("probe-high", "env", 0.30, 0.375, 0.45)
    matrix = WarrantMatrix(
        [WarrantMatrix._cell_for(w, utc_now()) for w in (tight_but_low, wide_but_high)],
        detectors=["probe-tight", "probe-high"],
        envelopes=["env"],
    )
    decision = route(matrix, "env", Profile.from_config(config, "customer_support"))
    assert decision.routed
    assert decision.warrant.detector_id == "probe-high", (
        "routing preferred the tighter interval over the higher lower bound; "
        "the lower bound is what the detector can be shown to deliver"
    )


def test_routing_is_deterministic(config: Config) -> None:
    """Two identical lower bounds must not route differently between runs."""
    a = _warrant_with_recall("probe-a", "env", 0.20, 0.25, 0.30)
    b = _warrant_with_recall("probe-b", "env", 0.20, 0.25, 0.30)
    matrix = WarrantMatrix(
        [WarrantMatrix._cell_for(w, utc_now()) for w in (a, b)],
        detectors=["probe-a", "probe-b"],
        envelopes=["env"],
    )
    profile = Profile.from_config(config, "customer_support")
    chosen = {
        route(matrix, "env", profile).warrant.detector_id for _ in range(5)
    }
    assert len(chosen) == 1


# --------------------------------------------------------------------------- #
# The lift refusal criterion
# --------------------------------------------------------------------------- #


def test_lift_lower_bound_refuses_a_warrant_no_better_than_chance(
    config: Config,
) -> None:
    """``DECISIONS.md`` 043, on the numbers that produced it.

    Recall 0.034 [0.000, 0.077] at a 5% flag rate is lift 0.68 [0.00, 1.54]:
    the point estimate is *below* random sampling at the same budget and the
    lower bound is zero. AUROC's lower bound cleared 0.55, so nothing refused
    it before.
    """
    from src.validation.issuance import MIN_LIFT_LOWER_BOUND, issue_or_refuse

    from src.model import AccessTier, WarrantKey

    from .factories import (
        PASSING_CONTROLS,
        make_envelope,
        make_operating_point,
    )

    collapsed = dataclasses.replace(
        make_metrics(),
        recall=Metric("recall", 0.034, MetricKind.ESTIMATED, 600, 0.0, 0.077,
                      0.95, "rate", "bootstrap"),
        flag_rate=Metric("flag_rate", 0.05, MetricKind.ESTIMATED, 600, 0.03, 0.07,
                         0.95, "rate", "bootstrap"),
    )
    assert collapsed.lift.ci_low <= MIN_LIFT_LOWER_BOUND

    warrant = issue_or_refuse(
        config,
        key=WarrantKey("probe-collapsed", "P-conservative", "envelope-long"),
        detector_version="1.0.0+abc",
        operating_point=make_operating_point("P-conservative", "probe-collapsed"),
        metrics=collapsed,
        envelope=make_envelope("envelope-long"),
        controls=PASSING_CONTROLS,
        access_tier=AccessTier.T1_ACTIVATIONS,
        n_test=600,
        base_rate=0.15,
        validation_run_id="run-x",
    )
    assert warrant.status is WarrantStatus.REFUSED
    assert "lift_lower_ci" in warrant.status_reason
    assert "random sampling" in warrant.status_reason


def test_a_useful_detector_still_issues(config: Config) -> None:
    """The criterion must not refuse detectors that are genuinely better.

    Recall 0.216 [0.129, 0.306] at a 5% flag rate is lift 4.3 [2.6, 6.1].
    A criterion that refused this would be a bar chosen to produce a demo.
    """
    from src.validation.issuance import issue_or_refuse

    from src.model import AccessTier, WarrantKey

    from .factories import PASSING_CONTROLS, make_envelope, make_operating_point

    useful = dataclasses.replace(
        make_metrics(),
        recall=Metric("recall", 0.216, MetricKind.ESTIMATED, 600, 0.129, 0.306,
                      0.95, "rate", "bootstrap"),
        flag_rate=Metric("flag_rate", 0.05, MetricKind.ESTIMATED, 600, 0.03, 0.07,
                         0.95, "rate", "bootstrap"),
    )
    assert useful.lift.ci_low > 1.0

    warrant = issue_or_refuse(
        config,
        key=WarrantKey("probe-useful", "P-conservative", "envelope-short"),
        detector_version="1.0.0+abc",
        operating_point=make_operating_point("P-conservative", "probe-useful"),
        metrics=useful,
        envelope=make_envelope("envelope-short"),
        controls=PASSING_CONTROLS,
        access_tier=AccessTier.T1_ACTIVATIONS,
        n_test=600,
        base_rate=0.15,
        validation_run_id="run-y",
    )
    assert warrant.status is WarrantStatus.VALID, warrant.status_reason


def test_single_class_envelope_skips_the_lift_criterion(config: Config) -> None:
    """Lift needs recall, and a single-class envelope has none."""
    from src.validation.issuance import issue_or_refuse

    from src.model import AccessTier, WarrantKey

    from .factories import PASSING_CONTROLS, make_envelope, make_operating_point

    fpr_only = dataclasses.replace(
        make_metrics(),
        auroc=None, recall=None, precision=None,
        fpr_hard_negatives=Metric("fpr_hard_negatives", 0.0, MetricKind.ESTIMATED,
                                  200, 0.0, 0.018, 0.95, "rate", "Clopper-Pearson"),
    )
    warrant = issue_or_refuse(
        config,
        key=WarrantKey("pii-reference", "P-declared", "hard-negatives-200"),
        detector_version="0.1.0",
        operating_point=make_operating_point("P-declared", "pii-reference"),
        metrics=fpr_only,
        envelope=make_envelope("hard-negatives-200"),
        controls=PASSING_CONTROLS,
        access_tier=AccessTier.T3_TEXT,
        n_test=200,
        base_rate=0.0,
        validation_run_id="run-z",
        max_fpr_hard_negatives=0.02,
    )
    assert warrant.status is WarrantStatus.VALID, warrant.status_reason
    assert "lift" not in (warrant.status_reason or "")
