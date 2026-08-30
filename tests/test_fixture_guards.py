"""Guards that stop a fixture number being read as a measurement.

Every activation-tier number in this repo is currently produced by a generator
we wrote. It is internally valid and it is not evidence about a language model.
Until the real extraction lands, the only thing between a fixture number and a
slide is somebody remembering — which is the control this project exists to
argue against, so these tests are that control instead.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from controlplane.config import Config
from controlplane.matrix import WarrantMatrix
from controlplane.model import (
    DistributionEnvelope,
    FindingError,
    Metric,
    MetricKind,
    WarrantStatus,
    utc_now,
)
from controlplane.report.results import FIXTURE_MARKER, REQUIRED_EXTRACTIONS, render_results
from controlplane.validation.metrics_builder import (
    assert_metric_shape_compatible,
    build_warrant_metrics,
)

from .factories import failing_controls, make_envelope, make_metrics, make_warrant


def _matrix(*warrants, detectors=None, envelopes=None) -> WarrantMatrix:
    cells = [WarrantMatrix._cell_for(w, utc_now()) for w in warrants]
    return WarrantMatrix(
        cells,
        detectors=detectors or sorted({w.detector_id for w in warrants}),
        envelopes=envelopes or sorted({w.eval_set_id for w in warrants}),
    )


PROVENANCE = {
    "config_hash": "c89257bc4adc10c2",
    "git_commit": "0" * 40,
    "dirty": False,
    "seed": 1729,
    "timestamp_utc": "2026-08-23T00:00:00+00:00",
}


# --------------------------------------------------------------------------- #
# The renderer refuses fixture numbers
# --------------------------------------------------------------------------- #


def test_results_refuses_to_print_a_synthetic_number() -> None:
    """A number that is not on the page cannot be read off it."""
    synthetic = dataclasses.replace(
        make_warrant(detector_id="probe-fixture", eval_set_id="triviaqa-600-synthetic"),
        envelope=make_envelope("triviaqa-600-synthetic", data_source="synthetic"),
    )
    rendered = render_results(_matrix(synthetic), provenance=PROVENANCE)

    assert FIXTURE_MARKER in rendered
    # The actual recall value must appear nowhere.
    recall = synthetic.metrics.recall
    assert f"{recall.value:.4f}" not in rendered
    assert f"{recall.value:.3f}" not in rendered
    assert "No measured results yet" in rendered


def test_results_prints_a_measured_number(config: Config) -> None:
    """The guard must not swallow real results, or it would be useless."""
    measured = make_warrant(detector_id="pii-reference", eval_set_id="hinglish-pii-200")
    assert measured.envelope.data_source == "measured"
    rendered = render_results(_matrix(measured), provenance=PROVENANCE)

    assert FIXTURE_MARKER not in rendered
    assert f"{measured.metrics.recall.value:.3f}" in rendered
    assert "No measured results yet" not in rendered


def test_an_envelope_without_a_declared_source_is_refused() -> None:
    """Fail-closed: the failure mode is a new path that forgets to set it."""

    class LegacyEnvelope:
        """An envelope from before the field existed."""

        envelope_id = "sha256:legacy"
        eval_set_id = "legacy-set"

    warrant = make_warrant()
    object.__setattr__(warrant, "envelope", LegacyEnvelope())
    rendered = render_results(_matrix(warrant), provenance=PROVENANCE)
    assert FIXTURE_MARKER in rendered
    assert "does not declare a data_source" in rendered


def test_the_matrix_table_masks_fixture_cells_too() -> None:
    """The same hole one level down: the table inside RESULTS.md."""
    synthetic = dataclasses.replace(
        make_warrant(detector_id="probe-fixture", eval_set_id="triviaqa-600-synthetic"),
        envelope=make_envelope("triviaqa-600-synthetic", data_source="synthetic"),
    )
    rendered = render_results(_matrix(synthetic), provenance=PROVENANCE)
    matrix_section = rendered.split("## Warrant matrix", 1)[1].split("## ", 1)[0]
    assert FIXTURE_MARKER in matrix_section
    assert "R=" not in matrix_section


def test_outstanding_extractions_are_listed_until_they_land() -> None:
    """A dependency tracked only in someone's head is one that slips."""
    rendered = render_results(_matrix(make_warrant()), provenance=PROVENANCE)
    assert "Outstanding measurement" in rendered
    for required in REQUIRED_EXTRACTIONS:
        assert required.eval_set_id in rendered

    # Once measured, the row disappears rather than lingering as noise.
    landed = make_warrant(detector_id="probe", eval_set_id="triviaqa-600")
    rendered = render_results(
        _matrix(landed, make_warrant(detector_id="p2", eval_set_id="triviaqa-longctx-600")),
        provenance=PROVENANCE,
    )
    assert "Outstanding measurement" not in rendered


def test_envelope_rejects_an_unknown_data_source() -> None:
    with pytest.raises(FindingError, match="data_source must be"):
        DistributionEnvelope(
            envelope_id="x", eval_set_id="y", n_reference=10,
            features=make_envelope().features, data_source="probably-real",
        )


# --------------------------------------------------------------------------- #
# Lift is not interpretable without its base rate
# --------------------------------------------------------------------------- #


def test_lift_carries_its_ceiling_and_base_rate(config: Config) -> None:
    """``DECISIONS.md`` 047: 1.0 is a base-rate-dependent bar stated as absolute."""
    metrics = make_metrics()
    assert metrics.base_rate is not None
    ceiling = metrics.lift_ceiling
    assert ceiling is not None
    # ceiling = 1 / max(base_rate, flag_rate)
    expected = 1.0 / max(metrics.base_rate, metrics.flag_rate.value)
    assert ceiling == pytest.approx(expected)
    assert "ceiling" in metrics.lift.estimator
    assert "base rate" in metrics.lift.estimator


def test_lift_on_an_enriched_set_is_near_its_ceiling_not_near_useless(
    config: Config,
) -> None:
    """The reading the ceiling exists to correct.

    At base rate 0.51 and flag rate 0.62 the ceiling is 1.61. A measured lift of
    1.28 is 79% of everything achievable, which reads as "strong". Without the
    ceiling it reads as "barely better than chance", and that is wrong.
    """
    metrics = dataclasses.replace(
        make_metrics(),
        recall=Metric("recall", 0.794, MetricKind.ESTIMATED, 200, 0.698, 0.885,
                      0.95, "rate", "bootstrap",
            convention="two_sided_95",
        ),
        flag_rate=Metric("flag_rate", 0.62, MetricKind.ESTIMATED, 200, 0.51, 0.73,
                         0.95, "rate", "bootstrap",
            convention="two_sided_95",
        ),
        base_rate=0.51,
    )
    assert metrics.lift.value == pytest.approx(0.794 / 0.62, rel=1e-6)
    assert metrics.lift_ceiling == pytest.approx(1.0 / 0.62, rel=1e-6)
    assert metrics.lift_fraction_of_ceiling > 0.75


def test_lift_ceiling_is_absent_when_the_base_rate_is(config: Config) -> None:
    """No base rate, no ceiling — and no invented one."""
    metrics = dataclasses.replace(make_metrics(), base_rate=None)
    assert metrics.lift_ceiling is None
    assert metrics.lift_fraction_of_ceiling is None
    assert "ceiling" not in metrics.lift.estimator


def test_the_builder_records_the_base_rate(config: Config) -> None:
    rng = np.random.default_rng(0)
    labels = (rng.random(300) < 0.3).astype(int)
    scores = rng.random(300)
    metrics = build_warrant_metrics(config, labels, scores, 0.5)
    assert metrics.base_rate == pytest.approx(labels.mean())


# --------------------------------------------------------------------------- #
# Shape compatibility, for when the real extraction lands
# --------------------------------------------------------------------------- #


def test_metric_shape_assertion_accepts_different_values(config: Config) -> None:
    """Values must differ between paths; shape must not.

    Exercised on the two paths that exist today — a probe-style scoring and a
    text-style scoring on the same eval set shape — so the assertion is known to
    work before the real extraction needs it.
    """
    rng = np.random.default_rng(1)
    labels = (rng.random(300) < 0.3).astype(int)
    groups = np.repeat(np.arange(150), 2).astype(object)

    fixture_scores = rng.normal(labels * 0.9, 1.0)
    fixture_scores = (fixture_scores - fixture_scores.min()) / np.ptp(fixture_scores)
    measured_scores = rng.normal(labels * 0.4, 1.0)
    measured_scores = (measured_scores - measured_scores.min()) / np.ptp(measured_scores)

    fixture = build_warrant_metrics(
        config, labels, fixture_scores, 0.6, groups=groups
    )
    measured = build_warrant_metrics(
        config, labels, measured_scores, 0.6, groups=groups
    )
    assert fixture.recall.value != measured.recall.value
    assert_metric_shape_compatible(
        fixture, measured, first_name="fixture", second_name="measured"
    )


def test_metric_shape_assertion_catches_a_missing_metric(config: Config) -> None:
    """A path that silently drops recall is the failure this catches."""
    rng = np.random.default_rng(2)
    labels = (rng.random(200) < 0.4).astype(int)
    scores = rng.random(200)
    both = build_warrant_metrics(config, labels, scores, 0.5)
    single_class = build_warrant_metrics(
        config, np.zeros(200, dtype=int), scores, 0.5
    )
    with pytest.raises(AssertionError, match="differ in which metrics exist"):
        assert_metric_shape_compatible(
            both, single_class, first_name="two-class", second_name="single-class"
        )


def test_the_type_already_forbids_a_kind_flip(config: Config) -> None:
    """The shape assertion's kind check is belt-and-braces, and that is worth knowing.

    Attempting to build the divergence it guards against -- an ``EXACT`` count on
    one path and an ``ESTIMATED`` one on the other -- is refused by
    ``WarrantMetrics`` itself. So the assertion cannot be exercised on that axis
    through the real types, which is a stronger position than having a test for
    it: the failure is unconstructible rather than merely detected.
    """
    from controlplane.model import MetricError

    rng = np.random.default_rng(3)
    labels = (rng.random(200) < 0.4).astype(int)
    scores = rng.random(200)
    normal = build_warrant_metrics(config, labels, scores, 0.5)
    with pytest.raises(MetricError, match="confirmed_errors must be EXACT"):
        dataclasses.replace(
            normal,
            confirmed_errors=Metric(
                "confirmed_errors", normal.confirmed_errors.value,
                MetricKind.ESTIMATED, n=200, ci_low=0.0,
                ci_high=normal.confirmed_errors.value + 1, ci_level=0.95,
                unit="count", estimator="bootstrap",
            convention="two_sided_95",
        ),
        )


def test_metric_shape_assertion_catches_a_unit_change(config: Config) -> None:
    """A quantity reported as a rate on one path and a ratio on the other.

    The axis that IS constructible, and the one a normalisation difference
    between the fixture and the real extraction would show up on.
    """
    rng = np.random.default_rng(4)
    labels = (rng.random(200) < 0.4).astype(int)
    scores = rng.random(200)
    normal = build_warrant_metrics(config, labels, scores, 0.5)
    original = normal.extra[0]
    relabelled = dataclasses.replace(
        normal,
        extra=(dataclasses.replace(original, unit="ratio"),),
    )
    with pytest.raises(AssertionError, match="different shape"):
        assert_metric_shape_compatible(
            normal, relabelled, first_name="rate", second_name="ratio"
        )


# --------------------------------------------------------------------------- #
# Routing determinism under a tie
# --------------------------------------------------------------------------- #


def test_routing_tiebreak_is_stated_not_incidental(config: Config) -> None:
    """Phase 5 chooses the fallback detector live; sort order is not good enough.

    Two warrants with identical recall intervals must route the same way
    regardless of the order the matrix yields them, which depends on ledger
    insertion order.
    """
    from controlplane.matrix import Profile, route

    def warrant(detector_id: str):
        return dataclasses.replace(
            make_warrant(detector_id=detector_id, eval_set_id="env"),
            metrics=dataclasses.replace(
                make_metrics(),
                recall=Metric("recall", 0.25, MetricKind.ESTIMATED, 600, 0.20, 0.30,
                              0.95, "rate", "bootstrap",
            convention="two_sided_95",
        ),
            ),
        )

    forward = _matrix(warrant("probe-b"), warrant("probe-a"), envelopes=["env"])
    backward = _matrix(warrant("probe-a"), warrant("probe-b"), envelopes=["env"])
    profile = Profile.from_config(config, "customer_support")

    first = route(forward, "env", profile).warrant.detector_id
    second = route(backward, "env", profile).warrant.detector_id
    assert first == second == "probe-a", "tiebreak is not order-independent"


def test_an_unvalidated_cell_never_outranks_a_measured_one(config: Config) -> None:
    """Invariant 2: an absence is not a weak positive, however promising."""
    from controlplane.matrix import Profile, route

    weak_but_measured = dataclasses.replace(
        make_warrant(detector_id="probe-measured", eval_set_id="env"),
        metrics=dataclasses.replace(
            make_metrics(),
            recall=Metric("recall", 0.15, MetricKind.ESTIMATED, 600, 0.11, 0.20,
                          0.95, "rate", "bootstrap",
            convention="two_sided_95",
        ),
        ),
    )
    matrix = _matrix(
        weak_but_measured,
        detectors=["probe-measured", "probe-never-tried"],
        envelopes=["env"],
    )
    decision = route(matrix, "env", Profile.from_config(config, "customer_support"))
    assert decision.routed
    assert decision.warrant.detector_id == "probe-measured"
    assert any(
        k.detector_id == "probe-never-tried" for k in decision.enqueued_for_validation
    )
