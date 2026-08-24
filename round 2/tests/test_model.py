"""Record-model invariants, including two of the three Phase 1 gate tests.

``test_warrant_key`` and ``test_yield_vs_rate`` are named in ``SPEC.md`` §10 and
live here. The third, ``test_hash_chain``, is in ``test_store.py`` beside the
ledger it exercises.
"""

from __future__ import annotations

import dataclasses
from datetime import timedelta

import pytest

from src.model import (
    AccessTier,
    Action,
    Category,
    Certificate,
    CertificateError,
    ControlResult,
    EnvelopeState,
    Finding,
    FindingError,
    Metric,
    MetricError,
    MetricKind,
    OperatingPoint,
    Resolution,
    SerdeError,
    Severity,
    Span,
    Warrant,
    WarrantError,
    WarrantKey,
    WarrantMetrics,
    WarrantStatus,
    canonical_json,
    content_hash,
    from_jsonable,
    to_jsonable,
    utc_now,
)

from .factories import (
    PASSING_CONTROLS,
    failing_controls,
    make_certificate,
    make_envelope,
    make_envelope_match,
    make_finding,
    make_metrics,
    make_operating_point,
    make_warrant,
)


# --------------------------------------------------------------------------- #
# Gate test: warrants are keyed by all three elements
# --------------------------------------------------------------------------- #


def test_warrant_key() -> None:
    """The same detector on two envelopes yields two distinct warrants.

    ``CLAUDE.md`` invariant 1. An envelope violation is a property of the input
    distribution, so keying by detector alone would let long-context traffic
    inherit numbers measured on short-context traffic — which is the specific
    unbacked claim this project exists to refuse.
    """
    short = make_warrant(eval_set_id="triviaqa-600")
    long = make_warrant(eval_set_id="triviaqa-longctx-600")

    assert short.key != long.key
    assert short.warrant_id != long.warrant_id
    assert short.key.eval_set_id != long.key.eval_set_id
    assert short.detector_id == long.detector_id

    # ... and the same detector at two operating points is also two warrants,
    # because recall at one threshold says nothing about recall at another.
    conservative = make_warrant(operating_point_id="P-conservative")
    aggressive = make_warrant(operating_point_id="P-aggressive")
    assert conservative.key != aggressive.key
    assert conservative.warrant_id != aggressive.warrant_id


def test_warrant_key_requires_all_three_elements() -> None:
    for missing in ("detector_id", "operating_point_id", "eval_set_id"):
        parts = {
            "detector_id": "probe",
            "operating_point_id": "P-1",
            "eval_set_id": "triviaqa-600",
        }
        parts[missing] = ""
        with pytest.raises(WarrantError, match="all three elements"):
            WarrantKey(**parts)


def test_warrant_id_is_content_derived_and_stable() -> None:
    """Two runs of the same validation on the same code produce the same id."""
    key = WarrantKey("probe", "P-1", "triviaqa-600")
    first = Warrant.compute_id(key, "1.0.0+abc", "run-1")
    assert first == Warrant.compute_id(key, "1.0.0+abc", "run-1")
    assert first != Warrant.compute_id(key, "1.0.1+abc", "run-1")
    assert first != Warrant.compute_id(key, "1.0.0+abc", "run-2")


def test_envelope_must_match_the_key(project_root) -> None:
    """A claim and the distribution it was measured on cannot come apart."""
    warrant = make_warrant(eval_set_id="triviaqa-600")
    with pytest.raises(WarrantError, match="envelope describes"):
        dataclasses.replace(warrant, envelope=make_envelope("hinglish-pii-200"))


# --------------------------------------------------------------------------- #
# Gate test: yield is exact, rate is estimated
# --------------------------------------------------------------------------- #


def test_yield_vs_rate() -> None:
    """An EXACT metric carries no interval; an ESTIMATED one always does.

    ``CLAUDE.md``, "the single most important distinction in this codebase".
    Conflating them converts a free exact claim into an unbacked estimate and
    nobody notices, which is why the rule is in the constructor rather than in
    a renderer.
    """
    exact = Metric("confirmed_errors", 850, MetricKind.EXACT, n=2984, unit="count")
    assert not exact.has_interval
    assert exact.ci_low is None and exact.ci_high is None
    assert exact.width is None
    assert "exact" in exact.render()

    estimated = Metric(
        "recall", 0.1416, MetricKind.ESTIMATED, n=600,
        ci_low=0.10, ci_high=0.19, ci_level=0.95, estimator="bootstrap-percentile-1000",
    )
    assert estimated.has_interval
    assert "n=600" in estimated.render() and "95% CI" in estimated.render()

    # An exact count with an interval is a rate wearing a count's label.
    with pytest.raises(MetricError, match="EXACT metric carries no interval"):
        Metric("confirmed_errors", 850, MetricKind.EXACT, n=2984, ci_low=800, ci_high=900)

    # An estimate without one is a point estimate reaching a user.
    with pytest.raises(MetricError, match="must carry both bounds"):
        Metric("recall", 0.14, MetricKind.ESTIMATED, n=600)


def test_estimated_metric_must_name_its_n_and_estimator() -> None:
    """Invariant 4: every interval names its n. And its construction."""
    with pytest.raises(MetricError, match="must name its n"):
        Metric("recall", 0.14, MetricKind.ESTIMATED, n=0, ci_low=0.1, ci_high=0.2,
               ci_level=0.95, estimator="bootstrap")
    with pytest.raises(MetricError, match="how its interval was produced"):
        Metric("recall", 0.14, MetricKind.ESTIMATED, n=600, ci_low=0.1, ci_high=0.2,
               ci_level=0.95)


def test_value_outside_its_own_interval_is_rejected() -> None:
    """Usually means the estimate and the interval came from different data."""
    with pytest.raises(MetricError, match="outside its own interval"):
        Metric("recall", 0.42, MetricKind.ESTIMATED, n=600, ci_low=0.10, ci_high=0.19,
               ci_level=0.95, estimator="bootstrap")


@pytest.mark.parametrize("name", ["f1", "F1", "f1_score", "macro_f1", "fbeta", "f_measure"])
def test_blended_scores_are_refused_by_name(name: str) -> None:
    """Invariant 5, enforced where a label is built, not only where it is typed."""
    with pytest.raises(MetricError, match="blended precision/recall"):
        Metric(name, 0.4, MetricKind.ESTIMATED, n=600, ci_low=0.3, ci_high=0.5,
               ci_level=0.95, estimator="bootstrap")


def test_precision_and_recall_are_both_required() -> None:
    """A warrant claiming one without the other is unconstructible."""
    fields = {f.name for f in dataclasses.fields(WarrantMetrics)}
    assert {"precision", "recall"} <= fields
    with pytest.raises(TypeError):
        WarrantMetrics(  # type: ignore[call-arg]
            auroc=make_metrics().auroc,
            recall=make_metrics().recall,
            flag_rate=make_metrics().flag_rate,
            confirmed_errors=make_metrics().confirmed_errors,
        )


def test_confirmed_errors_must_be_an_exact_count() -> None:
    metrics = make_metrics()
    with pytest.raises(MetricError, match="confirmed_errors must be EXACT"):
        dataclasses.replace(
            metrics,
            confirmed_errors=Metric(
                "confirmed_errors", 850, MetricKind.ESTIMATED, n=2984,
                ci_low=800, ci_high=900, ci_level=0.95, unit="count",
                estimator="bootstrap",
            ),
        )


def test_rates_must_be_estimated() -> None:
    metrics = make_metrics()
    with pytest.raises(MetricError, match="recall must be ESTIMATED"):
        dataclasses.replace(
            metrics, recall=Metric("recall", 0.14, MetricKind.EXACT, n=600)
        )


def test_lift_carries_recall_interval_and_the_flag_rate_it_used() -> None:
    """Lift is never a bare ratio (``DECISIONS.md`` 018)."""
    lift = make_metrics().lift
    assert lift.kind is MetricKind.ESTIMATED
    assert lift.has_interval
    assert "flag rate" in (lift.estimator or "")
    assert lift.value == pytest.approx(0.1416 / 0.0148, rel=1e-9)


def test_widening_only_ever_loses_precision() -> None:
    """The STALE rung widens; nothing narrows without new evidence."""
    recall = make_metrics().recall
    wider = recall.widened(2.0, "envelope drift, PSI 0.14")
    assert wider.width > recall.width
    assert wider.value == recall.value
    assert "widened" in wider.estimator
    with pytest.raises(MetricError, match="Narrowing an interval"):
        recall.widened(0.5, "wishful thinking")
    with pytest.raises(MetricError, match="no interval to widen"):
        make_metrics().confirmed_errors.widened(2.0, "n/a")


# --------------------------------------------------------------------------- #
# Invariant 3: refusal has no override
# --------------------------------------------------------------------------- #


def test_failed_control_forces_refused() -> None:
    """No argument, flag or environment variable promotes a failed run."""
    for status in (WarrantStatus.VALID, WarrantStatus.STALE, WarrantStatus.REVOKED):
        with pytest.raises(WarrantError, match="failed, so this warrant cannot hold"):
            make_warrant(controls=failing_controls(), status=status,
                         status_reason="attempted override")

    refused = make_warrant(
        controls=failing_controls(), status=WarrantStatus.REFUSED,
        status_reason="label_shuffle scored 0.71, outside [0.45, 0.55]",
    )
    assert refused.failed_controls()[0].control == "label_shuffle"
    assert not refused.can_be_relied_upon(utc_now())


def test_a_refused_warrant_cannot_be_relabelled_valid() -> None:
    """with_status() goes through __post_init__ like every other construction."""
    refused = make_warrant(
        controls=failing_controls(), status=WarrantStatus.REFUSED,
        status_reason="null_feature scored 0.71",
    )
    with pytest.raises(WarrantError, match="failed, so this warrant cannot hold"):
        refused.with_status(WarrantStatus.VALID, "operational necessity")


def test_a_control_cannot_report_a_pass_it_did_not_achieve() -> None:
    with pytest.raises(WarrantError, match="passed with a negative margin"):
        ControlResult("canary", True, 0.9, "recall == 1.0", -0.1)
    with pytest.raises(WarrantError, match="failed with a positive margin"):
        ControlResult("canary", False, 1.0, "recall == 1.0", 0.1)
    with pytest.raises(WarrantError, match="pass condition must be stated"):
        ControlResult("canary", True, 1.0, "", 0.0)


def test_non_valid_status_requires_a_reason() -> None:
    """A refusal a reader cannot interpret is indistinguishable from a bug."""
    with pytest.raises(WarrantError, match="requires a status_reason"):
        dataclasses.replace(
            make_warrant(), status=WarrantStatus.REVOKED, status_reason=None
        )
    with pytest.raises(WarrantError, match="must state its reason"):
        make_warrant().with_status(WarrantStatus.REVOKED, "")


# --------------------------------------------------------------------------- #
# Invariant 2: three states, and UNVALIDATED is not a record
# --------------------------------------------------------------------------- #


def test_unvalidated_is_not_a_constructible_warrant_status() -> None:
    """``DECISIONS.md`` 024. An unvalidated cell has no metrics to record."""
    with pytest.raises(WarrantError, match="absence of a warrant"):
        make_warrant(status=WarrantStatus.UNVALIDATED, status_reason="never tested here")


def test_the_three_states_are_not_orderable() -> None:
    """No ordering in which REFUSED is 'worse than' UNVALIDATED.

    Offering that comparison is how three states collapse into two.
    """
    with pytest.raises(TypeError):
        _ = WarrantStatus.REFUSED < WarrantStatus.UNVALIDATED  # type: ignore[operator]


def test_reliance_requires_valid_and_unexpired() -> None:
    now = utc_now()
    fresh = make_warrant(issued_at=now - timedelta(hours=1), ttl_hours=24)
    assert fresh.can_be_relied_upon(now)

    aged = make_warrant(issued_at=now - timedelta(hours=48), ttl_hours=24)
    assert aged.status is WarrantStatus.VALID
    assert aged.is_expired(now)
    assert not aged.can_be_relied_upon(now)

    stale = fresh.with_status(WarrantStatus.STALE, "PSI 0.14 on token_length")
    assert not stale.can_be_relied_upon(now)
    assert stale.status.was_ever_measured_here


# --------------------------------------------------------------------------- #
# Selection never touches test
# --------------------------------------------------------------------------- #


def test_operating_point_refuses_a_threshold_selected_on_test() -> None:
    with pytest.raises(FindingError, match="selected_on must be one of"):
        make_operating_point(selected_on="test")


def test_operating_point_requires_its_objective() -> None:
    with pytest.raises(FindingError, match="objective is required"):
        OperatingPoint("P-1", "probe", 0.5, "validation", "")


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #


def test_unwarranted_finding_is_an_honest_state_not_an_error() -> None:
    finding = make_finding(warrant_id=None)
    assert not finding.is_warranted
    assert finding.confidence == pytest.approx(0.73)


def test_finding_requires_evidence() -> None:
    """A finding without offsets cannot be explained to the person it affects."""
    with pytest.raises(FindingError, match="at least one evidence span"):
        Finding(
            finding_id="F-1", detector_id="probe", detector_version="1.0.0",
            category=Category.PII, severity=Severity.HIGH, confidence=0.9,
            evidence_spans=(), access_tier=AccessTier.T3_TEXT, latency_ms=1.0,
        )


def test_envelope_match_insufficient_data_is_not_inside() -> None:
    """'We have not looked yet' is not 'we looked and it is fine'."""
    unknown = make_envelope_match(state=EnvelopeState.INSUFFICIENT_DATA, n_window=12)
    assert not unknown.is_inside
    inside = make_envelope_match(state=EnvelopeState.INSIDE)
    assert inside.is_inside


def test_envelope_match_max_psi_must_match_its_features() -> None:
    with pytest.raises(FindingError, match="does not match the largest"):
        dataclasses.replace(make_envelope_match(), max_psi=0.9)


def test_missing_envelope_feature_crashes_rather_than_returning_none() -> None:
    """A drift check that skips a dimension reports stability it never measured."""
    with pytest.raises(FindingError, match="carries no feature"):
        make_envelope().feature("reference_perplexity")


# --------------------------------------------------------------------------- #
# Certificates
# --------------------------------------------------------------------------- #


def test_certificate_round_trips_through_json() -> None:
    """The Phase 1 gate: a certificate round-trips."""
    warrant = make_warrant()
    certificate = make_certificate(warrant=warrant)
    restored = from_jsonable(Certificate, to_jsonable(certificate))
    assert restored == certificate
    assert content_hash(restored) == content_hash(certificate)


def test_certificate_claiming_valid_bounds_must_cite_a_warrant() -> None:
    with pytest.raises(CertificateError, match="no warrant is cited"):
        Certificate(
            certificate_id="C-1", request_id="R-1", session_id="S-1",
            timestamp=utc_now(), findings=(make_finding(),),
            resolution=Resolution(Action.ALLOW, "3.1", "sha256:p"),
            warrants_relied_upon=(), weakest_warrant_status=WarrantStatus.VALID,
            claimed_bounds={}, envelope_match=make_envelope_match(),
            access_tier_available=AccessTier.T1_ACTIVATIONS,
        )


def test_certificate_action_must_name_the_finding_that_caused_it() -> None:
    """An action nobody can trace to a finding is an action nobody can appeal."""
    with pytest.raises(CertificateError, match="must name the finding"):
        Resolution(Action.BLOCK, "3.1", "sha256:p", rationale="because")


def test_certificate_cannot_cite_a_finding_it_does_not_carry() -> None:
    with pytest.raises(CertificateError, match="not on this certificate"):
        Certificate(
            certificate_id="C-1", request_id="R-1", session_id="S-1",
            timestamp=utc_now(), findings=(make_finding("F-1"),),
            resolution=Resolution(
                Action.BLOCK, "3.1", "sha256:p", ("F-99",), "R-1", "blocked"
            ),
            warrants_relied_upon=(), weakest_warrant_status=WarrantStatus.UNVALIDATED,
            claimed_bounds={}, envelope_match=make_envelope_match(),
            access_tier_available=AccessTier.T3_TEXT,
        )


def test_unwarranted_certificate_is_valid_and_says_so() -> None:
    """UNVALIDATED reaches a certificate intact; it does not become VALID."""
    certificate = make_certificate(warrant=None)
    assert certificate.weakest_warrant_status is WarrantStatus.UNVALIDATED
    assert certificate.claimed_bounds == {}
    assert certificate.warrants_relied_upon == ()
    assert certificate.unchecked


def test_claimed_bounds_keep_the_kind_of_every_metric() -> None:
    """A reader can tell an exact count from an estimate without the schema."""
    bounds = make_warrant().claimed_bounds()
    assert bounds["confirmed_errors"]["kind"] == "EXACT"
    assert "ci_low" not in bounds["confirmed_errors"]
    assert bounds["recall"]["kind"] == "ESTIMATED"
    assert bounds["recall"]["ci_low"] < bounds["recall"]["value"] < bounds["recall"]["ci_high"]
    assert bounds["recall"]["n"] == 600
    assert "precision" in bounds and "recall" in bounds


def test_a_sealed_certificate_cannot_be_resealed() -> None:
    """Re-sealing is how an edited record would be laundered back in."""
    sealed = make_certificate().sealed_with("prev", "self")
    assert sealed.is_sealed
    with pytest.raises(CertificateError, match="already sealed"):
        sealed.sealed_with("other", "other")
    assert not sealed.unsealed().is_sealed


# --------------------------------------------------------------------------- #
# Serde
# --------------------------------------------------------------------------- #


def test_canonical_json_is_stable_and_keeps_non_latin_text() -> None:
    """Devanagari must hash as itself, not as escapes."""
    warrant = make_warrant()
    assert canonical_json(warrant) == canonical_json(warrant)
    rendered = canonical_json({"text": "आधार"})
    assert "आधार" in rendered


def test_decoding_rejects_unknown_and_missing_fields() -> None:
    body = to_jsonable(make_warrant())
    with pytest.raises(SerdeError, match="unknown field"):
        from_jsonable(Warrant, {**body, "surprise": 1})
    without = {k: v for k, v in body.items() if k != "n_test"}
    with pytest.raises(SerdeError, match="missing field"):
        from_jsonable(Warrant, without)


def test_decoding_reapplies_every_invariant() -> None:
    """A record that violates an invariant cannot be read back into a valid object."""
    body = to_jsonable(make_warrant())
    body["status"] = "VALID"
    body["controls"][1]["passed"] = False
    body["controls"][1]["margin"] = -0.16
    with pytest.raises(WarrantError, match="failed, so this warrant cannot hold"):
        from_jsonable(Warrant, body)


def test_naive_timestamps_are_refused() -> None:
    from datetime import datetime as _dt

    with pytest.raises(SerdeError, match="naive datetime"):
        to_jsonable(_dt(2026, 8, 23, 12, 0, 0))
