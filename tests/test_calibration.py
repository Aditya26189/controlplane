"""The calibration claim: a warrant's second, separable assertion.

A warrant says how well a detector *ranks* and what its operating point
*spends*. These tests pin the property that made the distinction necessary: a
detector can hold one claim while losing the other, and the weaker outcome must
not read as the stronger one.
"""

from __future__ import annotations

import pytest

from controlplane.model.enums import CalibrationStatus, MetricKind
from controlplane.model.findings import OperatingPoint
from controlplane.model.metrics import Metric
from controlplane.model.serde import from_jsonable, to_jsonable
from controlplane.validation.calibration import assess_calibration, n_to_detect_deviation


def _flag_rate(value: float, low: float, high: float, n: int = 600) -> Metric:
    return Metric(
        name="flag_rate",
        value=value,
        kind=MetricKind.ESTIMATED,
        ci_low=low,
        ci_high=high,
        n=n,
        ci_level=0.95,
        estimator="bootstrap-percentile-1000 over questions, seed=1729",
            convention="two_sided_95",
        )


def _operating_point(target: float | None = 0.05) -> OperatingPoint:
    return OperatingPoint(
        operating_point_id="P-conservative",
        detector_id="probe",
        threshold=0.9,
        objective="flag_rate",
        selected_on="validation",
        target_flag_rate=target,
    )


def test_target_outside_the_interval_is_drift() -> None:
    """The operating point is demonstrably not the operating point.

    Measured: ``max_rolling_means`` transferred to long context realised a flag
    rate of 0.5433 [0.5050, 0.5833] against a declared 0.05 target — thirteen
    times the budget, with the target nowhere near the interval.
    """
    claim = assess_calibration(_flag_rate(0.5433, 0.5050, 0.5833), _operating_point())
    assert claim.status is CalibrationStatus.DRIFTED
    assert not claim.underpowered
    assert "no longer the operating point" in claim.detail


def test_drift_in_the_other_direction_is_still_drift() -> None:
    """Underspending is drift too, and it is the one that looks like calm.

    ``mean_pool`` on long context flagged nothing at all: 0.0000 [0.0000,
    0.0061]. A detector that stops flagging reads as clean traffic on every
    dashboard, which is why the test is two-sided.
    """
    claim = assess_calibration(_flag_rate(0.0, 0.0, 0.0061), _operating_point())
    assert claim.status is CalibrationStatus.DRIFTED
    assert "underspends" in claim.detail


def test_target_inside_the_interval_is_not_a_clean_bill() -> None:
    """``CALIBRATED`` means drift was not shown, which is weaker than held.

    Measured: ``last_token`` on long context realised 0.0650 [0.0483, 0.0850]
    against a 0.05 target. The point estimate is 1.30x the budget and the
    interval covers the target, so this sample cannot distinguish them.
    Reporting that as drift would report a point estimate whose interval covers
    the null; reporting it as calibration holding would hide a 30% overspend.
    """
    claim = assess_calibration(_flag_rate(0.0650, 0.0483, 0.0850), _operating_point())
    assert claim.status is CalibrationStatus.CALIBRATED
    assert claim.underpowered, (
        "drift was not shown at n=600, and the interval [0.048, 0.085] extends "
        "past the +/-25% band [0.0375, 0.0625] -- so a deviation worth acting "
        "on is not ruled out. Without this flag, CALIBRATED reads as "
        "reassurance."
    )
    assert claim.n_to_detect == 1441, (
        "power is measured against a deviation worth acting on -- a 25% "
        "departure from the 0.05 budget -- not against the observed gap"
    )
    assert "drift is not shown" in claim.detail
    assert "NOT ruled out" in claim.detail


def test_underpowered_is_a_stored_field_not_a_property() -> None:
    """The ledger serialises fields, so a property would not be sealed.

    The one caveat that stops ``CALIBRATED`` reading as reassurance has to
    survive into the record. A reader who must recompute it is a reader who
    will not.
    """
    claim = assess_calibration(_flag_rate(0.0650, 0.0483, 0.0850), _operating_point())
    payload = to_jsonable(claim)
    assert payload["underpowered"] is True
    assert payload["n_to_detect"] == 1441
    assert payload["status"] == "CALIBRATED"

    # And it must survive the round trip. A derived field is serialised for the
    # reader's benefit but rejected by the constructor, so a naive decoder
    # raises TypeError -- which is how this broke reading the ledger back. The
    # recomputed value has to agree with the written one, or the record and the
    # object it decodes to are two different claims.
    from controlplane.model.calibration import CalibrationClaim

    restored = from_jsonable(CalibrationClaim, payload)
    assert restored.underpowered == payload["underpowered"]
    assert restored.status is claim.status
    assert restored.n_to_detect == claim.n_to_detect


def test_a_claim_nobody_could_evaluate_is_unknown_not_calibrated() -> None:
    """Absence must not read as a pass (``DECISIONS.md`` 050)."""
    no_target = assess_calibration(
        _flag_rate(0.05, 0.03, 0.07), _operating_point(target=None)
    )
    assert no_target.status is CalibrationStatus.UNKNOWN
    assert not no_target.underpowered

    no_metric = assess_calibration(None, _operating_point())
    assert no_metric.status is CalibrationStatus.UNKNOWN


def test_a_well_powered_agreement_is_not_flagged() -> None:
    """A tight interval around the target is the case that needs no caveat."""
    claim = assess_calibration(_flag_rate(0.0501, 0.0480, 0.0520, n=20000), _operating_point())
    assert claim.status is CalibrationStatus.CALIBRATED
    assert not claim.underpowered, (
        "an n large enough to have detected drift, finding none, is the one "
        "case where CALIBRATED means what it sounds like"
    )


@pytest.mark.parametrize(
    "target,tolerance,expected",
    [
        (0.05, 0.25, 1441),
        (0.05, 0.10, 7987),
    ],
)
def test_n_to_detect_deviation(target: float, tolerance: float, expected: int) -> None:
    """Power against a declared effect, which does not diverge near agreement.

    The predecessor took the *observed* gap, so an estimate landing on its
    target reported an n of 18 million and flagged perfect agreement as
    underpowered. Detecting a tighter deviation legitimately costs more samples
    -- 10% needs 7987 against 25%'s 1441 -- and that is a property of the effect
    size, not of the estimate.
    """
    assert n_to_detect_deviation(target, tolerance) == expected


def test_every_issued_warrant_carries_a_calibration_claim() -> None:
    """No warrant may exist without one, issued or refused.

    Computed at issuance rather than bolted on by callers, because a claim that
    some code paths attach and others do not is a claim whose absence is
    indistinguishable from a pass.
    """
    import inspect

    from controlplane.validation import issuance

    source = inspect.getsource(issuance)
    assert "assess_calibration" in source, (
        "issuance does not compute a calibration claim; warrants would be "
        "sealed asserting a budget nobody checked"
    )
    assert "calibration=calibration" in source
