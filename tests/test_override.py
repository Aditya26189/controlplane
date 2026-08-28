"""Human overrides on escalated items. ``SPEC.md`` §6.5, ``DECISIONS.md`` 093.

The two fields the estimator cannot reconstruct — ``stratum`` and
``selection_probability`` — are the reason this file exists. A record missing
either is unrecoverable after the fact, so the tests that matter most are the
ones asserting a **hard failure** rather than a warning.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.model import utc_now
from src.model.override import (
    FLAGGED,
    UNFLAGGED,
    HumanDecision,
    OverrideDirection,
    OverrideError,
    OverrideRecord,
)
from src.store import Ledger, RecordKind


def record(**overrides) -> OverrideRecord:
    fields = dict(
        override_id="O-0001",
        certificate_id="C-0001",
        item_ref="item-0001",
        detector_id="probe-qwen2.5-7b-instruct-T1-last_token",
        score=0.91,
        threshold_in_force=0.8909,
        stratum=FLAGGED,
        selection_probability=1.0,
        envelope_id="sha256:1e2dee9d8fa91cba",
        human_decision=HumanDecision.OVERRIDDEN,
        direction=OverrideDirection.ESCALATE_TO_ALLOW,
        timestamp=utc_now(),
    )
    fields.update(overrides)
    return OverrideRecord(**fields)


# --------------------------------------------------------------------------- #
# The hard failure D.3 requires
# --------------------------------------------------------------------------- #


def test_a_record_without_a_stratum_cannot_be_built() -> None:
    """Not a validation warning. A record written without it is unrecoverable:
    the stratum depends on the threshold and envelope in force at capture, and
    both move."""
    with pytest.raises(OverrideError, match="stratum"):
        record(stratum="")
    with pytest.raises(OverrideError, match="stratum"):
        record(stratum="above_threshold")


def test_a_record_without_a_selection_probability_cannot_be_built() -> None:
    """Overrides exist only on escalated items, so the pool is conditioned on
    the detector having fired. Unweighted, that biases recall upward."""
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(OverrideError, match="selection_probability"):
            record(selection_probability=bad)


def test_the_refusal_explains_the_direction_of_the_bias() -> None:
    """A reader who hits this needs to know why it is not a formality."""
    with pytest.raises(OverrideError) as caught:
        record(selection_probability=0.0)
    assert "biased upward" in str(caught.value)


def test_a_record_missing_its_certificate_cannot_be_built() -> None:
    """A label with no record of what produced it cannot be attributed to an
    operating point."""
    with pytest.raises(OverrideError, match="certificate_id"):
        record(certificate_id="")


def test_an_unweighted_record_cannot_reach_the_ledger(tmp_path: Path) -> None:
    """The check is at the point of writing, not the point of reading.

    There is no path from a malformed record to a stored one, because the record
    validates on construction and the ledger only accepts records.
    """
    store = Ledger(tmp_path / "o.db", retention_days=400)
    try:
        with pytest.raises(OverrideError):
            store.append_override(record(stratum="made-up"))
        assert store.count(RecordKind.OVERRIDE) == 0
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# Consistency between the decision and its direction
# --------------------------------------------------------------------------- #


def test_upheld_and_overridden_must_agree_with_the_direction() -> None:
    """Letting them drift apart makes the false-negative count unreadable."""
    with pytest.raises(OverrideError, match="disagree"):
        record(human_decision=HumanDecision.UPHELD, direction=OverrideDirection.ESCALATE_TO_ALLOW)
    with pytest.raises(OverrideError, match="disagree"):
        record(human_decision=HumanDecision.OVERRIDDEN, direction=OverrideDirection.NONE)
    # The two consistent forms build.
    record(human_decision=HumanDecision.UPHELD, direction=OverrideDirection.NONE)
    record(
        human_decision=HumanDecision.OVERRIDDEN,
        direction=OverrideDirection.ALLOW_TO_ESCALATE,
    )


def test_the_two_error_kinds_are_distinguishable() -> None:
    """A false positive costs one wasted review; a false negative costs a
    customer acting on a wrong answer. They must not be one counter."""
    fp = record(direction=OverrideDirection.ESCALATE_TO_ALLOW)
    fn = record(direction=OverrideDirection.ALLOW_TO_ESCALATE, stratum=UNFLAGGED,
                selection_probability=0.005)
    assert fp.is_false_positive and not fp.is_false_negative
    assert fn.is_false_negative and not fn.is_false_positive


# --------------------------------------------------------------------------- #
# Weighting
# --------------------------------------------------------------------------- #


def test_an_unflagged_item_carries_the_weight_of_the_ones_nobody_saw() -> None:
    """An item drawn at 1 in 200 stands for the 199 like it that went unreviewed."""
    assert record(selection_probability=1.0).weight == pytest.approx(1.0)
    assert record(
        stratum=UNFLAGGED, selection_probability=0.005
    ).weight == pytest.approx(200.0)


def test_the_summary_reports_weighted_counts_beside_raw_ones(tmp_path: Path) -> None:
    """A raw count from a reviewed sample is not an estimate of anything.

    The sample is conditioned on the detector having fired, so quoting the raw
    false-negative count as a rate is the upward-biased number this schema
    exists to prevent.
    """
    store = Ledger(tmp_path / "o.db", retention_days=400)
    try:
        store.append_override(
            record(override_id="O-1", stratum=FLAGGED, selection_probability=1.0,
                   direction=OverrideDirection.ESCALATE_TO_ALLOW)
        )
        store.append_override(
            record(override_id="O-2", certificate_id="C-2", stratum=UNFLAGGED,
                   selection_probability=0.005,
                   direction=OverrideDirection.ALLOW_TO_ESCALATE)
        )
        store.append_override(
            record(override_id="O-3", certificate_id="C-3",
                   human_decision=HumanDecision.UPHELD,
                   direction=OverrideDirection.NONE)
        )
        summary = store.override_summary()
        assert summary["n"] == 3
        assert summary["upheld"] == 1 and summary["overridden"] == 2
        assert summary["escalate_to_allow"] == 1
        assert summary["allow_to_escalate"] == 1
        # One unflagged false negative drawn at 1-in-200 stands for 200.
        assert summary["weighted_allow_to_escalate"] == pytest.approx(200.0)
        assert summary["weighted_escalate_to_allow"] == pytest.approx(1.0)
        assert summary["by_stratum"] == {FLAGGED: 2, UNFLAGGED: 1}
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


def test_an_override_is_chained_and_queryable_by_certificate(tmp_path: Path) -> None:
    store = Ledger(tmp_path / "o.db", retention_days=400)
    try:
        store.append_override(record())
        store.append_override(record(override_id="O-2", certificate_id="C-0002"))
        assert store.verify_chain().ok
        found = store.overrides_for_certificate("C-0001")
        assert len(found) == 1
        assert found[0]["override_id"] == "O-0001"
        assert found[0]["stratum"] == FLAGGED
        assert found[0]["selection_probability"] == 1.0
    finally:
        store.close()


def test_a_naive_timestamp_is_refused() -> None:
    with pytest.raises(OverrideError, match="UTC offset"):
        record(timestamp=datetime(2026, 8, 28, 12, 0, 0))


def test_the_payload_carries_every_field_the_estimator_needs() -> None:
    """Phase 6 consumes this without a migration, so the payload is the contract."""
    payload = record().to_payload()
    for field in (
        "certificate_id", "item_ref", "detector_id", "score",
        "threshold_in_force", "stratum", "selection_probability",
        "envelope_id", "human_decision", "direction", "timestamp",
    ):
        assert field in payload, field
