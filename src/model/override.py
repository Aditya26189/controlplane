"""Human overrides on escalated items. ``SPEC.md`` §6.5, Phase 8 D.3.

The minimum viable feedback loop: capture what a reviewer decided about an item
the system escalated, store it against the certificate that caused the
escalation, and expose count and direction.

**This is not retraining.** Nothing here adjusts a detector. It is the
label-capture path Phase 6's stratified estimator needs anyway, built one phase
early because it fits the brief's named solutioning area, and it is deliberately
shaped so that Phase 6 can consume it without a migration.

## Why stratum and selection probability are required fields

Overrides exist **only on escalated items**. The label pool is therefore
conditioned on the detector having scored above threshold, which is the
definition of a biased sample. Fed to a stratified estimator unweighted, it
biases recall **upward** — the flagged stratum is enriched for true positives,
so measuring recall on reviewed items alone measures recall among the items the
detector already liked.

That is the exact direction the product exists to prevent, and it would arrive
as a number that looks better than the truth with nothing in the record to show
why.

Each record therefore carries the stratum it was drawn from and the probability
with which it was drawn, so the estimator can weight by `1/selection_probability`
and recover an unbiased estimate.

## Why they are stored rather than derived

The obvious alternative is to reconstruct the stratum at read time from the
score and the threshold. That is wrong and it fails silently.

The stratum an item belonged to depends on **the threshold and envelope in force
when it was captured**, and both move — a threshold is re-selected on every
revalidation (``DECISIONS.md`` 082 measured all three moving), and an envelope is
re-drawn on every renewal. A record read six months later would be reconstructed
against a threshold that did not exist when it was written, silently
reassigning items between strata and reweighting the whole estimate.

The failure has no error path: the reconstruction always succeeds and always
produces a plausible stratum. So the fields are captured at write time, and a
record missing them is **rejected**, not repaired. Records written without them
are unrecoverable, which is why this is a hard failure rather than a warning.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Optional

from .enums import Action

__all__ = [
    "FLAGGED",
    "OverrideDirection",
    "OverrideError",
    "OverrideRecord",
    "UNFLAGGED",
    "HumanDecision",
]

#: The two strata of ``SPEC.md`` §6.1. Named constants because a stratum is
#: compared for equality across the estimator and a typo would create a third.
FLAGGED = "flagged"
UNFLAGGED = "unflagged"
STRATA = (FLAGGED, UNFLAGGED)


class OverrideError(ValueError):
    """A record was written that the estimator could not use."""


class HumanDecision:
    """What the reviewer concluded about the system's action."""

    UPHELD = "upheld"
    OVERRIDDEN = "overridden"
    ALL = (UPHELD, OVERRIDDEN)


class OverrideDirection:
    """Which way an override went.

    Recorded separately from :class:`HumanDecision` because "the reviewer
    disagreed" and "the reviewer disagreed in the direction that means we missed
    something" are different facts, and only the second is a false negative.
    """

    #: The system escalated; the reviewer said it was fine. A false positive.
    ESCALATE_TO_ALLOW = "escalate_to_allow"
    #: The system allowed; the reviewer said it should have been caught. A false
    #: negative, and the expensive kind.
    ALLOW_TO_ESCALATE = "allow_to_escalate"
    #: The reviewer agreed with the system.
    NONE = "none"
    ALL = (ESCALATE_TO_ALLOW, ALLOW_TO_ESCALATE, NONE)


@dataclasses.dataclass(frozen=True)
class OverrideRecord:
    """One human judgement on one certified decision.

    Args:
        override_id: Unique id.
        certificate_id: The certificate whose decision this judges. The link is
            required — an override with no certificate is a label with no record
            of what produced it, and cannot be attributed to an operating point.
        item_ref: Opaque reference to the reviewed item. Not the content: this
            store is queried by session under DPDP Rule 6 and is not a place to
            accumulate message text.
        detector_id: Which detector's decision is being judged.
        score: The score it gave.
        threshold_in_force: The threshold **at capture time**. Stored because it
            moves on every revalidation.
        stratum: ``flagged`` or ``unflagged``, as of capture.
        selection_probability: Probability with which this item was drawn for
            review. The estimator weights by its inverse; without it a reviewed
            sample biases recall upward.
        envelope_id: The envelope in force at capture.
        human_decision: ``upheld`` or ``overridden``.
        direction: See :class:`OverrideDirection`.
        timestamp: When, tz-aware.
        reviewer_ref: Opaque reviewer reference, for inter-rater agreement.
            Never a name.
        notes: Free text from the reviewer.

    Raises:
        OverrideError: If any field the estimator needs is absent or invalid.
    """

    override_id: str
    certificate_id: str
    item_ref: str
    detector_id: str
    score: float
    threshold_in_force: float
    stratum: str
    selection_probability: float
    envelope_id: str
    human_decision: str
    direction: str
    timestamp: datetime
    reviewer_ref: Optional[str] = None
    notes: str = ""

    def __post_init__(self) -> None:
        for name in (
            "override_id",
            "certificate_id",
            "item_ref",
            "detector_id",
            "envelope_id",
        ):
            if not getattr(self, name):
                raise OverrideError(
                    f"{name} is required on an override record. A label that "
                    "cannot be traced to the decision that produced it cannot "
                    "be attributed to an operating point."
                )

        # The two fields the estimator cannot reconstruct. Checked first and
        # loudest, because a record missing them is not repairable later.
        if self.stratum not in STRATA:
            raise OverrideError(
                f"stratum must be one of {list(STRATA)}, got {self.stratum!r}. "
                "It is captured at write time, never derived at read time: the "
                "stratum depends on the threshold and envelope in force when "
                "the item was reviewed, and both move. Reconstructing it later "
                "always succeeds and is silently wrong."
            )
        if not 0.0 < self.selection_probability <= 1.0:
            raise OverrideError(
                "selection_probability must be in (0, 1], got "
                f"{self.selection_probability!r}. Overrides exist only on "
                "escalated items, so the label pool is conditioned on the "
                "detector having fired. Without the draw probability the "
                "estimator cannot weight by its inverse, and recall comes out "
                "biased upward — the direction this product exists to prevent."
            )

        if self.human_decision not in HumanDecision.ALL:
            raise OverrideError(
                f"human_decision must be one of {list(HumanDecision.ALL)}, got "
                f"{self.human_decision!r}"
            )
        if self.direction not in OverrideDirection.ALL:
            raise OverrideError(
                f"direction must be one of {list(OverrideDirection.ALL)}, got "
                f"{self.direction!r}"
            )
        if (self.human_decision == HumanDecision.UPHELD) != (
            self.direction == OverrideDirection.NONE
        ):
            raise OverrideError(
                f"human_decision {self.human_decision!r} and direction "
                f"{self.direction!r} disagree. An upheld decision has no "
                "direction and an overridden one always has a direction; "
                "letting them drift apart makes the false-negative count "
                "unreadable."
            )
        if self.timestamp.tzinfo is None:
            raise OverrideError(
                "override timestamps carry an explicit UTC offset; a naive "
                "timestamp means a different instant depending on where it is read"
            )
        if not 0.0 <= self.score <= 1.0:
            raise OverrideError(f"score must be in [0, 1], got {self.score}")

    @property
    def is_false_negative(self) -> bool:
        """The expensive kind: the system allowed something it should have caught."""
        return self.direction == OverrideDirection.ALLOW_TO_ESCALATE

    @property
    def is_false_positive(self) -> bool:
        """The cheap kind: one wasted review."""
        return self.direction == OverrideDirection.ESCALATE_TO_ALLOW

    @property
    def weight(self) -> float:
        """Inverse selection probability — the item's weight in the estimator.

        A flagged item reviewed with certainty weighs 1. An unflagged item drawn
        at 1 in 200 weighs 200, because it stands for the 199 like it that
        nobody looked at.
        """
        return 1.0 / self.selection_probability

    def to_payload(self) -> dict:
        return {
            "override_id": self.override_id,
            "certificate_id": self.certificate_id,
            "item_ref": self.item_ref,
            "detector_id": self.detector_id,
            "score": self.score,
            "threshold_in_force": self.threshold_in_force,
            "stratum": self.stratum,
            "selection_probability": self.selection_probability,
            "envelope_id": self.envelope_id,
            "human_decision": self.human_decision,
            "direction": self.direction,
            "timestamp": self.timestamp.isoformat(),
            "reviewer_ref": self.reviewer_ref,
            "notes": self.notes,
        }
