"""The enumerations every other record type is built from.

``SPEC.md`` §1.1. Two of these carry more weight than the rest:

* :class:`WarrantStatus` has five members but *three behaviours*
  (``CLAUDE.md`` invariant 2). ``UNVALIDATED`` is the modal state in production
  and must never collapse into ``VALID`` or ``REFUSED``.
* :class:`MetricKind` is not decorative. It is the type-level expression of
  "yield is exact, rate is estimated", and the renderer refuses to print an
  ``ESTIMATED`` value without an interval.

``IntEnum`` where the ordering is meaningful and gets compared (``severity >=
HIGH``, ``reversibility >= IRREVERSIBLE_WRITE``); plain ``Enum`` where an
ordering would be meaningless and inviting a comparison would be a bug.
"""

from __future__ import annotations

from enum import Enum, IntEnum

__all__ = [
    "AccessTier",
    "Action",
    "Category",
    "ConfidenceBand",
    "EnvelopeState",
    "MetricKind",
    "Reversibility",
    "Severity",
    "WarrantStatus",
]


class Severity(IntEnum):
    """How serious a finding is, ordered so policy can compare against a bar.

    Ordered because the policy rules compare (``severity >= HIGH``). The values
    are a detector's assessment of its own finding, not a warranted claim about
    the world.
    """

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AccessTier(IntEnum):
    """How deep into the model a detector is allowed to reach.

    Numbered so that a *lower* number means *more* access: T1 sees activations,
    T3 sees only text. The ladder exists because access is what an enterprise
    is deciding whether to pay for, and the tier ablation measures what that
    access buys.
    """

    T3_TEXT = 3
    T2_LOGPROBS = 2
    T1_ACTIVATIONS = 1


class WarrantStatus(Enum):
    """What is known about a detector's numbers on one input distribution.

    Five members, three behaviours (``CLAUDE.md`` invariant 2):

    * ``VALID`` — validated here, cleared the bar. Normal operation, full
      claimed bounds.
    * ``STALE`` and ``REVOKED`` — was valid, the envelope moved. ``STALE``
      widens the reported interval and triggers background revalidation;
      ``REVOKED`` refuses to certify and sends the request to the matrix.
    * ``REFUSED`` — validated here and *failed*. Out of service on this
      envelope until a human revalidates. No override exists (invariant 3).
    * ``UNVALIDATED`` — never tested here. The modal state in production.
      Conservative action, enqueue the cell, log. Collapsing this into
      ``REFUSED`` makes the system unusable on day one; collapsing it into
      ``VALID`` is the failure the whole product argues against.

    Not an ``IntEnum``: there is no ordering in which ``REFUSED`` is
    "worse than" ``UNVALIDATED``, and inviting that comparison is precisely how
    the three states get collapsed into two.
    """

    VALID = "VALID"
    STALE = "STALE"
    REVOKED = "REVOKED"
    REFUSED = "REFUSED"
    UNVALIDATED = "UNVALIDATED"

    @property
    def can_be_relied_upon(self) -> bool:
        """True only for ``VALID``.

        Deliberately not "not REFUSED". Every other state means the bounds are
        either unknown or known to be wrong, and a caller that wants to proceed
        anyway has to say which it is.
        """
        return self is WarrantStatus.VALID

    @property
    def was_ever_measured_here(self) -> bool:
        """True unless the cell was never tested on this envelope.

        The distinction ``UNVALIDATED`` exists to preserve: "we have no number"
        is a different claim from "we measured and it failed", and they lead to
        different operational responses — enqueue versus escalate to a human.
        """
        return self is not WarrantStatus.UNVALIDATED


class CalibrationStatus(Enum):
    """Whether a warrant's operating point still delivers its declared budget.

    Separate from :class:`WarrantStatus` because a warrant makes two claims and
    can hold one while losing the other. ``T1-last_token`` transferred to the
    long-context envelope with its ranking essentially intact (AUROC 0.826 ->
    0.813) while the frozen threshold went from flagging 4.2% of traffic to
    6.5%. Collapsing that into one status would have to call it either sound --
    hiding a budget claim that may no longer hold -- or refused, discarding a
    ranking that demonstrably survived.

    There is deliberately no state meaning "calibration verified". ``CALIBRATED``
    means drift was **not shown**, which at a small ``n`` is a much weaker
    statement, and :class:`CalibrationClaim` carries the ``n`` that would have
    been needed so the two cannot be confused.
    """

    #: Target flag rate lies outside the realised interval. The operating point
    #: is demonstrably not the operating point any more.
    DRIFTED = "DRIFTED"
    #: Target lies inside the realised interval, so drift is not shown. Check
    #: ``CalibrationClaim.underpowered`` before reading this as reassurance.
    CALIBRATED = "CALIBRATED"
    #: No target declared, or no interval to test against. Never a pass.
    UNKNOWN = "UNKNOWN"


class MetricKind(Enum):
    """Whether a metric is a count of reviewed items or an estimate over a pool.

    ``CLAUDE.md``, "the single most important distinction in this codebase".

    * ``EXACT`` — a count of reviewed, confirmed items. *"We surfaced 850 real
      errors."* No sampling, no interval, and free, because the flagged stratum
      is reviewed already.
    * ``ESTIMATED`` — a quantity requiring inference about items nobody
      reviewed. *"We caught 14% of errors."* Always carries an interval, and
      the interval always names its ``n``.

    Conflating them converts a free exact claim into an unbacked estimate and
    nobody notices. :class:`src.model.metrics.Metric` refuses to be constructed
    in violation of this, which is what ``test_yield_vs_rate`` asserts.
    """

    EXACT = "EXACT"
    ESTIMATED = "ESTIMATED"


class Reversibility(IntEnum):
    """The blast radius of a tool call, registered at tool-definition time.

    A **static property of the tool**, never an inference about the request
    (``SPEC.md`` §9.1). That is exactly what makes the action gate robust to an
    attacker who has defeated every detector: the gate's first two rules read
    this and the session flags, and consult no detector score at all.

    Ordered because the gate compares (``reversibility >= IRREVERSIBLE_WRITE``).
    """

    READ_ONLY = 0
    REVERSIBLE_WRITE = 1
    IRREVERSIBLE_WRITE = 2
    EXTERNAL_COMM = 3


class Category(Enum):
    """What kind of problem a finding describes.

    ``SPEC.md`` §1.2 types this as ``str``; a closed enumeration gives the same
    JSON while making a typo a crash rather than a category that silently
    matches no policy rule.

    **Categories overlap by design.** A fabricated detail about a named person
    emits both ``HALLUCINATION`` and ``PII``. Detectors never resolve the
    conflict — policy does.
    """

    PII = "PII"
    HALLUCINATION = "HALLUCINATION"
    INJECTION = "INJECTION"
    UNSAFE = "UNSAFE"
    COST = "COST"
    BIAS_SIGNAL = "BIAS_SIGNAL"


class Action(Enum):
    """What the policy decided to do about a request.

    ``ALLOW`` is a decision like any other and is certified like any other: the
    certificate says what was checked and at what measured bounds, so an allowed
    request carries the same evidence as a blocked one.
    """

    ALLOW = "ALLOW"
    REDACT = "REDACT"
    CONFIRM = "CONFIRM"
    ESCALATE = "ESCALATE"
    BLOCK = "BLOCK"


class ConfidenceBand(Enum):
    """Where a score sits relative to the operating point's measured bands.

    ``UNCERTAIN`` is the band around the threshold where the measured error
    rate is high enough that the policy escalates rather than deciding. Naming
    the band is what lets a policy rule say "escalate when the detector is near
    its own threshold" without inventing a second threshold.
    """

    CONFIDENT_NEGATIVE = "CONFIDENT_NEGATIVE"
    UNCERTAIN = "UNCERTAIN"
    CONFIDENT_POSITIVE = "CONFIDENT_POSITIVE"


class EnvelopeState(Enum):
    """Where live traffic sits relative to a warrant's reference distribution.

    The rungs of the revocation ladder (``SPEC.md`` §5.3), keyed to PSI bands
    from ``config.yaml``: inside, moderate drift, significant drift. A fourth
    state, ``INSUFFICIENT_DATA``, covers the window minimum — below it there is
    no verdict, because revoking on noise is how a drift monitor gets turned
    off by its operators.
    """

    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INSIDE = "INSIDE"
    MODERATE_SHIFT = "MODERATE_SHIFT"
    SIGNIFICANT_SHIFT = "SIGNIFICANT_SHIFT"
