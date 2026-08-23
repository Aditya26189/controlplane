"""Warrants: what a detector's score is worth, on one distribution, until when.

A warrant is not a score and not a quality badge. It is a time-bounded,
evidence-backed statement about what a score is worth *right now, on this input
distribution* — the measured bounds, the ``n`` behind them, the controls that
passed, the envelope they were measured inside, and an expiry.

**The key is ``(detector_id, operating_point_id, eval_set_id)``** — ``CLAUDE.md``
invariant 1, and the whole matrix depends on it. Each element can independently
invalidate the claim: a different detector obviously; a different threshold,
because recall at τ₁ says nothing about recall at τ₂; and a different input
distribution, because an envelope violation is a property of the *traffic*, so
long-context inputs stop T1, T2 and T3 from being characterised simultaneously.
That last one is why "drift fired, downgrade a tier" is wrong — it replaces one
unwarranted claim with another.

Invariant 3 — refusal has no override — is enforced here structurally: a
:class:`Warrant` with a failed control cannot be constructed with a status other
than ``REFUSED``. There is no flag, no argument and no environment variable that
relaxes it, because the check is in ``__post_init__`` and every construction
path, including reading a record back from the store, goes through it.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import datetime, timedelta
from typing import Optional

from .enums import AccessTier, WarrantStatus
from .findings import DistributionEnvelope, OperatingPoint
from .metrics import Metric, WarrantMetrics
from .serde import content_hash

__all__ = [
    "ControlResult",
    "Warrant",
    "WarrantError",
    "WarrantKey",
]


class WarrantError(ValueError):
    """Raised when a warrant would claim more than its evidence supports."""


@dataclasses.dataclass(frozen=True)
class WarrantKey:
    """The three-part identity of a warrant. ``CLAUDE.md`` invariant 1.

    A separate type rather than three loose strings, because the failure this
    prevents is passing two of the three and getting a plausible answer about
    the wrong distribution.
    """

    detector_id: str
    operating_point_id: str
    eval_set_id: str

    def __post_init__(self) -> None:
        missing = [
            name
            for name in ("detector_id", "operating_point_id", "eval_set_id")
            if not getattr(self, name)
        ]
        if missing:
            raise WarrantError(
                f"a warrant key needs all three elements; missing {missing}. "
                "Keying by detector alone is invariant 1's failure: an envelope "
                "violation is a property of the input distribution, so it "
                "invalidates every detector measured on that distribution at once."
            )

    def as_string(self) -> str:
        """Stable single-string form, used as a matrix cell address."""
        return f"{self.detector_id}|{self.operating_point_id}|{self.eval_set_id}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.as_string()


@dataclasses.dataclass(frozen=True)
class ControlResult:
    """One control's outcome, with the margin it cleared or missed by.

    The margin is required. "Control passed" is an assertion; "label shuffle
    scored 0.497, inside [0.45, 0.55] with 0.047 of margin" is evidence, and the
    difference is what a judge is actually asking about when they ask whether
    the controls mean anything.

    Args:
        control: One of ``config.validation.controls``.
        passed: Whether the pass condition was met.
        measured: The measured value the condition was evaluated against.
        expected: The pass condition, in words, e.g. ``"AUROC in [0.45, 0.55]"``.
        margin: Signed distance to the nearest edge of the pass condition.
            Positive means inside with room to spare; negative means outside by
            that much.
        detail: Anything a reader needs to interpret the two numbers.
    """

    control: str
    passed: bool
    measured: float
    expected: str
    margin: float
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.control:
            raise WarrantError("a control result must name its control")
        if not self.expected:
            raise WarrantError(
                f"control {self.control}: the pass condition must be stated. "
                "'Passed' without the condition it passed is not evidence."
            )
        if not math.isfinite(self.measured) or not math.isfinite(self.margin):
            raise WarrantError(
                f"control {self.control}: measured and margin must be finite"
            )
        if self.passed and self.margin < 0:
            raise WarrantError(
                f"control {self.control}: reported as passed with a negative "
                f"margin ({self.margin}). One of the two is wrong, and a control "
                "that reports a pass it did not achieve is worse than no control."
            )
        if not self.passed and self.margin > 0:
            raise WarrantError(
                f"control {self.control}: reported as failed with a positive "
                f"margin ({self.margin})"
            )


@dataclasses.dataclass(frozen=True)
class Warrant:
    """What a detector's score is worth on one envelope, until it expires.

    ``SPEC.md`` §1.3.

    Args:
        warrant_id: Stable id; content-derived via :meth:`compute_id`.
        detector_id: First element of the key.
        detector_version: Semver plus weights hash. A silent model update
            produces a different version and therefore cannot inherit this
            warrant (``SPEC.md`` §5.4).
        operating_point: Second element of the key, carried whole so the
            threshold and the objective that chose it travel with the claim.
        eval_set_id: Third element of the key.
        validation_run_id: The run that produced these numbers.
        issued_at: When the run completed.
        expires_at: After which the claim is stale by age alone, regardless of
            drift. A measurement with no expiry is a permanent claim about a
            world that changes.
        metrics: The measured bounds. Precision and recall are both required.
        n_test: How many items the test scoring used.
        base_rate: Prevalence of the positive class in the test set. Reported
            beside AUROC always: at an 85% correct rate, a probe that always
            predicts "correct" scores 0.85 accuracy and 0.5 AUROC, and the base
            rate is what stops that reading as signal.
        envelope: The reference distribution these numbers were measured inside.
        controls: All five control results.
        access_tier: The depth of access this detector required.
        kappa: Inter-rater agreement where labels are human, else None. A ±5pp
            interval is meaningless at κ = 0.5, so the warrant carries its own
            label-quality statistic (``SPEC.md`` §6.5).
        status: Current state. See :class:`WarrantStatus` — five members, three
            behaviours.
        status_reason: Why, in words. Required for anything other than ``VALID``:
            a refusal a reader cannot interpret is indistinguishable from a bug.
        superseded_by: The warrant that replaced this one, when revalidation has
            happened.

    Raises:
        WarrantError: If the warrant would claim more than its evidence supports.
            In particular, a failed control forces ``REFUSED`` and no argument
            relaxes that (invariant 3).
    """

    warrant_id: str
    detector_id: str
    detector_version: str
    operating_point: OperatingPoint
    eval_set_id: str
    validation_run_id: str
    issued_at: datetime
    expires_at: datetime
    metrics: WarrantMetrics
    n_test: int
    base_rate: float
    envelope: DistributionEnvelope
    controls: tuple[ControlResult, ...]
    access_tier: AccessTier
    status: WarrantStatus
    kappa: Optional[float] = None
    status_reason: Optional[str] = None
    superseded_by: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.warrant_id or not self.validation_run_id:
            raise WarrantError("warrant_id and validation_run_id are required")
        if self.operating_point.detector_id != self.detector_id:
            raise WarrantError(
                f"operating point belongs to {self.operating_point.detector_id!r} "
                f"but the warrant is for {self.detector_id!r}. A threshold "
                "measured on one detector says nothing about another."
            )
        if self.envelope.eval_set_id != self.eval_set_id:
            raise WarrantError(
                f"envelope describes {self.envelope.eval_set_id!r} but the warrant "
                f"is keyed to {self.eval_set_id!r}. The envelope IS the key's "
                "third element (invariant 1); a mismatch means the claim and the "
                "distribution it was measured on have come apart."
            )
        if self.expires_at <= self.issued_at:
            raise WarrantError(
                f"expires_at ({self.expires_at.isoformat()}) must be after "
                f"issued_at ({self.issued_at.isoformat()})"
            )
        if self.n_test < 0:
            raise WarrantError(f"n_test must be non-negative, got {self.n_test}")
        if not 0.0 <= self.base_rate <= 1.0:
            raise WarrantError(f"base_rate must be in [0, 1], got {self.base_rate}")
        if self.kappa is not None and not -1.0 <= self.kappa <= 1.0:
            raise WarrantError(f"kappa must be in [-1, 1], got {self.kappa}")

        # -- invariant 3: refusal has no override --------------------------- #
        failed = self.failed_controls()
        if failed and self.status is not WarrantStatus.REFUSED:
            raise WarrantError(
                f"control(s) {[c.control for c in failed]} failed, so this warrant "
                f"cannot hold status {self.status.value}. A failed control refuses "
                "the warrant — CLAUDE.md invariant 3, no flag, no environment "
                "variable, no admin bypass. If the control is wrong, fix the "
                "control and re-run; do not route around it."
            )
        if self.status is not WarrantStatus.VALID and not self.status_reason:
            raise WarrantError(
                f"status {self.status.value} requires a status_reason. A refusal a "
                "reader cannot interpret is indistinguishable from a bug, and the "
                "reason is what the certificate quotes."
            )
        if self.status is WarrantStatus.UNVALIDATED:
            raise WarrantError(
                "UNVALIDATED is the absence of a warrant, not a warrant with a "
                "status. It is represented by there being no record in the matrix "
                "cell, so that code cannot read bounds off it by accident "
                "(CLAUDE.md invariant 2)."
            )

    # -- identity ----------------------------------------------------------- #

    @property
    def key(self) -> WarrantKey:
        """The three-part key this warrant is filed under."""
        return WarrantKey(
            detector_id=self.detector_id,
            operating_point_id=self.operating_point.operating_point_id,
            eval_set_id=self.eval_set_id,
        )

    @staticmethod
    def compute_id(
        key: WarrantKey, detector_version: str, validation_run_id: str
    ) -> str:
        """Derive a warrant id from what makes the warrant unique.

        Content-derived rather than random so that two runs of the same
        validation on the same code produce the same id, which is what makes
        determinism checkable end to end.

        Args:
            key: The three-part warrant key.
            detector_version: Semver plus weights hash.
            validation_run_id: The run that produced the numbers.

        Returns:
            ``"W-"`` followed by 16 hex characters.
        """
        digest = content_hash(
            {
                "detector_id": key.detector_id,
                "operating_point_id": key.operating_point_id,
                "eval_set_id": key.eval_set_id,
                "detector_version": detector_version,
                "validation_run_id": validation_run_id,
            }
        )
        return f"W-{digest[:16]}"

    # -- state -------------------------------------------------------------- #

    def failed_controls(self) -> tuple[ControlResult, ...]:
        """Controls that did not pass. Non-empty forces ``REFUSED``."""
        return tuple(c for c in self.controls if not c.passed)

    def is_expired(self, now: datetime) -> bool:
        """Whether the claim has aged out, independently of drift.

        Age and drift are separate reasons to stop relying on a number, and a
        warrant can be inside its envelope and still too old to quote.
        """
        return now >= self.expires_at

    def age(self, now: datetime) -> timedelta:
        """How long ago the numbers were measured. Rendered in the demo banner."""
        return now - self.issued_at

    def can_be_relied_upon(self, now: datetime) -> bool:
        """Whether this warrant currently backs a claim.

        ``VALID`` and unexpired, and nothing else. Deliberately not
        "not REFUSED": ``STALE``, ``REVOKED`` and an expired ``VALID`` all mean
        the bounds are unknown or known-wrong, and a caller that wants to
        proceed anyway must say which case it is handling.
        """
        return self.status.can_be_relied_upon and not self.is_expired(now)

    def with_status(self, status: WarrantStatus, reason: str) -> "Warrant":
        """Return a copy in a new state, carrying the reason with it.

        Used by the revocation ladder. Returns a new record rather than mutating,
        because the store is append-only and a status change is a new fact about
        the world, not an edit to an old one.

        Raises:
            WarrantError: If the transition would promote a warrant to ``VALID``
                despite a failed control, or drop the reason.
        """
        if not reason:
            raise WarrantError("a status change must state its reason")
        return dataclasses.replace(self, status=status, status_reason=reason)

    # -- claims ------------------------------------------------------------- #

    def claimed_bounds(self) -> dict[str, object]:
        """The bounded, falsifiable assertion this warrant licenses.

        This is what makes liability bounded (``SPEC.md`` §1.4): the system
        asserts *"checked at measured recall 0.14 [0.10, 0.19] on envelope E,
        n=600"*, not *"this is safe"*. Precision travels with recall in the same
        structure, so a consumer cannot render one without the other
        (invariant 5).

        Returns:
            A plain mapping, ready to be embedded in a certificate.
        """
        bounds: dict[str, object] = {
            "warrant_id": self.warrant_id,
            "detector_id": self.detector_id,
            "detector_version": self.detector_version,
            "operating_point_id": self.operating_point.operating_point_id,
            "envelope_id": self.envelope.envelope_id,
            "eval_set_id": self.eval_set_id,
            "status": self.status.value,
            "n_test": self.n_test,
            "base_rate": self.base_rate,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "kappa": self.kappa,
        }
        for metric in self.metrics.all_metrics():
            bounds[metric.name] = _metric_claim(metric)
        return bounds


def _metric_claim(metric: Metric) -> dict[str, object]:
    """Render one metric into a certificate's claimed bounds.

    Keeps ``kind`` in the output so a reader can tell an exact count from an
    estimate without consulting the schema — which is the distinction the whole
    project turns on, and the one most easily lost in a rendering.
    """
    claim: dict[str, object] = {
        "value": metric.value,
        "kind": metric.kind.value,
        "n": metric.n,
        "unit": metric.unit,
    }
    if metric.has_interval:
        claim["ci_low"] = metric.ci_low
        claim["ci_high"] = metric.ci_high
        claim["ci_level"] = metric.ci_level
        claim["estimator"] = metric.estimator
    return claim
