"""Warrant issuance and refusal. ``SPEC.md`` §2.3.

One function decides, :func:`issue_or_refuse`, and it takes **no argument that
could relax the bar**. No ``force``, no ``allow_failed_controls``, no
``override_reason``. That absence is invariant 3, and it is deliberate that
adding one would require changing a signature — visible in a diff — rather than
passing a flag at a call site.

Even if someone did, it would not help. :class:`~src.model.warrant.Warrant`
refuses to be constructed with a status other than ``REFUSED`` when any control
failed, so the refusal is enforced twice: once by the policy here, once by the
record type. ``test_no_override`` asserts both, behaviourally and by grep.

Refusal criteria, all of them:

* any control failed;
* AUROC lower CI bound ≤ ``config.validation.min_auroc_lower_ci``;
* recall below the profile's declared minimum at this operating point;
* FPR on the hard-negative set above the declared maximum;
* ``n_test`` < ``config.validation.min_n_test``.

A refused warrant is **stored**, not discarded. A log recording only the
warrants that were granted would let a refusal be retried quietly until it
passed, and the append-only ledger is what makes that checkable by someone who
was not there.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime, timedelta
from typing import Optional, Sequence

from ..config import Config
from ..model import (
    AccessTier,
    ControlResult,
    DistributionEnvelope,
    OperatingPoint,
    Warrant,
    WarrantKey,
    WarrantMetrics,
    WarrantStatus,
    utc_now,
)

__all__ = ["RefusalReason", "issue_or_refuse"]

_LOG = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class RefusalReason:
    """One reason a warrant was refused, with the numbers that produced it.

    Carried separately from the joined ``status_reason`` string so the demo can
    render them as a list and so a reader can see *which* bar was missed and by
    how much, rather than being told the run failed.
    """

    criterion: str
    measured: str
    required: str

    def render(self) -> str:
        """One line, readable aloud."""
        return f"{self.criterion}: measured {self.measured}, required {self.required}"


def _control_refusals(controls: Sequence[ControlResult]) -> list[RefusalReason]:
    """Refusals arising from failed controls."""
    return [
        RefusalReason(
            criterion=f"control:{c.control}",
            measured=f"{c.measured:.4f} (margin {c.margin:+.4f})",
            required=c.expected,
        )
        for c in controls
        if not c.passed
    ]


def issue_or_refuse(
    config: Config,
    *,
    key: WarrantKey,
    detector_version: str,
    operating_point: OperatingPoint,
    metrics: WarrantMetrics,
    envelope: DistributionEnvelope,
    controls: Sequence[ControlResult],
    access_tier: AccessTier,
    n_test: int,
    base_rate: float,
    validation_run_id: str,
    min_recall: Optional[float] = None,
    max_fpr_hard_negatives: Optional[float] = None,
    kappa: Optional[float] = None,
    issued_at: Optional[datetime] = None,
) -> Warrant:
    """Issue a warrant, or refuse one, from measured evidence.

    Always returns a :class:`Warrant`. A refusal is a warrant with status
    ``REFUSED`` and a reason, not an exception and not ``None`` — because a
    refusal is a *finding about the detector on this envelope* that has to be
    stored, routed on, and read back later. Raising would make refusals the one
    outcome the ledger never sees.

    There is deliberately no parameter that relaxes any criterion. See the
    module docstring.

    Args:
        config: Resolved config, supplying the thresholds and the warrant TTL.
        key: The three-part warrant key.
        detector_version: Semver plus weights hash.
        operating_point: The threshold, which must belong to this detector and
            must not have been selected on test.
        metrics: Measured bounds. Precision and recall are both required by the
            type.
        envelope: The reference distribution these numbers were measured inside.
        controls: All five control results.
        access_tier: What depth of access the detector needed.
        n_test: Items in the test scoring.
        base_rate: Positive-class prevalence in the test split.
        validation_run_id: The run that produced this.
        min_recall: The consuming profile's declared minimum, when one applies.
        max_fpr_hard_negatives: Declared maximum FPR on hard negatives.
        kappa: Inter-rater agreement where labels are human.
        issued_at: Issue time; defaults to now.

    Returns:
        A ``VALID`` warrant if every criterion is met, else a ``REFUSED`` one
        whose ``status_reason`` names each criterion, the measured value and the
        bar it missed.
    """
    issued = issued_at or utc_now()
    expires = issued + timedelta(hours=config.validation.warrant_ttl_hours)

    reasons = _control_refusals(controls)

    if len(controls) != len(config.validation.controls):
        reasons.append(
            RefusalReason(
                criterion="control_suite",
                measured=f"{len(controls)} control(s) reported",
                required=f"all {len(config.validation.controls)} controls run",
            )
        )

    if n_test < config.validation.min_n_test:
        reasons.append(
            RefusalReason(
                criterion="n_test",
                measured=str(n_test),
                required=f">= {config.validation.min_n_test}",
            )
        )

    auroc_lower = metrics.auroc.ci_low
    if auroc_lower is None or auroc_lower <= config.validation.min_auroc_lower_ci:
        reasons.append(
            RefusalReason(
                criterion="auroc_lower_ci",
                measured="none" if auroc_lower is None else f"{auroc_lower:.4f}",
                required=f"> {config.validation.min_auroc_lower_ci}",
            )
        )

    if min_recall is not None:
        # Compared against the interval's lower bound, not the point estimate.
        # A profile declaring "at least 10% recall" is asking for a guarantee,
        # and a point estimate of 0.11 with a lower bound of 0.06 does not
        # supply one. This is the stricter reading and it is the one the
        # profile suspension in Beat 4 depends on.
        recall_lower = metrics.recall.ci_low
        if recall_lower is None or recall_lower < min_recall:
            reasons.append(
                RefusalReason(
                    criterion="recall_lower_ci",
                    measured=(
                        "none" if recall_lower is None else f"{recall_lower:.4f}"
                    ),
                    required=f">= {min_recall} (profile minimum)",
                )
            )

    if max_fpr_hard_negatives is not None:
        fpr = metrics.fpr_hard_negatives
        if fpr is None:
            reasons.append(
                RefusalReason(
                    criterion="fpr_hard_negatives",
                    measured="not measured",
                    required=f"<= {max_fpr_hard_negatives}",
                )
            )
        else:
            # Upper bound, by the same argument reversed: a declared maximum is
            # a promise, and the promise has to hold at the pessimistic end.
            upper = fpr.ci_high if fpr.ci_high is not None else fpr.value
            if upper > max_fpr_hard_negatives:
                reasons.append(
                    RefusalReason(
                        criterion="fpr_hard_negatives_upper_ci",
                        measured=f"{upper:.4f}",
                        required=f"<= {max_fpr_hard_negatives}",
                    )
                )

    warrant_id = Warrant.compute_id(key, detector_version, validation_run_id)
    status = WarrantStatus.VALID if not reasons else WarrantStatus.REFUSED
    status_reason = None
    if reasons:
        status_reason = "; ".join(r.render() for r in reasons)
        _LOG.warning(
            "REFUSED %s on %s: %s", key.detector_id, key.eval_set_id, status_reason
        )
    else:
        _LOG.info(
            "issued %s for %s on %s (AUROC lower CI %.4f, n_test %d)",
            warrant_id,
            key.detector_id,
            key.eval_set_id,
            auroc_lower,
            n_test,
        )

    return Warrant(
        warrant_id=warrant_id,
        detector_id=key.detector_id,
        detector_version=detector_version,
        operating_point=operating_point,
        eval_set_id=key.eval_set_id,
        validation_run_id=validation_run_id,
        issued_at=issued,
        expires_at=expires,
        metrics=metrics,
        n_test=n_test,
        base_rate=base_rate,
        envelope=envelope,
        controls=tuple(controls),
        access_tier=access_tier,
        status=status,
        kappa=kappa,
        status_reason=status_reason,
    )
