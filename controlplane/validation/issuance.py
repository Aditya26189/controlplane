"""Warrant issuance and refusal. ``SPEC.md`` §2.3.

One function decides, :func:`issue_or_refuse`, and it takes **no argument that
could relax the bar**. No ``force``, no ``allow_failed_controls``, no
``override_reason``. That absence is invariant 3, and it is deliberate that
adding one would require changing a signature — visible in a diff — rather than
passing a flag at a call site.

Even if someone did, it would not help. :class:`~controlplane.model.warrant.Warrant`
refuses to be constructed with a status other than ``REFUSED`` when any control
failed, so the refusal is enforced twice: once by the policy here, once by the
record type. ``test_no_override`` asserts both, behaviourally and by grep.

Refusal criteria, all of them:

* any control failed;
* AUROC lower CI bound ≤ ``config.validation.min_auroc_lower_ci``;
* **lift lower CI bound ≤ 1.0** — the detector is not demonstrably better than
  random sampling at the same budget (``DECISIONS.md`` 043);
* recall below the profile's declared minimum at this operating point;
* FPR on the hard-negative set above the declared maximum;
* ``n_test`` < ``config.validation.min_n_test``.

The lift criterion was **found by measurement, not designed in**. Phase 4
produced a warrant issued ``VALID`` on recall ``0.034 [0.000, 0.077]``: its
AUROC lower bound cleared the bar, because AUROC is a *ranking-quality* bar
while the product's claim is about *usefulness at a budget*, and those come
apart exactly there. See ``DECISIONS.md`` 043 for the derivation, and for why
profile suspension is kept alongside this rather than replaced by it.

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
from .calibration import assess_calibration
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

__all__ = ["MIN_LIFT_LOWER_BOUND", "RefusalReason", "issue_or_refuse"]

#: A warrant must be demonstrably better than random sampling at the same
#: budget. Not a tunable: 1.0 is the definition of "no better than chance at
#: this cost" rather than a threshold anyone chose, which is why it is a
#: constant here and not a value in config.yaml. Wanting a *higher* bar is a
#: policy judgement and belongs in a profile's declared minimum.
MIN_LIFT_LOWER_BOUND = 1.0

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

    if metrics.auroc is None:
        # No ranking claim, so the AUROC bar cannot be applied. The warrant is
        # then a claim about FPR alone, and max_fpr_hard_negatives becomes the
        # criterion that must be supplied -- without it there would be no bar
        # at all, which is worse than refusing.
        #
        # The CAUSE is read from the metrics, never inferred. An absent AUROC
        # meant "single class" until DECISIONS 108 made a threshold that flags
        # nothing produce the same shape, at which point this branch wrote
        # "measured no positives in this eval set" into a refusal for a
        # 600-item set that had plenty.
        if max_fpr_hard_negatives is None:
            reasons.append(
                RefusalReason(
                    criterion=(
                        "single_class_envelope"
                        if metrics.ranking_absent_reason is None
                        or "single-class" in metrics.ranking_absent_reason
                        else "no_ranking_claim"
                    ),
                    measured=(
                        metrics.ranking_absent_reason
                        or "no positives in this eval set, so AUROC is undefined"
                    ),
                    required=(
                        "a declared max_fpr_hard_negatives, since FPR is the only "
                        "claim an envelope without a ranking can support"
                    ),
                )
            )
    else:
        auroc_lower = metrics.auroc.ci_low
        if auroc_lower is None or auroc_lower <= config.validation.min_auroc_lower_ci:
            reasons.append(
                RefusalReason(
                    criterion="auroc_lower_ci",
                    measured="none" if auroc_lower is None else f"{auroc_lower:.4f}",
                    required=f"> {config.validation.min_auroc_lower_ci}",
                )
            )

    # Lift is recall over the measured flag rate: how many more errors this
    # finds than random sampling at the same cost. A lower bound at or below
    # 1.0 means the detector cannot be shown to beat drawing the same number of
    # items at random, which is the null the whole product is measured against.
    # Undefined on a single-class envelope, where there is no recall and the
    # warrant claims FPR only.
    if metrics.recall is not None and metrics.flag_rate.value > 0:
        lift = metrics.lift
        lift_lower = lift.ci_low if lift.ci_low is not None else lift.value
        if lift_lower <= MIN_LIFT_LOWER_BOUND:
            reasons.append(
                RefusalReason(
                    criterion="lift_lower_ci",
                    measured=(
                        f"{lift.value:.3f} [{lift.ci_low:.3f}, {lift.ci_high:.3f}] "
                        f"at measured flag rate {metrics.flag_rate.value:.4f}"
                    ),
                    required=(
                        f"> {MIN_LIFT_LOWER_BOUND} — otherwise not demonstrably "
                        "better than random sampling at the same budget"
                    ),
                )
            )

    if min_recall is not None and metrics.recall is None:
        reasons.append(
            RefusalReason(
                criterion="recall_undefined",
                measured="no positives in this eval set",
                required=f">= {min_recall} (profile minimum)",
            )
        )
    elif min_recall is not None:
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
        # The refusal names the exact build it refused. Without this prefix the
        # sentence reads as a claim about a third-party library -- "Presidio
        # ships no UPI VPA recognizer" -- which is true today and silently
        # false the day upstream adds one, with nothing in this repository
        # noticing. With it the sentence is a measurement of
        # `presidio-stock==2.2.364` on a named envelope, which stays true
        # permanently. Applies to every detector, not just the third-party
        # ones: a probe refusal is also a statement about one build.
        status_reason = "[%s==%s] %s" % (
            key.detector_id,
            detector_version,
            "; ".join(r.render() for r in reasons),
        )
        _LOG.warning(
            "REFUSED %s on %s: %s", key.detector_id, key.eval_set_id, status_reason
        )
    else:
        _LOG.info(
            "issued %s for %s on %s (AUROC lower CI %s, n_test %d)",
            warrant_id,
            key.detector_id,
            key.eval_set_id,
            "n/a (single-class envelope)"
            if metrics.auroc is None
            else f"{metrics.auroc.ci_low:.4f}",
            n_test,
        )

    # Every warrant carries its calibration claim, issued or refused. A warrant
    # asserts two separable things -- how well the detector ranks, and what its
    # operating point spends -- and the first measurement to separate them found
    # a detector whose ranking survived an envelope shift while its threshold
    # spent 56% more than declared. Computed here so no warrant can exist
    # without it (DECISIONS 069).
    calibration = assess_calibration(
        metrics.flag_rate,
        operating_point,
        tolerance=config.validation.calibration_tolerance,
    )

    return Warrant(
        warrant_id=warrant_id,
        calibration=calibration,
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
