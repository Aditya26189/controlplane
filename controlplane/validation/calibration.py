"""The calibration claim: does the threshold still spend what it said it would?

**A warrant makes two claims, not one.** It asserts how well the detector
*ranks* — AUROC, and the lift derived from it — and it asserts an *operating
point*: this recall, at this flag rate. Those are separable properties, and the
first measurement to separate them found a detector where one held and the other
was in question.

``T1-last_token`` transferred to the long-context envelope with AUROC 0.826 ->
0.813, barely moved, while the frozen threshold went from flagging 4.2% of
traffic to 6.5% — a 56% budget overrun at an unchanged ranking. A warrant that
pins recall to a stated budget is, on that envelope, asserting something the
detector may no longer do, even though everything about its ranking is sound.
That is a warrant making an unbacked claim, in exactly the sense this product
exists to refuse (``DECISIONS.md`` 067, 069).

**Two states, and neither of them is "fine".** The test is whether the declared
target falls inside the realised flag rate's interval:

* ``DRIFTED`` — the target is outside. The operating point is demonstrably not
  the operating point any more.
* ``CALIBRATED`` — the target is inside, so drift is *not shown*. That is weaker
  than "calibration held", and the claim carries ``underpowered`` to say which
  of the two it is.

**Power is judged against a deviation worth acting on, never the observed one.**
``config.validation.calibration_tolerance`` declares that deviation (25% by
default), and a claim is underpowered when the realised interval extends outside
the resulting band — the sample cannot rule out a departure large enough to act
on. Measuring power against the *observed* gap is the obvious thing and it is
wrong: as an estimate approaches its target the gap shrinks and the ``n`` needed
to resolve it diverges, so near-perfect agreement reports an ``n`` of 18 million
and gets flagged as underpowered. A test caught that.

At n=600 a realised 6.5% against a 5% target has interval [0.048, 0.085]. The
target is inside, so drift is not shown — but the interval reaches past 0.0625,
so a 25% overspend is not ruled out either. Detecting one needs n≥1441.
"""

from __future__ import annotations

import math
from typing import Optional

from ..model.calibration import CalibrationClaim
from ..model.enums import CalibrationStatus
from ..model.findings import OperatingPoint
from ..model.metrics import Metric

__all__ = ["CalibrationClaim", "assess_calibration", "n_to_detect_deviation"]


def n_to_detect_deviation(
    target: float, tolerance: float, confidence: float = 1.96
) -> int:
    """Sample size needed to detect a deviation of ``tolerance`` from ``target``.

    Power is measured against **an effect worth acting on**, never against the
    effect that happened to be observed. The first version of this used the
    observed gap, which meant an estimate landing near the target reported an
    enormous required ``n`` — 18 million to separate 0.0501 from 0.0500 — and so
    flagged near-perfect agreement as underpowered. That is backwards: as an
    estimate approaches its target the observed gap shrinks and the ``n`` to
    resolve it diverges, while the sample's ability to catch a deviation that
    mattered is unchanged.

    A test caught it. ``test_a_well_powered_agreement_is_not_flagged``.
    """
    deviated = target * (1.0 + tolerance)
    gap = abs(deviated - target)
    if gap <= 0.0:
        raise ValueError("tolerance must be positive to define a detectable effect")
    variance = deviated * (1.0 - deviated)
    return max(1, int(math.ceil((confidence**2) * variance / (gap**2))))


def assess_calibration(
    flag_rate: Optional[Metric],
    operating_point: OperatingPoint,
    tolerance: float = 0.25,
) -> CalibrationClaim:
    """Compare the realised flag rate against the budget the threshold declared.

    Args:
        flag_rate: Measured flag rate on this envelope, with its interval.
        operating_point: Carries ``target_flag_rate``, the budget aimed at.
        tolerance: Relative deviation from the budget worth acting on, from
            ``config.validation.calibration_tolerance``. Power is judged
            against this rather than against the observed deviation.

    Returns:
        A :class:`CalibrationClaim`. Absent inputs produce ``UNKNOWN`` rather
        than ``CALIBRATED``: a claim nobody could evaluate must not read as a
        claim that passed (``DECISIONS.md`` 050).
    """
    target = operating_point.target_flag_rate
    if target is None or flag_rate is None:
        return CalibrationClaim(
            status=CalibrationStatus.UNKNOWN,
            target_flag_rate=target,
            realised=flag_rate,
            n_to_detect=None,
            tolerance=tolerance,
            detail=(
                "no target flag rate declared on the operating point"
                if target is None
                else "no measured flag rate on this envelope"
            ),
        )

    low, high = flag_rate.ci_low, flag_rate.ci_high
    needed = n_to_detect_deviation(target, tolerance)
    if low is None or high is None:
        return CalibrationClaim(
            status=CalibrationStatus.UNKNOWN,
            target_flag_rate=target,
            realised=flag_rate,
            n_to_detect=needed,
            tolerance=tolerance,
            detail="flag rate carries no interval, so drift cannot be tested",
        )

    # The band of budgets close enough to the target to be worth accepting.
    # Power is judged against this, not against the observed deviation.
    band_low, band_high = target * (1.0 - tolerance), target * (1.0 + tolerance)

    if low <= target <= high:
        ratio = flag_rate.value / target if target else float("nan")
        resolved = band_low <= low and high <= band_high
        return CalibrationClaim(
            status=CalibrationStatus.CALIBRATED,
            target_flag_rate=target,
            realised=flag_rate,
            n_to_detect=needed,
            tolerance=tolerance,
            detail=(
                "target %.4f lies inside the realised interval [%.4f, %.4f] at "
                "n=%s, so drift is not shown. Point estimate %.4f is %.2fx the "
                "target. %s"
                % (
                    target, low, high, flag_rate.n, flag_rate.value, ratio,
                    (
                        "The interval also fits inside the +/-%.0f%% band "
                        "[%.4f, %.4f], so a deviation worth acting on is ruled "
                        "out." % (tolerance * 100, band_low, band_high)
                        if resolved
                        else "But the interval extends outside the +/-%.0f%% "
                        "band [%.4f, %.4f], so a deviation worth acting on is "
                        "NOT ruled out; detecting one needs n>=%s."
                        % (tolerance * 100, band_low, band_high, needed)
                    ),
                )
            ),
        )

    direction = "over" if flag_rate.value > target else "under"
    return CalibrationClaim(
        status=CalibrationStatus.DRIFTED,
        target_flag_rate=target,
        realised=flag_rate,
        n_to_detect=needed,
        tolerance=tolerance,
        detail=(
            "target %.4f lies outside the realised interval [%.4f, %.4f] at "
            "n=%s: the threshold %sspends its budget and the operating point is "
            "no longer the operating point."
            % (target, low, high, flag_rate.n, direction)
        ),
    )
