"""The calibration claim: a record, kept beside the warrant that makes it.

A warrant asserts two separable things -- how well a detector *ranks*, and
what its operating point *spends*. This record carries the second. It lives
in ``controlplane/model`` because it is a record rather than a computation; the
assessment that produces it is ``controlplane.validation.calibration``.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from .enums import CalibrationStatus
from .metrics import Metric

__all__ = ["CalibrationClaim"]


@dataclasses.dataclass(frozen=True)
class CalibrationClaim:
    """Whether the operating point still delivers the budget it declared.

    Args:
        status: See :class:`CalibrationStatus`.
        target_flag_rate: The budget the threshold was chosen to hit, from the
            operating point. None when none was declared, which makes the claim
            ``UNKNOWN`` rather than passing by default.
        realised: The measured flag rate on this envelope, with its interval.
        n_to_detect: Sample size that would separate ``realised`` from
            ``target``. Populated even when the status is ``CALIBRATED``,
            because that is precisely when it is needed.
        detail: One line, for the warrant's status reason.
    """

    status: CalibrationStatus
    target_flag_rate: Optional[float]
    realised: Optional[Metric]
    n_to_detect: Optional[int]
    detail: str
    #: Relative deviation from the budget that would be worth acting on.
    tolerance: float = 0.25
    #: True when drift was not shown at an ``n`` that could not have shown it.
    #: A **stored field**, not a property: the ledger serialises fields, so a
    #: property would seal a record whose reader had to recompute the one
    #: caveat that stops ``CALIBRATED`` reading as reassurance. A caveat that
    #: requires derivation is a caveat nobody derives (``DECISIONS.md`` 050).
    #: Computed in ``__post_init__`` so it cannot disagree with the other fields.
    underpowered: bool = dataclasses.field(init=False, default=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "underpowered", self._compute_underpowered())

    def _compute_underpowered(self) -> bool:
        """Whether this sample could have caught a deviation worth acting on.

        Judged against the tolerance band, never against the observed gap. The
        first version compared ``n_to_detect`` for the *observed* deviation
        against ``n``, which flagged near-perfect agreement as underpowered:
        as an estimate approaches its target the gap shrinks and the ``n`` to
        resolve it diverges, while the sample's power to catch a deviation that
        mattered is unchanged.

        Derived here from the stored fields rather than passed in, so the flag
        cannot disagree with the interval and tolerance it describes. An earlier
        version took it as a constructor argument named ``underpowered_override``
        and ``test_no_override`` rejected the name. The guard was right twice
        over: the value is computable, so a caller-supplied one is a second
        source of truth.
        """
        if self.status is not CalibrationStatus.CALIBRATED:
            return False
        if self.realised is None or self.target_flag_rate is None:
            return False
        low, high = self.realised.ci_low, self.realised.ci_high
        if low is None or high is None:
            return False
        # The sample resolves the question only if the whole interval sits
        # inside the band of budgets close enough to the target to accept.
        band_low = self.target_flag_rate * (1.0 - self.tolerance)
        band_high = self.target_flag_rate * (1.0 + self.tolerance)
        return not (band_low <= low and high <= band_high)

    def to_payload(self) -> dict:
        return {
            "status": self.status.value,
            "target_flag_rate": self.target_flag_rate,
            "realised_flag_rate": None if self.realised is None else self.realised.value,
            "realised_ci": (
                None
                if self.realised is None
                else [self.realised.ci_low, self.realised.ci_high]
            ),
            "n": None if self.realised is None else self.realised.n,
            "n_to_detect": self.n_to_detect,
            "underpowered": self.underpowered,
            "detail": self.detail,
        }
