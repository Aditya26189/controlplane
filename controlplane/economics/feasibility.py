"""The abstention floor: what no detector can do, however good.

This answers the question every threshold discussion eventually reaches —
*"why not just tighten the threshold until the error rate is acceptable?"* — with
an impossibility result rather than an opinion.

**The bound.** Let ``mu`` be the base error rate of the traffic and ``alpha``
the risk you are willing to ship on the responses you keep. Let ``a`` be the
fraction abstained on — routed to review, escalated, refused. In the *best
possible* case every abstained item is an error, so the errors remaining among
the kept items are ``mu - a`` out of ``1 - a``:

    (mu - a) / (1 - a)  <=  alpha
    mu - a              <=  alpha - alpha*a
    mu - alpha          <=  a * (1 - alpha)
    a                   >=  (mu - alpha) / (1 - alpha)

That final quantity is the floor. It assumes a **perfect** selector — one that
abstains only on errors and never on a correct response. No detector can do
better, so no threshold, no ensemble and no amount of tuning gets under it.

Two consequences worth stating plainly:

- When ``mu <= alpha`` the floor is zero: the traffic already meets the target
  and selection is not needed for risk, only for cost.
- When ``alpha`` is small and ``mu`` is not, the floor approaches ``mu``. At the
  measured TriviaQA base rate this is the difference between "tighten the
  threshold" and "abstain on nearly half the traffic".

``mu`` is **measured** — it comes from an artifact in ``results/``. ``alpha`` is
**declared** — it is a risk appetite someone chooses. Every result here carries
that distinction, because the bound is only as defensible as the base rate under
it, and the base rate is a property of one envelope.

This is the standard selective-prediction feasibility argument, and it is the
same inequality that underlies distribution-free risk control: it constrains the
achievable (risk, coverage) pair from below regardless of the score function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "AbstentionFloor",
    "AchievedRisk",
    "abstention_floor",
    "achieved_risk",
    "feasibility_curve",
]


@dataclass(frozen=True)
class AbstentionFloor:
    """The minimum abstention rate compatible with a target risk.

    Attributes:
        base_error_rate: ``mu``. Measured, and a property of one envelope.
        target_risk: ``alpha``. Declared.
        floor: Minimum fraction of traffic any selector must abstain on.
        attainable: Whether the target is reachable at all. False only when
            ``alpha >= 1``, which is not a target.
        binding: Whether the floor is above zero, i.e. whether selection is
            forced by the risk target rather than merely useful for cost.
        envelope_id: The envelope ``mu`` was measured on, carried so the number
            cannot be quoted against a distribution it does not describe.
        source: The artifact ``mu`` came from.
    """

    base_error_rate: float
    target_risk: float
    floor: float
    attainable: bool
    binding: bool
    envelope_id: str = ""
    source: str = ""

    @property
    def retained(self) -> float:
        """The largest fraction of traffic that can be served without review."""
        return 1.0 - self.floor

    def render(self) -> str:
        if not self.binding:
            return (
                f"base error rate {self.base_error_rate:.4f} already meets a "
                f"target of {self.target_risk:.4f}; no abstention is forced"
            )
        return (
            f"to hold risk at {self.target_risk:.4f} on traffic with a measured "
            f"base error rate of {self.base_error_rate:.4f}, ANY selector must "
            f"abstain on at least {self.floor:.4f} of it "
            f"({self.floor * 100:.1f}%), keeping at most {self.retained * 100:.1f}%"
        )


def abstention_floor(
    base_error_rate: float,
    target_risk: float,
    *,
    envelope_id: str = "",
    source: str = "",
) -> AbstentionFloor:
    """Minimum abstention for a target risk, assuming a perfect selector.

    Args:
        base_error_rate: ``mu``, the measured error rate of the traffic.
        target_risk: ``alpha``, the declared risk ceiling on retained responses.
        envelope_id: Envelope ``mu`` was measured on. Carried, not checked.
        source: Artifact ``mu`` came from.

    Returns:
        The floor, with its inputs and their provenance attached.

    Raises:
        ValueError: If either rate is outside ``[0, 1]``. A rate outside the
            unit interval is a caller error, and silently clamping it would
            produce a plausible-looking bound from an impossible input.
    """
    for name, value in (("base_error_rate", base_error_rate), ("target_risk", target_risk)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}={value!r} is not a rate in [0, 1]")

    if target_risk >= 1.0:
        # Everything is acceptable, so nothing is forced -- but "risk <= 1" is
        # not a target, and returning 0.0 without saying so would read as a
        # result rather than as a degenerate input.
        return AbstentionFloor(
            base_error_rate=base_error_rate,
            target_risk=target_risk,
            floor=0.0,
            attainable=False,
            binding=False,
            envelope_id=envelope_id,
            source=source,
        )

    raw = (base_error_rate - target_risk) / (1.0 - target_risk)
    floor = max(0.0, raw)
    return AbstentionFloor(
        base_error_rate=base_error_rate,
        target_risk=target_risk,
        floor=floor,
        attainable=True,
        binding=floor > 0.0,
        envelope_id=envelope_id,
        source=source,
    )


def feasibility_curve(
    base_error_rate: float,
    target_risks: Sequence[float],
    *,
    envelope_id: str = "",
    source: str = "",
) -> list[AbstentionFloor]:
    """The floor at each of several declared targets, for one measured base rate.

    A curve rather than a point because ``alpha`` is a choice, and presenting one
    value would hide that the shape is what makes the argument: the floor rises
    steeply as the target tightens, and no engineering moves it.
    """
    return [
        abstention_floor(
            base_error_rate, alpha, envelope_id=envelope_id, source=source
        )
        for alpha in target_risks
    ]


@dataclass(frozen=True)
class AchievedRisk:
    """What an operating point actually delivers, and how close to optimal.

    Every field here is derived from three quantities measured on **one**
    envelope -- base rate, recall, flag rate -- so unlike the review sizing
    there is no declared input anywhere in it and no scenario to mix.

    Attributes:
        residual_risk: Error rate among the responses this point does NOT flag.
            The number a profile is actually shipping.
        floor_at_residual_risk: What a perfect selector would have to abstain
            on to achieve that same residual risk.
        abstention: The measured flag rate -- what this point does abstain on.
        efficiency: ``abstention / floor_at_residual_risk``. 1.0 is optimal;
            2.0 means twice as much review as the risk actually requires.
    """

    operating_point_id: str
    base_error_rate: float
    recall: float
    abstention: float
    residual_risk: float
    floor_at_residual_risk: float
    efficiency: float

    def render(self) -> str:
        return (
            f"{self.operating_point_id}: abstains on {self.abstention:.4f}, "
            f"ships residual risk {self.residual_risk:.4f}. A perfect selector "
            f"at that same risk would abstain on {self.floor_at_residual_risk:.4f}, "
            f"so this point costs {self.efficiency:.2f}x the review the risk "
            f"strictly requires"
        )


def achieved_risk(
    *,
    operating_point_id: str,
    base_error_rate: float,
    recall: float,
    flag_rate: float,
) -> AchievedRisk:
    """What a measured operating point ships, and its distance from the floor.

    All three inputs must come from the **same** envelope. Mixing a base rate
    from one distribution with a recall from another produces a number that
    describes neither, which is the error ``CLAUDE.md`` names as most damaging.

    Args:
        operating_point_id: For labelling.
        base_error_rate: Measured ``mu`` on the envelope.
        recall: Measured recall at this point on the same envelope.
        flag_rate: Measured flag rate at this point on the same envelope.

    Returns:
        The residual risk, the floor at that risk, and the ratio between the
        measured abstention and that floor.

    Raises:
        ValueError: If a rate is outside ``[0, 1]``, or if the flag rate is 1.0
            and nothing is retained, so a residual risk is undefined rather
            than zero.
    """
    for name, value in (
        ("base_error_rate", base_error_rate),
        ("recall", recall),
        ("flag_rate", flag_rate),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}={value!r} is not a rate in [0, 1]")
    retained = 1.0 - flag_rate
    if retained <= 0.0:
        raise ValueError(
            "flag_rate=1.0 retains nothing, so a residual risk is undefined "
            "rather than zero"
        )

    # A detector cannot catch more errors than it flagged. mu * recall is the
    # share of ALL traffic that is a caught error, and it can never exceed the
    # share that was flagged. When it does, the three rates did not come from
    # one envelope -- and the result would be an efficiency below 1.0, i.e. an
    # operating point apparently beating a bound that no selector can beat.
    #
    # Found by test_efficiency_is_never_below_one_for_an_achievable_point,
    # which generated the impossible combination before any real one hit it.
    caught = base_error_rate * recall
    if caught > flag_rate + 1e-9:
        raise ValueError(
            f"recall={recall!r} at base rate {base_error_rate!r} implies "
            f"{caught:.4f} of all traffic is a caught error, but only "
            f"{flag_rate!r} was flagged. These three rates cannot describe one "
            "envelope; check they were not taken from different measurements."
        )

    missed = base_error_rate * (1.0 - recall)
    residual = missed / retained
    floor = abstention_floor(base_error_rate, min(residual, 1.0)).floor
    # A floor of zero means the residual risk this point ships is already at or
    # above the base rate -- the point is not reducing risk at all -- so an
    # efficiency ratio would divide by zero and would not mean anything.
    efficiency = float("inf") if floor <= 0.0 else flag_rate / floor
    return AchievedRisk(
        operating_point_id=operating_point_id,
        base_error_rate=base_error_rate,
        recall=recall,
        abstention=flag_rate,
        residual_risk=residual,
        floor_at_residual_risk=floor,
        efficiency=efficiency,
    )
