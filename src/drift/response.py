"""Drift response: detect, transition, route — with no manual trigger.

This composes the Phase 5 gate. Traffic arrives, the monitor scores it against
the envelope stored in the warrant, the ladder says what the warrant is still
worth, and a revocation consults the matrix for a detector holding a valid
warrant on the envelope the traffic is *actually* in.

**Nothing here waits for a human.** That is the whole point of the gate: a
revocation that needs an operator to notice it is a dashboard, not a control.
The one thing a human is required for is the opposite direction — a warrant
refused by its controls stays refused until revalidation, and no amount of
favourable traffic lifts it.

**Routing may return the same detector on a different envelope, and that is a
result rather than a no-op.** Measured here: ``T1-last_token`` holds a warrant
on ``triviaqa-600`` at R=0.08 @f=0.042 and another on ``triviaqa-longctx-600``
at R=0.13 @f=0.065. When long-context traffic revokes the first, the system does
not change detector — it stops quoting bounds measured on short context and
starts quoting the ones measured on the envelope the traffic is in. The detector
was never the thing that went stale; the *claim* was.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

from ..matrix.routing import Profile, RoutingDecision, route
from ..model.enums import EnvelopeState, WarrantStatus
from ..model.warrant import Warrant
from .ladder import LadderTransition, apply_ladder
from .monitor import DriftMonitor

__all__ = ["DriftResponse", "respond_to_drift"]


@dataclasses.dataclass(frozen=True)
class DriftResponse:
    """What the system did about the drift, and everything needed to explain it.

    Args:
        transition: The ladder's decision on the observed warrant.
        routing: The matrix consultation, when the transition required one.
            ``None`` when the warrant is still usable and nothing was needed.
        claimed_bounds: What may now be asserted. Empty when the warrant is
            revoked and nothing else holds one here — a system with nothing to
            claim must claim nothing rather than fall back to its last good
            numbers.
        acted_without_operator: Always true; recorded explicitly because "no
            manual trigger" is the gate's requirement and a certificate that
            merely implies it proves nothing.
    """

    transition: LadderTransition
    routing: Optional[RoutingDecision]
    claimed_bounds: dict[str, Any]
    acted_without_operator: bool = True

    @property
    def can_certify(self) -> bool:
        """Whether anything backs a claim on this traffic."""
        if self.transition.to_status in (WarrantStatus.VALID, WarrantStatus.STALE):
            return True
        return self.routing is not None and self.routing.routed

    def to_payload(self) -> dict:
        return {
            "transition": self.transition.to_payload(),
            "routing": None if self.routing is None else self.routing.to_payload(),
            "claimed_bounds": self.claimed_bounds,
            "can_certify": self.can_certify,
            "acted_without_operator": self.acted_without_operator,
        }


def _widened(bounds: dict[str, Any], factor: float) -> dict[str, Any]:
    """Widen interval half-widths about their point estimate.

    Applied to a ``STALE`` warrant's bounds. The factor is declared policy from
    the ladder, not derived from the PSI: no calibration between PSI and
    interval width has been measured here, and inventing one would be precisely
    the unbacked claim this product refuses.
    """
    widened: dict[str, Any] = {}
    for name, value in bounds.items():
        if (
            isinstance(value, dict)
            and value.get("ci_low") is not None
            and value.get("ci_high") is not None
            and value.get("value") is not None
        ):
            point = float(value["value"])
            low = point - (point - float(value["ci_low"])) * factor
            high = point + (float(value["ci_high"]) - point) * factor
            widened[name] = {
                **value,
                "ci_low": low,
                "ci_high": high,
                "widened_by": factor,
                "widening_note": (
                    "envelope is MODERATE_SHIFT; bounds widened by declared "
                    "policy, not derived from the PSI"
                ),
            }
        else:
            widened[name] = value
    return widened


def respond_to_drift(
    warrant: Warrant,
    monitor: DriftMonitor,
    *,
    matrix,
    profile: Profile,
    live_eval_set_id: str,
    widen_factor: float = 1.5,
) -> DriftResponse:
    """Score the window, apply the ladder, and route if the warrant is revoked.

    Args:
        warrant: The warrant currently being relied on.
        monitor: Its drift monitor, already fed the live window.
        matrix: The warrant matrix, for the revocation consultation.
        profile: What the caller requires of a replacement.
        live_eval_set_id: The envelope the traffic is actually in, keyed the
            way the matrix keys its axis. Routing is done against **this**, not
            against the warrant's own envelope — searching the envelope that
            was just revoked would return the warrant that was just revoked.
            Passing a content hash raises rather than reporting an empty
            result, because a missed lookup must not read as a measured
            absence.
        widen_factor: Multiplier for ``STALE`` bounds.

    Returns:
        A :class:`DriftResponse`.
    """
    transition = apply_ladder(warrant, monitor.verdict(), widen_factor=widen_factor)

    if not transition.needs_routing:
        # ``claimed_bounds()`` and not the metrics payload: the assertion a
        # certificate makes carries the envelope, the n and the operating point
        # alongside each interval, and a bare metrics dump would quote the
        # numbers without the conditions they were measured under.
        bounds = warrant.claimed_bounds()
        if transition.to_status is WarrantStatus.STALE and transition.widen_factor:
            bounds = _widened(bounds, transition.widen_factor)
        return DriftResponse(
            transition=transition, routing=None, claimed_bounds=bounds
        )

    decision = route(matrix, live_eval_set_id, profile)
    return DriftResponse(
        transition=transition,
        routing=decision,
        # Empty when nothing holds a warrant here. A revoked system must claim
        # nothing rather than keep quoting the bounds it just lost.
        claimed_bounds=dict(decision.claimed_bounds) if decision.routed else {},
    )
