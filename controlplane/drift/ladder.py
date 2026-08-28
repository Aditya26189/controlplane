"""The revocation ladder: a drift verdict decides what a warrant is still worth.

``SPEC.md`` §5.3. A warrant is issued against one envelope; live traffic moves;
the ladder says what remains claimable.

| envelope state | warrant becomes | behaviour |
|---|---|---|
| ``INSIDE`` | ``VALID`` | full bounds |
| ``MODERATE_SHIFT`` | ``STALE`` | widen reported bounds, schedule revalidation |
| ``SIGNIFICANT_SHIFT`` | ``REVOKED`` | refuse to certify; consult the matrix |
| ``INSUFFICIENT_DATA`` | unchanged | no verdict, so no transition |

**The last row is the one that matters and it is easy to get wrong.** Below the
window minimum there is no evidence, and no evidence must not become a decision
in either direction — neither "still valid" nor "revoke to be safe". The warrant
keeps whatever status it already had, and the transition records that nothing
was concluded. A monitor that revokes on forty requests gets switched off; one
that certifies on forty requests is asserting stability it never measured.

**Widening is not a courtesy.** A ``STALE`` warrant reports wider bounds because
the distribution it was measured on has moved and the old interval no longer
covers the new one honestly. The widening factor is declared in config rather
than derived, because deriving it from the PSI would imply a calibration between
PSI and interval width that nobody has measured — and inventing one would be
exactly the unbacked claim this product refuses. It is a stated policy, and the
certificate says so.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from ..model.enums import EnvelopeState, WarrantStatus
from ..model.warrant import Warrant
from .monitor import DriftVerdict

__all__ = ["LadderTransition", "apply_ladder"]

#: Envelope state -> the status a warrant takes on. ``INSUFFICIENT_DATA`` is
#: absent deliberately: it maps to "no transition", which is not a status.
_TRANSITIONS = {
    EnvelopeState.INSIDE: WarrantStatus.VALID,
    EnvelopeState.MODERATE_SHIFT: WarrantStatus.STALE,
    EnvelopeState.SIGNIFICANT_SHIFT: WarrantStatus.REVOKED,
}


@dataclasses.dataclass(frozen=True)
class LadderTransition:
    """What the ladder decided, and everything a certificate needs to explain it.

    Args:
        warrant_id: The warrant this applies to.
        detector_id: Its detector.
        eval_set_id: The envelope it was issued against.
        from_status: Status before the verdict.
        to_status: Status after. Equal to ``from_status`` when nothing changed.
        envelope_state: The rung the monitor reported.
        verdict: The drift verdict, carried whole so the certificate can quote
            the PSI, the driving feature and the smoothing caveat.
        widen_factor: Applied to reported interval half-widths when ``STALE``.
            ``None`` otherwise.
        reason: One line, for the certificate and the log.
    """

    warrant_id: str
    detector_id: str
    eval_set_id: str
    from_status: WarrantStatus
    to_status: WarrantStatus
    envelope_state: EnvelopeState
    verdict: DriftVerdict
    widen_factor: Optional[float]
    reason: str

    @property
    def changed(self) -> bool:
        return self.from_status is not self.to_status

    @property
    def needs_routing(self) -> bool:
        """A revoked warrant cannot certify; something else must be found."""
        return self.to_status is WarrantStatus.REVOKED

    def to_payload(self) -> dict:
        return {
            "warrant_id": self.warrant_id,
            "detector_id": self.detector_id,
            "eval_set_id": self.eval_set_id,
            "from_status": self.from_status.value,
            "to_status": self.to_status.value,
            "changed": self.changed,
            "envelope_state": self.envelope_state.value,
            "widen_factor": self.widen_factor,
            "needs_routing": self.needs_routing,
            "reason": self.reason,
            "verdict": self.verdict.to_payload(),
        }


def apply_ladder(
    warrant: Warrant, verdict: DriftVerdict, *, widen_factor: float = 1.5
) -> LadderTransition:
    """Decide what a warrant is worth given what the monitor saw.

    Args:
        warrant: The warrant under observation.
        verdict: The monitor's verdict for its envelope.
        widen_factor: Multiplier applied to interval half-widths while
            ``STALE``. Declared, not derived from the PSI: no calibration
            between PSI and interval width has been measured here, and
            inventing one would be an unbacked claim.

    Returns:
        A :class:`LadderTransition`. Never mutates the warrant — the ledger is
        append-only and the transition is the record.
    """
    state = verdict.state

    if state is EnvelopeState.INSUFFICIENT_DATA:
        return LadderTransition(
            warrant_id=warrant.warrant_id,
            detector_id=warrant.detector_id,
            eval_set_id=warrant.eval_set_id,
            from_status=warrant.status,
            to_status=warrant.status,
            envelope_state=state,
            verdict=verdict,
            widen_factor=None,
            reason=(
                "no transition: %s. The warrant keeps the status it already "
                "had, because absent evidence is not a decision in either "
                "direction." % verdict.reason
            ),
        )

    target = _TRANSITIONS[state]

    # A warrant already refused by its controls does not climb back up because
    # live traffic happens to look like its reference. Control failure is a
    # statement about the detector; drift is a statement about the traffic, and
    # the first is not cured by the second.
    if warrant.status is WarrantStatus.REFUSED:
        return LadderTransition(
            warrant_id=warrant.warrant_id,
            detector_id=warrant.detector_id,
            eval_set_id=warrant.eval_set_id,
            from_status=warrant.status,
            to_status=WarrantStatus.REFUSED,
            envelope_state=state,
            verdict=verdict,
            widen_factor=None,
            reason=(
                "stays REFUSED: the envelope is %s, but a warrant refused by "
                "its controls is out of service until human revalidation. "
                "Drift describes the traffic; the refusal describes the "
                "detector." % state.value
            ),
        )

    factor = widen_factor if target is WarrantStatus.STALE else None
    if target is WarrantStatus.STALE:
        tail = (
            " Reported bounds widen by %.2fx and revalidation is scheduled. The "
            "factor is declared policy, not derived from the PSI: no calibration "
            "between PSI and interval width has been measured." % widen_factor
        )
    elif target is WarrantStatus.REVOKED:
        tail = (
            " Refusing to certify on this envelope. The matrix is consulted for "
            "a detector holding a valid warrant here."
        )
    else:
        tail = ""

    return LadderTransition(
        warrant_id=warrant.warrant_id,
        detector_id=warrant.detector_id,
        eval_set_id=warrant.eval_set_id,
        from_status=warrant.status,
        to_status=target,
        envelope_state=state,
        verdict=verdict,
        widen_factor=factor,
        reason="%s -> %s. %s%s" % (
            warrant.status.value, target.value, verdict.reason, tail
        ),
    )
