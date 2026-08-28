"""Model-version invalidation. ``SPEC.md`` §5.4.

The second way a warrant stops being true, and the one with no measurement
behind it. Drift is something you observe: the traffic moved, the PSI says so,
the window has 200 requests in it. A model change is something you are *told*,
and the correct response is immediate and unconditional — a probe's weights are
fitted to one model's residual stream, and the same weights against a different
model are not a weaker detector, they are an unrelated one.

So this module does not measure anything. It reads the pin recorded at
validation time and takes every activation-tier warrant out of service until
somebody revalidates.

**The operational cost is the point, not a caveat.** A model upgrade suspends
the entire T1 row of the matrix. `SPEC.md` §5.4 offers the mitigation that T2
and T3 keep working through the gap, and this module implements that — but see
``DECISIONS.md`` 074 for what that claim does and does not cover. T2 and T3
survive here because their *detectors* carry no model-fitted parameters, not
because their measured numbers were shown to transfer. Nothing has measured
that, and the certificate says so.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Iterable, Optional

from ..model.enums import AccessTier, WarrantStatus
from ..model.warrant import Warrant

__all__ = [
    "ModelVersionInvalidation",
    "invalidate_for_model_change",
    "pins_to_model",
]


def pins_to_model(warrant: Warrant) -> bool:
    """Whether this warrant's numbers depend on the identity of the model.

    Keyed on ``access_tier`` rather than on the detector id. Parsing the model
    out of a name like ``probe-qwen2.5-7b-instruct-T1-last_token`` works until a
    detector is renamed, and a renaming that silently switches off invalidation
    is a failure nobody would see.

    Args:
        warrant: The warrant.

    Returns:
        True for ``T1_ACTIVATIONS``.
    """
    return warrant.access_tier is AccessTier.T1_ACTIVATIONS


@dataclasses.dataclass(frozen=True)
class ModelVersionInvalidation:
    """What a model change did to one warrant, and why.

    Args:
        warrant_id: The warrant.
        detector_id: Its detector.
        eval_set_id: Its envelope.
        access_tier: The tier, which is what decided.
        pinned_model: The model recorded at validation time. ``None`` when the
            run predates the pin being recorded — see ``invalidated``.
        live_model: The model now in service.
        from_status: Status before.
        to_status: Status after. ``REVOKED`` when invalidated.
        invalidated: Whether this warrant was taken out of service.
        reason: One line, for the certificate and the log.
    """

    warrant_id: str
    detector_id: str
    eval_set_id: str
    access_tier: AccessTier
    pinned_model: Optional[str]
    live_model: str
    from_status: WarrantStatus
    to_status: WarrantStatus
    invalidated: bool
    reason: str

    def to_payload(self) -> dict:
        return {
            "warrant_id": self.warrant_id,
            "detector_id": self.detector_id,
            "eval_set_id": self.eval_set_id,
            "access_tier": self.access_tier.name,
            "pinned_model": self.pinned_model,
            "live_model": self.live_model,
            "from_status": self.from_status.value,
            "to_status": self.to_status.value,
            "invalidated": self.invalidated,
            "reason": self.reason,
        }


def invalidate_for_model_change(
    warrants: Iterable[Warrant],
    *,
    live_model: str,
    pinned_model_of: Callable[[Warrant], Optional[str]],
) -> tuple[ModelVersionInvalidation, ...]:
    """Take every activation-tier warrant pinned to a different model out of service.

    Args:
        warrants: The warrants to assess. Typically every warrant in the matrix.
        live_model: The model now serving. Compared by exact string equality:
            a version comparison would need a version scheme, and inventing one
            here would let ``7b-instruct-v0.2`` be judged compatible with
            ``7b-instruct-v0.1`` on no evidence at all.
        pinned_model_of: Resolves the model a warrant's validation run read
            activations from. Returning ``None`` means the pin is unknown.

    Returns:
        One :class:`ModelVersionInvalidation` per warrant, invalidated or not.
        Warrants that survive are returned too — a caller needs to be able to
        show what a model change did *not* touch, which is the entire content
        of the "T2 and T3 keep working" mitigation.
    """
    out: list[ModelVersionInvalidation] = []

    for warrant in warrants:
        pinned = pinned_model_of(warrant)
        tier = warrant.access_tier

        def record(to_status: WarrantStatus, invalidated: bool, reason: str,
                   _w: Warrant = warrant, _p: Optional[str] = pinned,
                   _t: AccessTier = tier) -> None:
            """Append one verdict. Loop variables are bound as defaults rather
            than closed over, so the record cannot describe a later warrant."""
            out.append(
                ModelVersionInvalidation(
                    warrant_id=_w.warrant_id,
                    detector_id=_w.detector_id,
                    eval_set_id=_w.eval_set_id,
                    access_tier=_t,
                    pinned_model=_p,
                    live_model=live_model,
                    from_status=_w.status,
                    to_status=to_status,
                    invalidated=invalidated,
                    reason=reason,
                )
            )

        # A warrant refused by its controls is already out of service and stays
        # there. Invalidating it again would let a later revalidation against
        # the new model read as though the refusal had been a model problem.
        if warrant.status is WarrantStatus.REFUSED:
            record(
                WarrantStatus.REFUSED,
                False,
                "already REFUSED by its controls; a model change does not "
                "alter that and does not explain it",
            )
            continue

        if not pins_to_model(warrant):
            record(
                warrant.status,
                False,
                "%s carries no model-fitted parameters, so the model change "
                "leaves the detector intact. Its measured bounds were not "
                "re-measured on %s and are not claimed to transfer "
                "(DECISIONS.md 074)." % (tier.name, live_model),
            )
            continue

        # An activation-tier warrant whose pin was never recorded cannot be
        # shown to match the live model. Unknown is not the same as matching,
        # and defaulting it to "keep serving" would make the one warrant class
        # this module exists to protect the one it silently skips.
        if pinned is None:
            record(
                WarrantStatus.REVOKED,
                True,
                "activation-tier warrant with no recorded model pin. It cannot "
                "be shown to have been measured on %s, and an unrecorded pin is "
                "not a matching one." % live_model,
            )
            continue

        if pinned != live_model:
            record(
                WarrantStatus.REVOKED,
                True,
                "measured on %r, serving %r. A probe's weights are fitted to "
                "one model's residual stream; against a different model they "
                "are not a weaker detector but an unrelated one. Out of "
                "service until revalidation." % (pinned, live_model),
            )
            continue

        record(warrant.status, False, "pinned to %r, which is serving" % pinned)

    return tuple(out)
