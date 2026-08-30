"""Composing two warranted detectors into one decision. ``DECISIONS.md`` 088.

The rules were written down before this file existed. Where a reader wants to
know *why* a case resolves the way it does, 088 is the answer and this module is
the mechanism; nothing here decides policy that is not stated there.

## The one-line version of the argument

The brief observes that a fabricated detail about a person is simultaneously a
hallucination and a privacy concern, which makes clean categorisation hard. Our
answer is that **no categorisation is required**: a warrant certifies a
detector's operating point, not a taxonomy bucket. Two detectors score the same
input, each carries its own warrant with its own bounds on its own envelope, and
this module composes the two *decisions*. Nothing has to decide which category
the input really is.

There is deliberately no taxonomy classifier here. Building one would concede
the point while appearing to answer it.

## Actions compose. Bounds do not.

An action is a decision about one request, so a composed action is a function of
the two detectors' actions. A bound is a measured claim about one detector on
one envelope, and no arithmetic turns two of them into a joint bound without a
measurement of the pair — the detectors' errors are not independent, since both
read the same text, and assuming independence to multiply them would manufacture
a number nobody measured.

So :class:`ComposedDecision` carries both bounds **keyed by detector**, never
merged, and claims no joint recall.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional, Sequence

from ..model.enums import Action, WarrantStatus
from ..model.findings import Finding
from ..model.warrant import Warrant

__all__ = [
    "RESTRICTIVENESS",
    "DetectorVerdict",
    "ComposedDecision",
    "compose",
]

#: The ladder case 1 resolves on. Ordered by how much the action withholds from
#: the user, which is the only ordering that makes "more restrictive" mean
#: anything. ``Action`` is deliberately not an ``IntEnum`` — there is no
#: universal ordering on it — so the ordering is declared here, where it is
#: being used for one specific purpose.
RESTRICTIVENESS: Mapping[Action, int] = {
    Action.ALLOW: 0,
    Action.REDACT: 1,
    Action.CONFIRM: 2,
    Action.ESCALATE: 3,
    Action.BLOCK: 4,
}


@dataclasses.dataclass(frozen=True)
class DetectorVerdict:
    """What one detector said, and what its warrant says that is worth.

    Args:
        detector_id: Which detector.
        status: Its warrant status **on this envelope**. ``UNVALIDATED`` when it
            has never been measured here, which is the modal state.
        fired: Whether it flagged this input.
        action: What it would do on its own. Ignored when it did not fire.
        warrant: The warrant, when one exists. ``None`` for ``UNVALIDATED``.
        finding: The finding it produced, if any. Carried even when the detector
            holds no warrant — an unwarranted finding is still information, it
            just licenses no claim about how often it is right.
    """

    detector_id: str
    status: WarrantStatus
    fired: bool
    action: Action = Action.ALLOW
    warrant: Optional[Warrant] = None
    finding: Optional[Finding] = None

    @property
    def relied_upon(self) -> bool:
        """Whether this detector's bounds may be quoted.

        ``VALID`` only. ``REFUSED`` is known-unreliable here and ``UNVALIDATED``
        has no measurement at all; neither supplies a bound.
        """
        return self.status is WarrantStatus.VALID and self.warrant is not None


@dataclasses.dataclass(frozen=True)
class ComposedDecision:
    """One action, both detectors' bounds, and what was not checked.

    Args:
        action: What to do.
        rule: Which of the four cases in ``DECISIONS.md`` 088 decided it.
        reason: Why, in words.
        claimed_bounds: Keyed **by detector id**. Never merged into a joint
            claim; see the module docstring.
        warrants_relied_upon: Ids of the warrants whose bounds are quoted.
        weakest_status: Across the detectors actually relied upon. ``UNVALIDATED``
            when none were.
        findings: Every finding, including from detectors with no warrant.
        unchecked: What this decision cannot speak to, in words.
        enqueue_for_validation: Detector ids that fired without a warrant here.
            How the matrix fills itself in from live traffic.
    """

    action: Action
    rule: str
    reason: str
    claimed_bounds: dict[str, Any]
    warrants_relied_upon: tuple[str, ...]
    weakest_status: WarrantStatus
    findings: tuple[Finding, ...]
    unchecked: tuple[str, ...]
    enqueue_for_validation: tuple[str, ...] = ()

    def to_payload(self) -> dict:
        return {
            "action": self.action.value,
            "rule": self.rule,
            "reason": self.reason,
            "claimed_bounds": self.claimed_bounds,
            "warrants_relied_upon": list(self.warrants_relied_upon),
            "weakest_status": self.weakest_status.value,
            "n_findings": len(self.findings),
            "unchecked": list(self.unchecked),
            "enqueue_for_validation": list(self.enqueue_for_validation),
        }


def _most_restrictive(actions: Sequence[Action]) -> Action:
    return max(actions, key=lambda a: RESTRICTIVENESS[a])


def compose(
    verdicts: Sequence[DetectorVerdict],
    *,
    conservative_default: Action = Action.ESCALATE,
) -> ComposedDecision:
    """Compose several detectors' verdicts into one decision.

    Implements the four cases in ``DECISIONS.md`` 088 plus the overriding rule.
    Written for two detectors and correct for any number, because the cases are
    stated over *sets* of verdicts rather than over a pair.

    Args:
        verdicts: One per detector that ran.
        conservative_default: What the profile does when it cannot rely on a
            measured claim.

    Returns:
        A :class:`ComposedDecision`.

    Raises:
        ValueError: If no detector ran at all, which is a caller error rather
            than a policy outcome — a decision about zero detectors is not a
            conservative decision, it is an empty one.
    """
    if not verdicts:
        raise ValueError(
            "compose() needs at least one detector verdict. Zero detectors is a "
            "caller error, not a conservative outcome."
        )

    relied = [v for v in verdicts if v.relied_upon]
    findings = tuple(v.finding for v in verdicts if v.finding is not None)
    bounds = {
        v.detector_id: v.warrant.claimed_bounds() for v in relied if v.warrant
    }
    unchecked: list[str] = []

    for verdict in verdicts:
        if verdict.status is WarrantStatus.REFUSED:
            unchecked.append(
                f"{verdict.detector_id}: REFUSED on this envelope — measured "
                "here and failed its controls, so it contributes no finding and "
                "no bound. Out of service until a human revalidates."
            )
        elif verdict.status is not WarrantStatus.VALID:
            unchecked.append(
                f"{verdict.detector_id}: {verdict.status.value} on this "
                "envelope — it ran, but nothing here says what its output is "
                "worth, so no bound is claimed for it."
            )

    # The overriding rule: nothing valid means nothing claimable. Two
    # unvalidated detectors do not add up to one validated one.
    if not relied:
        return ComposedDecision(
            action=conservative_default,
            rule="no-valid-warrant",
            reason=(
                "no detector holds a valid warrant on this envelope, so the "
                "profile's conservative default applies and no bound is claimed"
            ),
            claimed_bounds={},
            warrants_relied_upon=(),
            weakest_status=WarrantStatus.UNVALIDATED,
            findings=findings,
            unchecked=tuple(unchecked),
            enqueue_for_validation=tuple(
                v.detector_id for v in verdicts if v.status is WarrantStatus.UNVALIDATED
            ),
        )

    relied_ids = tuple(v.warrant.warrant_id for v in relied if v.warrant)
    firing_valid = [v for v in relied if v.fired]
    firing_unvalidated = [
        v for v in verdicts if v.fired and v.status is WarrantStatus.UNVALIDATED
    ]

    # Case 4 takes precedence over 1 and 2 when it applies: an unvalidated
    # detector that fires is information of unknown quality, which the profile
    # handles with its conservative default rather than by ignoring it (as it
    # would a REFUSED one) or trusting it (as it would a VALID one).
    if firing_unvalidated:
        names = ", ".join(v.detector_id for v in firing_unvalidated)
        candidates = [v.action for v in firing_valid] + [conservative_default]
        action = _most_restrictive(candidates)
        return ComposedDecision(
            action=action,
            rule="valid-plus-unvalidated-fired",
            reason=(
                f"{names} fired but holds no warrant on this envelope, so its "
                "finding is carried without a bound and the profile's "
                "conservative default applies. UNVALIDATED is not REFUSED: "
                "nothing here says the output is wrong, only that nobody has "
                "measured it."
            ),
            claimed_bounds=bounds,
            warrants_relied_upon=relied_ids,
            weakest_status=WarrantStatus.VALID,
            findings=findings,
            unchecked=tuple(unchecked),
            enqueue_for_validation=tuple(v.detector_id for v in firing_unvalidated),
        )

    if not firing_valid:
        return ComposedDecision(
            action=Action.ALLOW,
            rule="none-fired",
            reason="no detector holding a valid warrant flagged this input",
            claimed_bounds=bounds,
            warrants_relied_upon=relied_ids,
            weakest_status=WarrantStatus.VALID,
            findings=findings,
            unchecked=tuple(unchecked),
        )

    action = _most_restrictive([v.action for v in firing_valid])
    silent = [v.detector_id for v in relied if not v.fired]

    if len(firing_valid) == len(relied):
        rule, reason = (
            "all-valid-agree",
            "every detector holding a valid warrant flagged this input; the "
            "most restrictive action applies. Not a vote — these detectors look "
            "for different things, so the response has to satisfy both.",
        )
    else:
        rule, reason = (
            "valid-disagree",
            "%s flagged this input; %s did not. The flagging detector's action "
            "applies. A detector that does not fire on something outside what "
            "it detects is working correctly, and a correct silence does not "
            "cancel a correct finding."
            % (
                ", ".join(v.detector_id for v in firing_valid),
                ", ".join(silent),
            ),
        )

    return ComposedDecision(
        action=action,
        rule=rule,
        reason=reason,
        claimed_bounds=bounds,
        warrants_relied_upon=relied_ids,
        weakest_status=WarrantStatus.VALID,
        findings=findings,
        unchecked=tuple(unchecked),
    )
