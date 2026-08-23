"""Routing on the matrix. ``SPEC.md`` §3.2.

Drift fires, and the question is **not** *"which tier is next?"* — it is *"which
detector holds a valid warrant on the envelope I am actually in?"*. Those differ,
and the difference is the whole design. Downgrading a tier without re-checking
the envelope replaces one unwarranted claim with another, which is why
:func:`route` looks up a column of the matrix rather than walking a ladder.

Three states, three behaviours (``CLAUDE.md`` invariant 2), and this module is
where they stop being an enum and start being different code paths:

* ``VALID`` — route there and **adopt that warrant's bounds**, which are usually
  wider than the ones just lost. The certificate then claims what the new
  detector can support, not what the old one used to.
* ``REFUSED`` — the detector is out of service on this envelope until a human
  revalidates. It is never a routing candidate, and it is not the same as never
  having tried.
* ``UNVALIDATED`` — no claim is available. Take the profile's conservative
  default, **enqueue the cell for validation**, and log. This is how the matrix
  self-populates from live traffic.

If a column is empty of valid warrants, :func:`route` returns no warrant and the
profile's conservative default applies. That is a refusal to certify, not a
failure to run: the request is still handled, and the certificate says the
system could not say what its checks were worth.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Optional, Sequence

from ..config import Config, ProfileConfig
from ..model import Action, Warrant, WarrantKey, WarrantMetrics, WarrantStatus

__all__ = ["Profile", "RoutingDecision", "route"]

_LOG = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Profile:
    """A deployment profile's requirements, as routing sees them.

    Wraps :class:`~src.config.ProfileConfig` so routing depends on the shape of
    a profile rather than on the config module, and so a test can build one
    without a config file.

    Args:
        name: Profile name from ``config.profiles``.
        min_recall: Recall the profile requires. Compared against the interval's
            **lower bound**, not the point estimate: a profile asking for "at
            least 25% recall" is asking for a guarantee, and 0.26 with a lower
            bound of 0.18 does not supply one.
        max_fpr: Maximum false-positive rate, compared against the **upper**
            bound by the same argument reversed.
        inline_budget_ms: Latency budget. A warrant whose detector cannot answer
            inside it is not eligible however good its numbers are.
        conservative_default: What to do when nothing holds a warrant here.
    """

    name: str
    min_recall: float
    max_fpr: float
    inline_budget_ms: int
    conservative_default: Action = Action.ESCALATE

    @classmethod
    def from_config(cls, config: Config, name: str) -> "Profile":
        """Build from ``config.profiles[name]``.

        Raises:
            KeyError: Naming the profiles that do exist, since the usual cause
                is a typo in a policy bundle.
        """
        if name not in config.profiles:
            raise KeyError(
                f"no profile {name!r}; config declares {sorted(config.profiles)}"
            )
        profile: ProfileConfig = config.profiles[name]
        return cls(
            name=name,
            min_recall=profile.min_recall,
            max_fpr=profile.max_fpr,
            inline_budget_ms=profile.inline_budget_ms,
        )

    def accepts(self, metrics: WarrantMetrics) -> tuple[bool, str]:
        """Whether this profile can run on these measured bounds.

        Returns:
            ``(accepted, reason)``. The reason is populated on rejection and
            names the bound that missed, because "profile suspended" is an alarm
            and "decision_support requires recall >= 0.50 and the warrant's
            lower bound is 0.31" is an explanation. Beat 4 shows the second.
        """
        if metrics.recall is None:
            return False, (
                f"{self.name} requires recall >= {self.min_recall}, and this "
                "envelope supports no recall claim at all (single-class)"
            )
        lower = metrics.recall.ci_low
        if lower is None or lower < self.min_recall:
            return False, (
                f"{self.name} requires recall >= {self.min_recall}; the warrant's "
                f"lower bound is {lower:.4f} (point estimate "
                f"{metrics.recall.value:.4f})"
            )
        fpr = metrics.fpr_hard_negatives
        if fpr is not None:
            upper = fpr.ci_high if fpr.ci_high is not None else fpr.value
            if upper > self.max_fpr:
                return False, (
                    f"{self.name} requires hard-negative FPR <= {self.max_fpr}; "
                    f"the warrant's upper bound is {upper:.4f}"
                )
        return True, ""


@dataclasses.dataclass(frozen=True)
class RoutingDecision:
    """Where a request was routed, and what may be claimed as a result.

    Args:
        envelope_id: The envelope the request landed in.
        profile: The profile that was asking.
        warrant: The warrant routed to, or None when nothing holds one here.
        action: What to do. The profile's conservative default when there is no
            warrant.
        claimed_bounds: What the certificate may assert. Empty when there is no
            warrant — **not** the previous detector's bounds, which is the
            failure this whole module exists to prevent.
        reason: Why this decision, in words.
        considered: Every candidate examined, with why each was or was not
            eligible. Present so a reader can see the ones that were rejected;
            a routing decision that only shows its winner is unauditable.
        enqueued_for_validation: Cells that have never been measured on this
            envelope. Logged and queued, which is how the matrix fills itself in.
        suspended_profile: True when a profile that would otherwise run cannot,
            because the available bounds fall below what it declared.
    """

    envelope_id: str
    profile: str
    warrant: Optional[Warrant]
    action: Action
    claimed_bounds: dict
    reason: str
    considered: tuple[tuple[str, str], ...] = ()
    enqueued_for_validation: tuple[WarrantKey, ...] = ()
    suspended_profile: bool = False

    @property
    def routed(self) -> bool:
        """Whether a warrant backs this decision."""
        return self.warrant is not None


def _rank(warrant: Warrant) -> tuple:
    """Order eligible warrants: tighter recall interval first, then higher recall.

    Tighter first because the point of routing is to keep making a *defensible*
    claim, and a narrow interval is a more useful claim than a high midpoint
    with a wide one. Recall breaks ties, and the detector id makes the order
    total so routing is deterministic.
    """
    recall = warrant.metrics.recall
    if recall is None:
        return (1, 0.0, 0.0, warrant.detector_id)
    width = (recall.ci_high or 0.0) - (recall.ci_low or 0.0)
    return (0, width, -recall.value, warrant.detector_id)


def route(
    matrix,
    envelope_id: str,
    profile: Profile,
    *,
    enqueue: Optional[Callable[[Sequence[WarrantKey]], None]] = None,
) -> RoutingDecision:
    """Find a detector that holds a valid warrant on this envelope.

    Args:
        matrix: A :class:`~src.matrix.matrix.WarrantMatrix`.
        envelope_id: The envelope the traffic is actually in — which is *not*
            necessarily the one the detector was measured on, and that gap is
            the reason this function exists.
        profile: What the caller requires.
        enqueue: Called with the unvalidated cells on this envelope. Defaults to
            logging them.

    Returns:
        A :class:`RoutingDecision`. Always returns one; there is no failure
        mode where a request goes unhandled, only ones where it goes unclaimed.
    """
    candidates = matrix.valid_warrants(envelope_id)
    considered: list[tuple[str, str]] = []

    eligible: list[Warrant] = []
    for warrant in candidates:
        accepted, why = profile.accepts(warrant.metrics)
        considered.append(
            (warrant.detector_id, "eligible" if accepted else why)
        )
        if accepted:
            eligible.append(warrant)

    # Cells measured here and refused are named too. "We tried this detector on
    # this traffic and it failed" is different information from "we never tried
    # it", and a reader of the decision should see which.
    for cell in matrix.cells_for_envelope(envelope_id):
        if cell.status is WarrantStatus.REFUSED:
            considered.append(
                (
                    cell.detector_id,
                    f"REFUSED on this envelope: {cell.reason or 'no reason recorded'}",
                )
            )
        elif cell.status in (WarrantStatus.STALE, WarrantStatus.REVOKED):
            considered.append(
                (cell.detector_id, f"{cell.status.value}: {cell.reason or ''}")
            )

    unvalidated = matrix.unvalidated_cells(envelope_id)
    if unvalidated:
        if enqueue is not None:
            enqueue(unvalidated)
        else:
            _LOG.info(
                "enqueued %d unvalidated cell(s) on %s: %s",
                len(unvalidated),
                envelope_id,
                [k.detector_id for k in unvalidated],
            )

    if eligible:
        best = sorted(eligible, key=_rank)[0]
        _LOG.info(
            "routed %s on %s for profile %s", best.detector_id, envelope_id, profile.name
        )
        return RoutingDecision(
            envelope_id=envelope_id,
            profile=profile.name,
            warrant=best,
            action=Action.ALLOW,
            claimed_bounds=best.claimed_bounds(),
            reason=(
                f"{best.detector_id} holds a valid warrant on {envelope_id} and "
                f"meets {profile.name}'s declared minimums"
            ),
            considered=tuple(considered),
            enqueued_for_validation=unvalidated,
        )

    # Something holds a warrant here, but not one this profile can run on. That
    # is a *suspension*, and it is a different event from having no warrant at
    # all: the system knows what it can prove and knows it is not enough.
    suspended = bool(candidates)
    reason = (
        f"no warrant on {envelope_id} meets {profile.name}'s declared minimums "
        f"(recall >= {profile.min_recall}, FPR <= {profile.max_fpr}); "
        f"{profile.name} is suspended on this envelope"
        if suspended
        else f"no valid warrant on envelope {envelope_id}"
    )
    _LOG.warning("%s; falling back to %s", reason, profile.conservative_default.value)
    return RoutingDecision(
        envelope_id=envelope_id,
        profile=profile.name,
        warrant=None,
        action=profile.conservative_default,
        claimed_bounds={},
        reason=reason,
        considered=tuple(considered),
        enqueued_for_validation=unvalidated,
        suspended_profile=suspended,
    )
