"""The certificate a drift response leaves behind. ``SPEC.md`` §1.4, §5.3.

The Phase 5 gate ends with *"and writes a certificate explaining all of it"*,
and that clause is doing more work than it looks like it is. A revocation that
changes an internal status is invisible; a revocation that emits a chained
record saying **what was revoked, on what evidence, what was routed to instead,
and what is now claimable** is the thing a compliance reader can audit six
months later.

Three shapes, and the rules separating them are enforced by
:class:`~controlplane.model.certificate.Certificate` itself rather than here:

* **still usable** — ``VALID`` or ``STALE``. The warrant is cited and its bounds
  are claimed, widened when stale.
* **revoked and routed** — the *replacement* warrant is cited and **its** bounds
  are claimed. Never the revoked one's. This is the whole point of the matrix.
* **revoked and unrouted** — nothing is cited and nothing is claimed. A system
  with no warrant on this envelope must say so, not quietly keep quoting the
  numbers it just lost.

**The policy stamp is honest about what decided.** ``Resolution`` requires a
policy version and hash, and in Phase 5 there is no policy bundle — the ladder
decided. So the stamp names the ladder and hashes its actual configuration (the
two PSI bands, the window minimum, the widening factor). Phase 7 replaces it
with the bundle's own version and content hash, and the field does not change
shape when it does.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Optional

from ..model.certificate import Certificate, Resolution
from ..model.enums import Action, EnvelopeState, WarrantStatus
from ..model.findings import EnvelopeMatchResult
from ..model.warrant import Warrant
from .response import DriftResponse

__all__ = ["LADDER_POLICY_VERSION", "certify_drift_response", "ladder_policy_hash"]

#: Version of the decision procedure that produced these certificates. Not a
#: policy bundle: Phase 5 has no policy engine, and stamping a version that
#: implies one would misdate every certificate written before Phase 7.
LADDER_POLICY_VERSION = "drift-ladder-1"


def ladder_policy_hash(
    *, psi_stable: float, psi_significant: float, window_size: int, widen_factor: float
) -> str:
    """Content hash of the configuration the ladder actually ran with.

    The version string is what a human cites; the hash is what makes the
    citation checkable. Two certificates stamped ``drift-ladder-1`` with
    different hashes were produced under different bands, and a reader who
    cannot see that cannot replay either decision.

    Args:
        psi_stable: Lower band edge.
        psi_significant: Upper band edge.
        window_size: Requests required before any verdict.
        widen_factor: Multiplier applied to a ``STALE`` warrant's half-widths.

    Returns:
        ``"sha256:"`` followed by the digest.
    """
    body = json.dumps(
        {
            "psi_stable": psi_stable,
            "psi_significant": psi_significant,
            "window_size": window_size,
            "widen_factor": widen_factor,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def _envelope_match(response: DriftResponse, envelope_id: str) -> EnvelopeMatchResult:
    """Render the verdict as the record every certificate carries.

    Args:
        response: The drift response.
        envelope_id: The envelope the window was scored **against** — the
            warrant's own reference, not the one traffic turned out to be in.

    Returns:
        An :class:`EnvelopeMatchResult`.
    """
    verdict = response.transition.verdict
    psi_by_feature = {name: r.psi for name, r in verdict.per_feature.items()}

    # An ``EnvelopeMatchResult`` refuses a verdict that names no PSI, which is
    # correct: a rung with no numbers under it cannot be checked. Below the
    # window minimum there legitimately are none, and that state is the one
    # exception the record itself allows.
    if not psi_by_feature or verdict.driver is None:
        return EnvelopeMatchResult(
            envelope_id=envelope_id,
            state=EnvelopeState.INSUFFICIENT_DATA,
            psi_by_feature={},
            max_psi=0.0,
            driving_feature="",
            n_window=verdict.n_observed,
        )

    return EnvelopeMatchResult(
        envelope_id=envelope_id,
        state=verdict.state,
        psi_by_feature=psi_by_feature,
        max_psi=max(psi_by_feature.values()),
        driving_feature=verdict.driver,
        n_window=verdict.n_observed,
        # No multivariate check ran. Left None rather than defaulted to a
        # passing p-value: MMD is declared in SPEC.md §5.2 and is not
        # implemented (DECISIONS.md 072), and a 1.0 here would read as a test
        # that ran and found nothing.
        mmd_p_value=None,
    )


def _unchecked(response: DriftResponse, warrant: Warrant) -> tuple[str, ...]:
    """What this certificate is *not* in a position to say.

    The honest half of the record. Everything here is a limit a reader would
    otherwise have to infer from an absence, and an absence is exactly what
    nobody notices.

    Args:
        response: The drift response.
        warrant: The warrant under observation.

    Returns:
        One string per limit, for ``Certificate.unchecked``.
    """
    verdict = response.transition.verdict
    notes: list[str] = [
        "multivariate drift (MMD on embeddings) — declared in SPEC.md §5.2, not "
        "implemented; this verdict is per-feature PSI only (DECISIONS.md 072)",
    ]

    if verdict.unobserved:
        notes.append(
            "envelope features carried no values in this window and were not "
            "scored as stable: " + ", ".join(sorted(verdict.unobserved))
        )

    driving = verdict.per_feature.get(verdict.driver) if verdict.driver else None
    if driving is not None and driving.driven_by_smoothing:
        notes.append(
            "the PSI magnitude on %s is set by the smoothing epsilon (%d of its "
            "bins were empty against a non-empty reference), so the finding is "
            "that traffic left the reference support — not that the shift "
            "measured that size" % (verdict.driver, driving.bins_smoothed)
        )

    if response.transition.needs_routing and not response.can_certify:
        notes.append(
            "everything this detector was warranted for on %s: the warrant is "
            "revoked and no other detector holds one on the envelope the "
            "traffic is in" % warrant.eval_set_id
        )

    if response.transition.to_status is WarrantStatus.STALE:
        notes.append(
            "whether the widened bounds cover the new distribution — the "
            "widening factor is declared policy, not a measured calibration "
            "between PSI and interval width"
        )

    return tuple(notes)


def certify_drift_response(
    response: DriftResponse,
    warrant: Warrant,
    *,
    certificate_id: str,
    request_id: str,
    session_id: str,
    live_envelope_id: str,
    timestamp: datetime,
    policy_hash: str,
    rule_id: Optional[str] = None,
) -> Certificate:
    """Turn a drift response into the record that explains it.

    Args:
        response: What the ladder and the matrix decided.
        warrant: The warrant that was under observation.
        certificate_id: Unique id for this record.
        request_id: The request that triggered the verdict.
        session_id: Its session. Required — the store must be queryable by
            session (DPDP Rule 6).
        live_envelope_id: The envelope traffic is **actually** in. Recorded
            separately from the one scored against, because "measured on
            triviaqa-600, running on triviaqa-longctx-600" is the entire finding
            and collapsing the two hides it.
        timestamp: Decision time, tz-aware.
        policy_hash: From :func:`ladder_policy_hash`.
        rule_id: The ladder rung that fired, if the caller names one.

    Returns:
        An unsealed :class:`Certificate`. Sealing is the ledger's job — a record
        that hashed itself could be re-hashed after an edit and still verify.
    """
    transition = response.transition
    routing = response.routing

    if transition.to_status is WarrantStatus.REFUSED:
        cited: tuple[str, ...] = ()
        bounds: dict = {}
        weakest = WarrantStatus.REFUSED
        action = Action.ESCALATE
    elif not transition.needs_routing:
        cited = (warrant.warrant_id,)
        bounds = dict(response.claimed_bounds)
        weakest = transition.to_status
        action = Action.ALLOW if weakest is WarrantStatus.VALID else Action.ESCALATE
    elif routing is not None and routing.routed and routing.warrant is not None:
        # Routed. The replacement's id and the replacement's bounds — citing the
        # revoked warrant here would reintroduce exactly the unbacked claim the
        # revocation just prevented.
        cited = (routing.warrant.warrant_id,)
        bounds = dict(response.claimed_bounds)
        weakest = routing.warrant.status
        action = routing.action
    else:
        cited = ()
        bounds = {}
        weakest = WarrantStatus.REVOKED
        action = routing.action if routing is not None else Action.ESCALATE

    rationale = transition.reason
    if routing is not None:
        rationale = "%s %s" % (rationale, routing.reason)

    resolution = Resolution(
        action=action,
        policy_version=LADDER_POLICY_VERSION,
        policy_hash=policy_hash,
        # No findings: this certificate is about what the *detector's numbers*
        # are worth on this traffic, not about any content in one response. A
        # triggering-finding citation would be fabricated, so the trigger is
        # declared as what it actually was — the envelope verdict.
        triggering_finding_ids=(),
        triggered_by="envelope:%s" % transition.envelope_state.value,
        rule_id=rule_id or ("ladder-%s" % transition.envelope_state.value),
        rationale=rationale,
    )

    return Certificate(
        certificate_id=certificate_id,
        request_id=request_id,
        session_id=session_id,
        timestamp=timestamp,
        findings=(),
        resolution=resolution,
        warrants_relied_upon=cited,
        weakest_warrant_status=weakest,
        claimed_bounds=bounds,
        envelope_match=_envelope_match(response, warrant.envelope.envelope_id),
        access_tier_available=warrant.access_tier,
        unchecked=_unchecked(response, warrant)
        + (
            "the live envelope is %r; the bounds above were measured on %r"
            % (live_envelope_id, warrant.eval_set_id),
        ),
    )
