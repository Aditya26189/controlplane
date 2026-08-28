"""Certificates: the bounded, falsifiable assertion left behind by a decision.

``SPEC.md`` §1.4. A certificate is what separates this system from a detector
with a dashboard. It does not say *"this response is safe"*. It says:

    checked with these detectors, at these measured bounds, on this envelope,
    under this policy version, and here is what I could not check.

``claimed_bounds`` is what makes liability bounded. An assertion with an
interval and an ``n`` can be shown to be wrong; *"this is safe"* cannot, which
is why nobody who says it can be held to it.

The chain fields (``prev_certificate_hash``, ``self_hash``) are assigned by the
store when the record is appended, not by the caller — a record that hashed
itself could be re-hashed after an edit, and the chain would still verify.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any, Optional

from .enums import AccessTier, Action, WarrantStatus
from .findings import EnvelopeMatchResult, Finding

__all__ = [
    "Certificate",
    "CertificateError",
    "Resolution",
    "UNSEALED",
]

#: Placeholder occupying the chain fields before the store seals a record. Not
#: an empty string: an empty hash could be mistaken for "hashed to nothing",
#: whereas this cannot be read as a hash by anything.
UNSEALED = "UNSEALED"


class CertificateError(ValueError):
    """Raised when a certificate would assert something it cannot support."""


@dataclasses.dataclass(frozen=True)
class Resolution:
    """What policy decided, under which version, and because of what.

    Any action other than ``ALLOW`` must name what triggered it. An action
    nobody can trace to a trigger is an action nobody can appeal, and the appeal
    path is most of what a compliance reader is looking for.

    **There are two kinds of trigger, not one.** Content triggers are findings,
    and ``triggering_finding_ids`` carries them. Warrant-level triggers are not:
    when a revoked warrant escalates a request, nothing was found *in* that
    request — the system stopped being able to say what its checks were worth.
    Forcing that through ``triggering_finding_ids`` would mean inventing a
    finding with a content category attached to it, which is a fabrication in
    the one record that exists to prevent fabrications. ``triggered_by`` names
    the non-content trigger instead, e.g. ``"envelope:SIGNIFICANT_SHIFT"``.
    ``DECISIONS.md`` 073.

    Args:
        action: The decision.
        policy_version: Version of the bundle that made it. Stamped on every
            certificate so a decision can be replayed against the rules that
            were live at the time, not the rules that are live now.
        policy_hash: Content hash of the bundle. The version string is what
            humans cite; the hash is what makes the citation checkable.
        triggering_finding_ids: Findings that caused the action.
        rule_id: The specific rule that fired.
        rationale: Human-readable reason, shown to whoever is affected.
        triggered_by: A non-content trigger, when no finding caused the action.
            Satisfies the traceability requirement in place of
            ``triggering_finding_ids``; exactly one of the two is needed.
    """

    action: Action
    policy_version: str
    policy_hash: str
    triggering_finding_ids: tuple[str, ...] = ()
    rule_id: Optional[str] = None
    rationale: str = ""
    triggered_by: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.policy_version or not self.policy_hash:
            raise CertificateError(
                "a resolution must name its policy version and hash; a decision "
                "that cannot be replayed against the rules that made it is not "
                "auditable"
            )
        if (
            self.action is not Action.ALLOW
            and not self.triggering_finding_ids
            and not self.triggered_by
        ):
            raise CertificateError(
                f"action {self.action.value} must name what triggered it — "
                "finding id(s), or a warrant-level trigger in triggered_by. An "
                "action nobody can trace to a trigger is an action nobody can "
                "appeal."
            )
        if self.action is not Action.ALLOW and not self.rationale:
            raise CertificateError(
                f"action {self.action.value} must carry a rationale readable by "
                "the person it affects"
            )


@dataclasses.dataclass(frozen=True)
class Certificate:
    """The record of one request's checks, decision and claimed bounds.

    Args:
        certificate_id: Unique id.
        request_id: The request this describes.
        session_id: The session it belongs to. Indexed, because DPDP Rule 6
            requires the store be queryable by session.
        timestamp: When the decision was made, UTC.
        findings: Every finding produced, including those from detectors with no
            warrant. An unwarranted finding is still information; it just
            licenses no claim about how often it is right.
        resolution: What policy decided.
        warrants_relied_upon: Warrant ids whose bounds this certificate quotes.
        weakest_warrant_status: The weakest status among them, which is what the
            policy's first rule keys on. ``UNVALIDATED`` here means at least one
            detector had no warrant on this envelope.
        claimed_bounds: The falsifiable assertion. Empty only when nothing was
            warranted, in which case ``weakest_warrant_status`` says so.
        envelope_match: Which envelope this input landed in, and how far outside
            it sat. Present on every certificate: a bound quoted without the
            envelope it was measured on is the unbacked claim this project
            exists to refuse.
        access_tier_available: The deepest access available for this request.
            Recorded because the same detector at a different tier is a
            different claim.
        unchecked: What was *not* checked, in words. The honest half of the
            certificate, and the half a dashboard never shows.
        prev_certificate_hash: Assigned by the store on append.
        self_hash: Assigned by the store on append.

    Raises:
        CertificateError: If the certificate would assert something unsupported.
    """

    certificate_id: str
    request_id: str
    session_id: str
    timestamp: datetime
    findings: tuple[Finding, ...]
    resolution: Resolution
    warrants_relied_upon: tuple[str, ...]
    weakest_warrant_status: WarrantStatus
    claimed_bounds: dict[str, Any]
    envelope_match: EnvelopeMatchResult
    access_tier_available: AccessTier
    unchecked: tuple[str, ...] = ()
    prev_certificate_hash: str = UNSEALED
    self_hash: str = UNSEALED

    def __post_init__(self) -> None:
        if not self.certificate_id or not self.request_id or not self.session_id:
            raise CertificateError(
                "certificate_id, request_id and session_id are all required; the "
                "store must be queryable by session (DPDP Rule 6)"
            )
        if self.timestamp.tzinfo is None:
            raise CertificateError(
                "certificate timestamps carry an explicit UTC offset; a naive "
                "timestamp means a different instant depending on where it is read"
            )
        finding_ids = {f.finding_id for f in self.findings}
        dangling = sorted(set(self.resolution.triggering_finding_ids) - finding_ids)
        if dangling:
            raise CertificateError(
                f"resolution cites finding(s) {dangling} that are not on this "
                "certificate; the decision and its evidence have come apart"
            )
        if self.weakest_warrant_status.can_be_relied_upon:
            if not self.warrants_relied_upon:
                raise CertificateError(
                    "weakest_warrant_status is VALID but no warrant is cited. A "
                    "certificate claiming validated bounds must name the warrants "
                    "supplying them."
                )
            if not self.claimed_bounds:
                raise CertificateError(
                    "weakest_warrant_status is VALID but claimed_bounds is empty. "
                    "A warrant that licenses no bound is a badge, not a warrant."
                )
        if self.warrants_relied_upon and not self.claimed_bounds:
            raise CertificateError(
                "warrants are cited but no bounds are claimed. Either the bounds "
                "were dropped in rendering or the citation is decorative; both "
                "reintroduce the unbacked claim this record exists to prevent."
            )

    # -- sealing ------------------------------------------------------------ #

    @property
    def is_sealed(self) -> bool:
        """Whether the store has assigned this record its place in the chain."""
        return self.self_hash != UNSEALED and self.prev_certificate_hash != UNSEALED

    def sealed_with(self, prev_hash: str, self_hash: str) -> "Certificate":
        """Return the same record with its chain position assigned.

        Only the store calls this. A record that hashed itself could be
        re-hashed after an edit and the chain would still verify, which would
        make the chain decorative.

        Raises:
            CertificateError: If the record has already been sealed. Re-sealing
                is how a tampered record would be laundered back into the chain.
        """
        if self.is_sealed:
            raise CertificateError(
                f"certificate {self.certificate_id} is already sealed. Re-sealing "
                "is how an edited record would be laundered back into the chain."
            )
        if not prev_hash or not self_hash:
            raise CertificateError("both chain hashes are required to seal a record")
        return dataclasses.replace(
            self, prev_certificate_hash=prev_hash, self_hash=self_hash
        )

    def unsealed(self) -> "Certificate":
        """Return the record as it was before sealing.

        The chain hash is taken over the *unsealed* body: a hash cannot cover a
        field that contains it. Verification re-derives the hash from this form.
        """
        return dataclasses.replace(
            self, prev_certificate_hash=UNSEALED, self_hash=UNSEALED
        )

    # -- querying ----------------------------------------------------------- #

    def categories_accessed(self) -> tuple[str, ...]:
        """Distinct finding categories on this certificate, sorted.

        Denormalised into a store column because DPDP Rule 6 requires the log be
        queryable by personal-data category accessed, and scanning every
        certificate body to answer that would make the query unusable at a
        year's retention.
        """
        return tuple(sorted({f.category.value for f in self.findings}))

    def detector_versions(self) -> tuple[str, ...]:
        """Distinct ``detector_id@version`` strings, sorted.

        Also denormalised for query: "which requests were decided by the
        detector version we have just found a fault in?" is the question asked
        the morning after an incident.
        """
        return tuple(
            sorted({f"{f.detector_id}@{f.detector_version}" for f in self.findings})
        )
