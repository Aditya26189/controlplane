"""Builders for record fixtures.

Not in ``conftest.py`` as fixtures because most tests need *several* variants of
the same record — two warrants on two envelopes, a certificate with and without
a valid warrant — and a pytest fixture returns one thing. These are plain
functions with defaults, so a test overrides only the field it is about and the
reader can see what that field is.

Everything here is synthetic and offline. Nothing downloads, nothing needs a
GPU, and no real personal data appears anywhere.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from src.model import (
    AccessTier,
    Action,
    Category,
    Certificate,
    ControlResult,
    DistributionEnvelope,
    EnvelopeFeature,
    EnvelopeMatchResult,
    EnvelopeState,
    Finding,
    Metric,
    MetricKind,
    OperatingPoint,
    Resolution,
    Severity,
    Span,
    Warrant,
    WarrantKey,
    WarrantMetrics,
    WarrantStatus,
    utc_now,
)

#: The five controls of SPEC.md §2.1, all passing with room to spare.
PASSING_CONTROLS = (
    ControlResult(
        "padding_fault", True, 0.9997, "cosine >= 0.999 on correct padding", 0.0007,
        "right-padded variant rejected as required",
    ),
    ControlResult("label_shuffle", True, 0.497, "AUROC in [0.45, 0.55]", 0.047),
    ControlResult("null_feature", True, 0.503, "AUROC in [0.45, 0.55]", 0.047),
    ControlResult("canary", True, 1.0, "recall == 1.0 on canary-20", 0.0),
    ControlResult("determinism", True, 0.0, "bit-identical across two runs", 0.0),
)


def failing_controls(which: str = "label_shuffle") -> tuple[ControlResult, ...]:
    """The five controls with one deliberately failed.

    Used to assert invariant 3: a failed control refuses the warrant, and no
    argument anywhere promotes it back.
    """
    return tuple(
        ControlResult(
            c.control, False, 0.71, c.expected, -0.16, "signal survived permutation"
        )
        if c.control == which
        else c
        for c in PASSING_CONTROLS
    )


def make_operating_point(
    operating_point_id: str = "P-conservative",
    detector_id: str = "probe-qwen2.5-7b-L23",
    threshold: float = 0.62,
    selected_on: str = "validation",
) -> OperatingPoint:
    """A threshold chosen on validation, as every threshold must be."""
    return OperatingPoint(
        operating_point_id=operating_point_id,
        detector_id=detector_id,
        threshold=threshold,
        selected_on=selected_on,
        objective="weighted_error(w_fpr_benign=50, w_fnr=5, w_fpr_hard_negative=2)",
        target_flag_rate=0.05,
    )


def make_envelope(
    eval_set_id: str = "triviaqa-600",
    envelope_id: Optional[str] = None,
    mean_tokens: float = 128.0,
) -> DistributionEnvelope:
    """A reference distribution with one binned token-length feature."""
    return DistributionEnvelope(
        envelope_id=envelope_id or f"sha256:{eval_set_id}",
        eval_set_id=eval_set_id,
        n_reference=600,
        features=(
            EnvelopeFeature(
                name="token_length",
                bin_edges=(0.0, 64.0, 128.0, 256.0, 512.0),
                bin_probabilities=(0.2, 0.4, 0.3, 0.1),
                mean=mean_tokens,
                std=51.0,
            ),
        ),
    )


def make_metrics(
    recall: float = 0.1416,
    precision: float = 0.285,
    flag_rate: float = 0.0148,
    confirmed_errors: int = 850,
) -> WarrantMetrics:
    """Measured bounds shaped like the Round 1 operating point.

    Precision and recall are both present because :class:`WarrantMetrics`
    requires both — invariant 5 is unconstructible to violate, not merely
    discouraged.
    """
    boot = "bootstrap-percentile-1000"
    return WarrantMetrics(
        auroc=Metric("auroc", 0.8551, MetricKind.ESTIMATED, 600, 0.8203, 0.8874, 0.95, "ratio", boot),
        recall=Metric("recall", recall, MetricKind.ESTIMATED, 600, recall - 0.04, recall + 0.05, 0.95, "rate", boot),
        precision=Metric("precision", precision, MetricKind.ESTIMATED, 600, precision - 0.045, precision + 0.045, 0.95, "rate", boot),
        flag_rate=Metric("flag_rate", flag_rate, MetricKind.ESTIMATED, 600, flag_rate - 0.003, flag_rate + 0.003, 0.95, "rate", boot),
        confirmed_errors=Metric("confirmed_errors", confirmed_errors, MetricKind.EXACT, 2984, unit="count"),
    )


def make_warrant(
    detector_id: str = "probe-qwen2.5-7b-L23",
    eval_set_id: str = "triviaqa-600",
    operating_point_id: str = "P-conservative",
    status: WarrantStatus = WarrantStatus.VALID,
    controls: tuple[ControlResult, ...] = PASSING_CONTROLS,
    issued_at: Optional[datetime] = None,
    ttl_hours: int = 24,
    status_reason: Optional[str] = None,
    validation_run_id: str = "run-0001",
    kappa: Optional[float] = 0.81,
) -> Warrant:
    """A warrant with every field populated, defaulting to valid and fresh."""
    issued = issued_at or utc_now()
    key = WarrantKey(detector_id, operating_point_id, eval_set_id)
    return Warrant(
        warrant_id=Warrant.compute_id(key, "1.0.0+ab12cd34", validation_run_id),
        detector_id=detector_id,
        detector_version="1.0.0+ab12cd34",
        operating_point=make_operating_point(operating_point_id, detector_id),
        eval_set_id=eval_set_id,
        validation_run_id=validation_run_id,
        issued_at=issued,
        expires_at=issued + timedelta(hours=ttl_hours),
        metrics=make_metrics(),
        n_test=600,
        base_rate=0.1517,
        envelope=make_envelope(eval_set_id),
        controls=controls,
        access_tier=AccessTier.T1_ACTIVATIONS,
        status=status,
        kappa=kappa,
        status_reason=status_reason
        or (None if status is WarrantStatus.VALID else "set by test factory"),
    )


def make_finding(
    finding_id: str = "F-1",
    detector_id: str = "probe-qwen2.5-7b-L23",
    category: Category = Category.HALLUCINATION,
    severity: Severity = Severity.HIGH,
    warrant_id: Optional[str] = None,
) -> Finding:
    """One finding with a full-range evidence span."""
    return Finding(
        finding_id=finding_id,
        detector_id=detector_id,
        detector_version="1.0.0+ab12cd34",
        category=category,
        severity=severity,
        confidence=0.73,
        evidence_spans=(Span(0, 42, "the whole response", "response"),),
        access_tier=AccessTier.T1_ACTIVATIONS,
        latency_ms=1.8,
        warrant_id=warrant_id,
    )


def make_envelope_match(
    envelope_id: str = "sha256:triviaqa-600",
    state: EnvelopeState = EnvelopeState.INSIDE,
    max_psi: float = 0.04,
    n_window: int = 200,
) -> EnvelopeMatchResult:
    """Where a window of traffic sat relative to an envelope."""
    if state is EnvelopeState.INSUFFICIENT_DATA:
        return EnvelopeMatchResult(
            envelope_id=envelope_id,
            state=state,
            psi_by_feature={},
            max_psi=0.0,
            driving_feature="",
            n_window=n_window,
        )
    return EnvelopeMatchResult(
        envelope_id=envelope_id,
        state=state,
        psi_by_feature={"token_length": max_psi, "script_mix": max_psi / 4},
        max_psi=max_psi,
        driving_feature="token_length",
        n_window=n_window,
        mmd_p_value=0.41,
    )


def make_certificate(
    certificate_id: str = "C-0001",
    session_id: str = "S-0001",
    request_id: str = "R-0001",
    warrant: Optional[Warrant] = None,
    action: Action = Action.ALLOW,
    weakest: Optional[WarrantStatus] = None,
    findings: Optional[tuple[Finding, ...]] = None,
    timestamp: Optional[datetime] = None,
) -> Certificate:
    """A certificate quoting a warrant's bounds, or none if unwarranted."""
    used = findings if findings is not None else (make_finding(
        warrant_id=warrant.warrant_id if warrant else None
    ),)
    status = weakest or (
        warrant.status if warrant is not None else WarrantStatus.UNVALIDATED
    )
    return Certificate(
        certificate_id=certificate_id,
        request_id=request_id,
        session_id=session_id,
        timestamp=timestamp or utc_now(),
        findings=used,
        resolution=Resolution(
            action=action,
            policy_version="3.1",
            policy_hash="sha256:policy-3-1",
            triggering_finding_ids=()
            if action is Action.ALLOW
            else tuple(f.finding_id for f in used),
            rule_id=None if action is Action.ALLOW else "R-irreversible-needs-warrant",
            rationale="" if action is Action.ALLOW else "no valid warrant backs this action",
        ),
        warrants_relied_upon=(warrant.warrant_id,) if warrant else (),
        weakest_warrant_status=status,
        claimed_bounds=warrant.claimed_bounds() if warrant else {},
        envelope_match=make_envelope_match(
            warrant.envelope.envelope_id if warrant else "sha256:unknown"
        ),
        access_tier_available=AccessTier.T1_ACTIVATIONS,
        unchecked=("retrieval grounding — no corpus configured for this request",),
    )
