"""Findings, operating points and distribution envelopes.

A :class:`Finding` is what a detector says. It is deliberately weak: a score, a
category, a severity, and character offsets for the evidence. It makes no claim
about what that score is *worth* — that claim lives in the warrant its
``warrant_id`` points at, and ``warrant_id is None`` is an honest state meaning
"nobody has measured this detector on this kind of input", not an error.

An :class:`OperatingPoint` is the threshold half of a warrant's key. It records
where it was selected, and refuses to be selected on test.

A :class:`DistributionEnvelope` is the reference distribution a warrant's
numbers were measured inside, computed at validation time and stored *in* the
warrant (``SPEC.md`` §5.1). It is what makes drift detectable a year later
without re-running the validation.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

from .enums import AccessTier, Category, ConfidenceBand, EnvelopeState, Severity

__all__ = [
    "DistributionEnvelope",
    "EnvelopeFeature",
    "EnvelopeMatchResult",
    "Finding",
    "FindingError",
    "OperatingPoint",
    "Span",
]


class FindingError(ValueError):
    """Raised when a finding, operating point or envelope is self-inconsistent."""


@dataclasses.dataclass(frozen=True)
class Span:
    """A character range in the text a finding refers to.

    Character offsets rather than token offsets: the certificate is read by
    people and by systems that never saw the tokenizer, and a token index is
    uninterpretable to both.

    Args:
        start: Inclusive start offset.
        end: Exclusive end offset.
        text: The spanned text, carried so a certificate is interpretable after
            the original request has been deleted under retention policy.
        label: What this span is evidence of, e.g. ``"IN_AADHAAR"``.
    """

    start: int
    end: int
    text: str
    label: str = ""

    def __post_init__(self) -> None:
        if self.start < 0:
            raise FindingError(f"span start must be non-negative, got {self.start}")
        if self.end < self.start:
            raise FindingError(
                f"span end {self.end} precedes start {self.start}"
            )

    @property
    def length(self) -> int:
        """Length of the span in characters."""
        return self.end - self.start


@dataclasses.dataclass(frozen=True)
class Finding:
    """One detector's assessment of one request. ``SPEC.md`` §1.2.

    Findings never resolve conflicts with each other. Categories overlap by
    design — a fabricated detail about a named person emits both
    ``HALLUCINATION`` and ``PII`` — and it is policy's job to decide what to do
    when two detectors disagree. A detector that arbitrated would be making a
    decision nobody could audit.

    Args:
        finding_id: Unique within a certificate.
        detector_id: e.g. ``"probe-qwen2.5-7b-L23"``, ``"presidio-stock"``.
        detector_version: Semver plus a weights hash. Part of what a warrant is
            pinned to, so a silent model update cannot inherit an old warrant.
        category: What kind of problem. Overlapping by design.
        severity: The detector's own assessment, not a warranted claim.
        confidence: Raw detector score in [0, 1]. A score, not a probability of
            being right — that mapping is what the warrant supplies, and only on
            an envelope where it was measured.
        evidence_spans: Character offsets. Required: a finding without evidence
            cannot be explained to the person it affects, and explainability is
            the reason the store exists.
        access_tier: What depth of access produced this finding.
        latency_ms: Wall-clock cost, needed to check a profile's inline budget.
        warrant_id: The warrant backing this detector on this envelope, or None.
            **None is an honest state**, not an error: it means no one has
            measured this detector on this kind of input, and the policy's
            conservative default applies.
        confidence_band: Where the score sits relative to the operating point's
            measured bands, when an operating point is known.

    Raises:
        FindingError: If the finding could not be explained or is out of range.
    """

    finding_id: str
    detector_id: str
    detector_version: str
    category: Category
    severity: Severity
    confidence: float
    evidence_spans: tuple[Span, ...]
    access_tier: AccessTier
    latency_ms: float
    warrant_id: Optional[str] = None
    confidence_band: Optional[ConfidenceBand] = None

    def __post_init__(self) -> None:
        if not self.finding_id or not self.detector_id or not self.detector_version:
            raise FindingError(
                "finding_id, detector_id and detector_version are all required; a "
                "finding that cannot be attributed cannot be audited"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise FindingError(
                f"{self.detector_id}: confidence must be in [0, 1], got "
                f"{self.confidence}"
            )
        if self.latency_ms < 0:
            raise FindingError(
                f"{self.detector_id}: latency_ms must be non-negative, got "
                f"{self.latency_ms}"
            )
        if not self.evidence_spans:
            raise FindingError(
                f"{self.detector_id}: a finding must carry at least one evidence "
                "span. Character offsets are what make a finding explainable to "
                "the person it affects; a whole-response finding uses a "
                "full-range span rather than none."
            )

    @property
    def is_warranted(self) -> bool:
        """Whether any warrant backs this finding.

        Says nothing about whether that warrant is currently *valid* — the
        certificate resolves that, because validity depends on the envelope the
        request actually landed in.
        """
        return self.warrant_id is not None


@dataclasses.dataclass(frozen=True)
class OperatingPoint:
    """A threshold on a detector's score, and where it was chosen.

    The middle element of the warrant key. Recall at threshold τ₁ says nothing
    about recall at τ₂, so a warrant without an operating point is a curve
    rather than a claim.

    ``selected_on`` is asserted not to be the test split. Selecting a layer, a
    regularisation strength or a threshold on test is the first thing a reviewer
    checks, it inflates the headline number, and it is one line to do by
    accident (``CLAUDE.md``, "Silent failures").

    Args:
        operating_point_id: Stable id, e.g. ``"P-conservative"``.
        detector_id: The detector this threshold applies to.
        threshold: Score at or above which the detector fires.
        selected_on: Which split the threshold was chosen on. Must not be test.
        objective: How it was chosen, e.g. ``"weighted_error"`` or
            ``"flag_rate_budget=0.05"``. Recorded because a threshold without
            its objective cannot be re-derived.
        target_flag_rate: The budget aimed at, when one was. Never used in a
            downstream calculation — invariant 6's measured-versus-target trap
            lives here, and the measured rate is what every calculation uses.
    """

    operating_point_id: str
    detector_id: str
    threshold: float
    selected_on: str
    objective: str
    target_flag_rate: Optional[float] = None

    #: Splits a threshold may be selected on. "test" is absent deliberately.
    ALLOWED_SELECTION_SPLITS = ("validation", "train", "calibration")

    def __post_init__(self) -> None:
        if not self.operating_point_id or not self.detector_id:
            raise FindingError("operating_point_id and detector_id are required")
        if not math.isfinite(self.threshold):
            raise FindingError(
                f"{self.operating_point_id}: threshold must be finite, got "
                f"{self.threshold}"
            )
        if self.selected_on not in self.ALLOWED_SELECTION_SPLITS:
            raise FindingError(
                f"{self.operating_point_id}: selected_on must be one of "
                f"{list(self.ALLOWED_SELECTION_SPLITS)}, got {self.selected_on!r}. "
                "Selecting a threshold on test inflates the headline number and "
                "is the first thing a reviewer checks (CLAUDE.md, 'Silent "
                "failures')."
            )
        if not self.objective:
            raise FindingError(
                f"{self.operating_point_id}: objective is required. A threshold "
                "without the objective it optimised cannot be re-derived, which "
                "makes the operating point unreproducible."
            )
        if self.target_flag_rate is not None and not 0.0 < self.target_flag_rate < 1.0:
            raise FindingError(
                f"{self.operating_point_id}: target_flag_rate must be in (0, 1), "
                f"got {self.target_flag_rate}"
            )


@dataclasses.dataclass(frozen=True)
class EnvelopeFeature:
    """One dimension of the reference distribution, summarised for PSI.

    Stored as bin edges plus reference probabilities rather than as raw values,
    for two reasons: PSI is computed against a binned reference, and the raw
    inputs may contain personal data that must not be retained inside a warrant
    that lives for a year.

    Args:
        name: One of ``config.drift.features``.
        bin_edges: ``k + 1`` monotonically increasing edges.
        bin_probabilities: ``k`` reference proportions, summing to 1.
        mean: Reference mean, for reporting.
        std: Reference standard deviation, for reporting.
    """

    name: str
    bin_edges: tuple[float, ...]
    bin_probabilities: tuple[float, ...]
    mean: float
    std: float

    def __post_init__(self) -> None:
        if len(self.bin_edges) < 2:
            raise FindingError(
                f"envelope feature {self.name}: need at least two bin edges, got "
                f"{len(self.bin_edges)}"
            )
        if len(self.bin_probabilities) != len(self.bin_edges) - 1:
            raise FindingError(
                f"envelope feature {self.name}: {len(self.bin_edges)} edges imply "
                f"{len(self.bin_edges) - 1} bins, got "
                f"{len(self.bin_probabilities)} probabilities"
            )
        if any(b <= a for a, b in zip(self.bin_edges, self.bin_edges[1:])):
            raise FindingError(
                f"envelope feature {self.name}: bin edges must be strictly "
                "increasing"
            )
        if any(p < 0 for p in self.bin_probabilities):
            raise FindingError(
                f"envelope feature {self.name}: bin probabilities must be "
                "non-negative"
            )
        total = sum(self.bin_probabilities)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise FindingError(
                f"envelope feature {self.name}: bin probabilities sum to {total}, "
                "not 1"
            )
        if self.std < 0:
            raise FindingError(f"envelope feature {self.name}: std must be non-negative")


@dataclasses.dataclass(frozen=True)
class DistributionEnvelope:
    """The input distribution a warrant's numbers were measured inside.

    ``SPEC.md`` §5.1. Computed at validation time and stored *inside* the
    warrant, so that a year later the system can still answer "is this traffic
    like the traffic those numbers came from?" without re-running validation.

    ``envelope_id`` is the eval set's content hash. That identity is the whole
    design: an envelope violation is a property of the input distribution, so it
    invalidates every detector measured on that distribution at once
    (``CLAUDE.md`` invariant 1).

    Args:
        envelope_id: The eval set's content hash — the warrant key's third element.
        eval_set_id: Human-readable set name, e.g. ``"triviaqa-longctx-600"``.
        n_reference: How many items the reference distribution was computed from.
        features: One entry per configured drift feature.
    """

    envelope_id: str
    eval_set_id: str
    n_reference: int
    features: tuple[EnvelopeFeature, ...]

    def __post_init__(self) -> None:
        if not self.envelope_id or not self.eval_set_id:
            raise FindingError("envelope_id and eval_set_id are required")
        if self.n_reference <= 0:
            raise FindingError(
                f"envelope {self.eval_set_id}: n_reference must be positive, got "
                f"{self.n_reference}"
            )
        names = [f.name for f in self.features]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise FindingError(
                f"envelope {self.eval_set_id}: duplicate feature(s) {duplicates}"
            )

    def feature(self, name: str) -> EnvelopeFeature:
        """Look up one feature by name.

        Raises:
            FindingError: If the envelope does not carry that feature. A missing
                feature is a crash rather than a None, because a drift check
                that silently skips a dimension reports "stable" for a
                distribution it never looked at.
        """
        for feature in self.features:
            if feature.name == name:
                return feature
        raise FindingError(
            f"envelope {self.eval_set_id} carries no feature {name!r}; it has "
            f"{[f.name for f in self.features]}. A drift check that skips a "
            "dimension reports stability it did not measure."
        )


@dataclasses.dataclass(frozen=True)
class EnvelopeMatchResult:
    """Where a window of live traffic sits relative to a warrant's envelope.

    Written into every certificate so that a reader can tell not only what was
    claimed but whether the input was the kind of input the claim was measured
    on. ``SPEC.md`` §1.4, §5.2.

    Args:
        envelope_id: The envelope compared against.
        state: The rung of the revocation ladder this window sits on.
        psi_by_feature: PSI per feature over the window.
        max_psi: The largest, which is what the ladder keys on.
        driving_feature: Which feature produced ``max_psi``. Named because "PSI
            crossed 0.25" is an alarm and "token length crossed 0.25" is an
            explanation.
        n_window: Window size. Below ``config.drift.window_size`` the state is
            ``INSUFFICIENT_DATA`` and no verdict is issued.
        mmd_p_value: Permutation-test p-value for the multivariate check, when
            embeddings were available.
    """

    envelope_id: str
    state: EnvelopeState
    psi_by_feature: dict[str, float]
    max_psi: float
    driving_feature: str
    n_window: int
    mmd_p_value: Optional[float] = None

    def __post_init__(self) -> None:
        if self.n_window < 0:
            raise FindingError(f"n_window must be non-negative, got {self.n_window}")
        if self.max_psi < 0:
            raise FindingError(f"max_psi must be non-negative, got {self.max_psi}")
        if self.state is EnvelopeState.INSUFFICIENT_DATA:
            return
        if not self.psi_by_feature:
            raise FindingError(
                "an envelope verdict must name the PSI it was based on; a verdict "
                "with no per-feature values cannot be checked by a reader"
            )
        if self.driving_feature not in self.psi_by_feature:
            raise FindingError(
                f"driving_feature {self.driving_feature!r} is not among the "
                f"measured features {sorted(self.psi_by_feature)}"
            )
        observed_max = max(self.psi_by_feature.values())
        if not math.isclose(self.max_psi, observed_max, rel_tol=1e-9, abs_tol=1e-12):
            raise FindingError(
                f"max_psi {self.max_psi} does not match the largest per-feature "
                f"PSI {observed_max}"
            )
        if self.mmd_p_value is not None and not 0.0 <= self.mmd_p_value <= 1.0:
            raise FindingError(
                f"mmd_p_value must be in [0, 1], got {self.mmd_p_value}"
            )

    @property
    def is_inside(self) -> bool:
        """True only when the window is inside the envelope.

        ``INSUFFICIENT_DATA`` is deliberately not inside. Below the window
        minimum there is no verdict, and treating "we haven't looked yet" as
        "we looked and it's fine" is invariant 2's failure in a different
        costume.
        """
        return self.state is EnvelopeState.INSIDE
