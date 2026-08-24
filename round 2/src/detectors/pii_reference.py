"""A reference PII detector: patterns plus checksums, written by us.

**Not Presidio.** This is our own baseline, and it exists for two reasons.

First, it lets ``/validate`` run against the PII eval sets *now*, so those sets
are measured rather than merely built. A frozen eval set nobody has scored is a
file, not a benchmark.

Second, it is the honest floor for the Phase 8 comparison. When we report that
stock Presidio has near-zero recall on Indian identifiers, the obvious reply is
*"then anything would beat it"* — and the right answer is to show what "anything"
actually scores, built in an afternoon from published formats. If this detector
beats Presidio's stock configuration, that is a fact about Presidio's defaults,
not a claim about our cleverness. ``DECISIONS.md`` 008 requires all three
Presidio configurations be reported; this is the fourth line of that table.

**It is a detector, not a verdict.** It emits a score and evidence spans. What
that score is worth is a warrant's job, and on the obfuscated disclosure form it
is worth considerably less than on the verbatim one — which is exactly what the
per-form measurement will show.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Iterable, Optional, Sequence

import numpy as np

from ..model import AccessTier, Category, Severity, Span
from .identifiers_patterns import (
    AADHAAR_LOOSE,
    IFSC_PATTERN,
    PAN_PATTERN,
    PHONE_LOOSE,
    UPI_PATTERN,
    digits_only,
)

__all__ = ["PiiMatch", "ReferencePiiDetector"]

from ..evalsets.identifiers import verhoeff_is_valid


@dataclasses.dataclass(frozen=True)
class PiiMatch:
    """One identifier found in text.

    Args:
        kind: Entity type, using Presidio's names where they exist.
        text: The matched substring.
        start: Character offset, inclusive.
        end: Character offset, exclusive.
        confidence: How much the detector trusts this match. Checksum-validated
            matches score higher than pattern-only ones, which is the whole
            reason to validate a checksum.
        checksum_checked: Whether a checksum was applicable and applied.
    """

    kind: str
    text: str
    start: int
    end: int
    confidence: float
    checksum_checked: bool

    def to_span(self) -> Span:
        """Render as a :class:`~src.model.findings.Span` for a finding."""
        return Span(start=self.start, end=self.end, text=self.text, label=self.kind)


class ReferencePiiDetector:
    """Pattern-and-checksum detection of Indian identifiers.

    Args:
        validate_checksums: Whether to apply Verhoeff to Aadhaar candidates and
            the structural rule to PAN. Exposed so the *effect* of checksum
            validation is measurable rather than assumed — Phase 8 reports the
            configuration with and without.
        detect_obfuscated: Whether to attempt spelled-out and masked digit
            forms. Also exposed so its contribution is measurable.
        min_confidence: Score below which a match is not emitted.
    """

    detector_id = "pii-reference"
    detector_version = "0.1.0"
    access_tier = AccessTier.T3_TEXT
    category = Category.PII

    def __init__(
        self,
        *,
        validate_checksums: bool = True,
        detect_obfuscated: bool = True,
        min_confidence: float = 0.35,
    ) -> None:
        self.validate_checksums = validate_checksums
        self.detect_obfuscated = detect_obfuscated
        self.min_confidence = min_confidence

    # -- matching ------------------------------------------------------------ #

    def find(self, text: str) -> tuple[PiiMatch, ...]:
        """Every identifier match in one message, highest confidence first."""
        matches: list[PiiMatch] = []
        matches.extend(self._find_aadhaar(text))
        matches.extend(self._find_pan(text))
        matches.extend(self._find_upi(text))
        matches.extend(self._find_phone(text))
        matches.extend(self._find_ifsc(text))
        kept = [m for m in matches if m.confidence >= self.min_confidence]
        kept.sort(key=lambda m: (-m.confidence, m.start))
        return tuple(self._drop_overlaps(kept))

    @staticmethod
    def _drop_overlaps(matches: Sequence[PiiMatch]) -> Iterable[PiiMatch]:
        """Keep the highest-confidence match on any overlapping span.

        A twelve-digit Aadhaar also matches a ten-digit phone pattern in its
        tail. Emitting both would double-count one disclosure and inflate recall
        against a set that counts disclosures, not matches.
        """
        taken: list[PiiMatch] = []
        for match in matches:
            if any(match.start < t.end and t.start < match.end for t in taken):
                continue
            taken.append(match)
        return taken

    def _find_aadhaar(self, text: str) -> list[PiiMatch]:
        """Aadhaar: twelve digits, Verhoeff check digit, various separators."""
        found: list[PiiMatch] = []
        for match in AADHAAR_LOOSE.finditer(text):
            raw = match.group(0)
            digits = digits_only(raw)
            if len(digits) != 12:
                continue
            if self.validate_checksums:
                valid = verhoeff_is_valid(digits)
                # A pattern-only match that fails the checksum is still reported,
                # at lower confidence. Dropping it entirely would mean a typo in
                # a real disclosure attempt becomes invisible, and the disclosure
                # is what matters, not the arithmetic.
                confidence = 0.95 if valid else 0.55
            else:
                confidence = 0.70
            found.append(
                PiiMatch(
                    kind="IN_AADHAAR",
                    text=raw,
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                    checksum_checked=self.validate_checksums,
                )
            )
        if self.detect_obfuscated:
            found.extend(self._find_obfuscated_digits(text, "IN_AADHAAR", 12))
        return found

    def _find_obfuscated_digits(
        self, text: str, kind: str, expected_digits: int
    ) -> list[PiiMatch]:
        """Spelled-out, masked and interleaved digit runs.

        The disclosure form a verbatim-tuned recogniser misses entirely, and the
        one that dominates real chat traffic. Confidence is deliberately lower:
        these matches are genuinely less certain, and a detector that claims the
        same confidence on ``XXXX XXXX 9812`` as on a checksum-valid number is
        overclaiming.
        """
        found: list[PiiMatch] = []
        word_digits = (
            r"(?:zero|one|two|three|four|five|six|seven|eight|nine)"
        )
        patterns = [
            # digits followed by spelled-out digits
            (rf"\b\d{{4,8}}(?:\s+{word_digits}){{3,}}", 0.62),
            # masked with X or *
            (r"\b[X*x]{4,}\s*\d{2,6}\b", 0.58),
            (r"\b\d{2}[*X]{4,}\d{2}\b", 0.58),
            # digits interleaved with a filler word
            (r"\b\d{3,7}\s*(?:ka|wala|\(space\)|-)\s*\d{3,7}\b", 0.52),
        ]
        for pattern, confidence in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                raw = match.group(0)
                found.append(
                    PiiMatch(
                        kind=kind,
                        text=raw,
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        checksum_checked=False,
                    )
                )
        return found

    def _find_pan(self, text: str) -> list[PiiMatch]:
        """PAN: five letters, four digits, one letter, with a structural rule."""
        found: list[PiiMatch] = []
        for match in PAN_PATTERN.finditer(text):
            raw = match.group(0)
            compact = raw.replace(" ", "").replace(".", "").replace("-", "").upper()
            structurally_valid = (
                len(compact) == 10
                and compact[:5].isalpha()
                and compact[5:9].isdigit()
                and compact[9].isalpha()
                and compact[3] in "PCHFATBLJG"  # holder-type character
            )
            if self.validate_checksums:
                confidence = 0.92 if structurally_valid else 0.50
            else:
                confidence = 0.70
            found.append(
                PiiMatch(
                    kind="IN_PAN",
                    text=raw,
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                    checksum_checked=self.validate_checksums,
                )
            )
        return found

    def _find_upi(self, text: str) -> list[PiiMatch]:
        found: list[PiiMatch] = []
        for match in UPI_PATTERN.finditer(text):
            found.append(
                PiiMatch(
                    kind="UPI_VPA",
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.88,
                    checksum_checked=False,
                )
            )
        if self.detect_obfuscated:
            for match in re.finditer(
                r"\b[a-z][a-z0-9._-]{2,}\s*(?:at the rate|@)\s*(?:ok\w+|ybl|paytm|upi)\b",
                text,
                flags=re.IGNORECASE,
            ):
                found.append(
                    PiiMatch(
                        kind="UPI_VPA",
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.66,
                        checksum_checked=False,
                    )
                )
        return found

    def _find_phone(self, text: str) -> list[PiiMatch]:
        found: list[PiiMatch] = []
        for match in PHONE_LOOSE.finditer(text):
            raw = match.group(0)
            digits = digits_only(raw)
            if len(digits) not in (10, 11, 12, 13):
                continue
            # An Indian mobile starts 6-9 after the optional country code.
            national = digits[-10:]
            plausible = national[0] in "6789"
            found.append(
                PiiMatch(
                    kind="IN_PHONE",
                    text=raw,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.80 if plausible else 0.40,
                    checksum_checked=False,
                )
            )
        return found

    def _find_ifsc(self, text: str) -> list[PiiMatch]:
        return [
            PiiMatch(
                kind="IN_IFSC",
                text=match.group(0),
                start=match.start(),
                end=match.end(),
                confidence=0.85,
                checksum_checked=False,
            )
            for match in IFSC_PATTERN.finditer(text)
        ]

    # -- scoring -------------------------------------------------------------- #

    def score_one(self, text: str) -> float:
        """A single score in [0, 1]: the strongest match, or 0 if none.

        The maximum rather than a sum, because the question the score answers is
        *"does this message contain an identifier?"* — two identifiers do not
        make it more true than one, and summing would make long messages score
        higher for being long.
        """
        matches = self.find(text)
        return max((m.confidence for m in matches), default=0.0)

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Score a batch of messages."""
        return np.array([self.score_one(t) for t in texts], dtype=float)
