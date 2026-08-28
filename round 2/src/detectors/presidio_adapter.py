"""Presidio behind the warrant machinery, unmodified. ``SPEC.md`` §8.1.

This adapter exists to answer one question — *what is an off-the-shelf PII
detector's error rate on Hinglish banking traffic?* — and the honest answer is
the deliverable whether or not it flatters Presidio.

**Presidio is not tuned here.** Three configurations are measured because
``DECISIONS.md`` 008 requires all three reported: reporting only the weakest is
open to "you crippled it", and reporting only the strongest hides what a team
gets on day one. What is forbidden is quietly improving one and reporting it as
another, which is why each configuration is a named constant and the recognizer
registry for each is built by a separate function.

## What "stock" actually means, verified rather than assumed

Presidio ships six Indian recognizers — ``InPanRecognizer``,
``InAadhaarRecognizer``, ``InGstinRecognizer``,
``InVehicleRegistrationRecognizer``, ``InVoterRecognizer``,
``InPassportRecognizer`` — and **registers none of them by default**. A default
``AnalyzerEngine()`` on this machine loads 17 recognizers, of which zero are
Indian. So "stock Presidio misses Indian identifiers" is not a subtle
performance claim; the recognizers are simply not loaded, and a team that
pip-installs Presidio and points it at Indian traffic gets nothing for Aadhaar,
PAN, UPI or IFSC until they know to go looking.

## What ``InAadhaarRecognizer`` validates, read from its source

Verified in Presidio 2.2.364 and recorded because ``TASKS.md`` Phase 8 asks for
it specifically. It is **not** a naive regex:

* two patterns, ``\\b[0-9]{12}\\b`` and ``\\b[0-9]{4}[- :][0-9]{4}[- :][0-9]{4}\\b``,
  both scored 0.01 and labelled "Very Weak";
* ``validate_result`` sanitises by removing ``-``, ``space`` and ``:`` only,
  then requires: twelve digits, numeric, **first digit >= 2**, a valid
  **Verhoeff** check digit, and not a palindrome.

Two consequences matter for measurement. The Verhoeff check is the real UIDAI
algorithm, so it is a genuinely strong validator and a *correct* rejection of a
made-up number is not a miss. And the sanitiser does not strip ``.``, so an
identifier written ``99.99.48.54.32.83`` — one of the disclosure forms in
``hinglish-pii-200`` — matches neither pattern and is never even offered to the
validator.

``evalsets/hinglish-pii-200.json`` draws its Aadhaar values from the UIDAI 9999
test range and records ``checksum_valid`` per item; that flag agrees with
Verhoeff on 34 of 34 Aadhaar items, so recall measured here is not confounded by
invalid fixture identifiers.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from ..model.enums import AccessTier, Category
from .pii_reference import PiiMatch

__all__ = [
    "CONFIGURATIONS",
    "INDIAN_RECOGNIZERS",
    "PresidioDetector",
    "presidio_available",
]

_LOG = logging.getLogger(__name__)

#: The six Indian recognizers Presidio ships and does not register by default.
INDIAN_RECOGNIZERS = (
    "InPanRecognizer",
    "InAadhaarRecognizer",
    "InGstinRecognizer",
    "InVehicleRegistrationRecognizer",
    "InVoterRecognizer",
    "InPassportRecognizer",
)

#: The three configurations ``config.detectors.presidio_configs`` declares.
#: Named constants rather than free strings so a typo is an error at import
#: rather than a silently different measurement.
CONFIGURATIONS = ("stock", "enabled", "enabled_plus_custom")

#: Entities we count as a personal identifier for this eval set. Presidio emits
#: many more (URL, DATE_TIME); counting those would score a detector as having
#: found PII when it found a date, and inflate recall against a set whose
#: positives are identifiers.
_IDENTIFIER_ENTITIES = frozenset(
    {
        # UPI_VPA and IN_IFSC are emitted only by the custom recognizers -- no
        # built-in Presidio recognizer produces them. They were missing from
        # this set at first, which filtered out exactly what the third
        # configuration adds and understated it. An allowlist that silently
        # drops a detector's output is the adapter misrepresenting the tool.
        "UPI_VPA",
        "IN_IFSC",
        "IN_AADHAAR",
        "IN_PAN",
        "IN_GSTIN",
        "IN_VEHICLE_REGISTRATION",
        "IN_VOTER",
        "IN_PASSPORT",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "CREDIT_CARD",
        "IBAN_CODE",
        "US_SSN",
    }
)


def presidio_available() -> bool:
    """Whether the optional dependency is importable.

    Presidio pulls spaCy and a language model, which is a large install and not
    present in every environment. Callers skip rather than fail, and the skip is
    reported — an adapter silently not running is how a detector's absence gets
    mistaken for a detector's clean sheet.
    """
    try:
        import presidio_analyzer  # noqa: F401
    except ImportError:
        return False
    return True


def _build_analyzer(configuration: str):
    """Build the engine for one named configuration.

    Args:
        configuration: One of :data:`CONFIGURATIONS`.

    Returns:
        A Presidio ``AnalyzerEngine``.

    Raises:
        ValueError: On an unknown configuration name.
    """
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer import predefined_recognizers

    if configuration not in CONFIGURATIONS:
        raise ValueError(
            f"unknown Presidio configuration {configuration!r}; expected one of "
            f"{list(CONFIGURATIONS)}"
        )

    engine = AnalyzerEngine()

    if configuration == "stock":
        # Nothing added. This is what `pip install presidio-analyzer` gives a
        # team that points it at Indian traffic on day one.
        return engine

    for name in INDIAN_RECOGNIZERS:
        engine.registry.add_recognizer(getattr(predefined_recognizers, name)())

    if configuration == "enabled_plus_custom":
        from .presidio_custom import custom_recognizers

        for recognizer in custom_recognizers():
            engine.registry.add_recognizer(recognizer)

    return engine


class PresidioDetector:
    """Presidio as a warranted detector. Satisfies ``TextDetector``.

    The point of this class is that it is **thin**. If wrapping a third-party
    detector required changes to the certificate schema, the validation harness
    or the drift monitor, that would be a finding about how detector-specific
    the machinery had become, and ``TASKS.md`` Phase 8 asks for it to be logged
    before the change is made. No such change was needed:
    :func:`~src.validation.text_runner.validate_text_detector` takes this class
    unmodified.

    Args:
        configuration: One of :data:`CONFIGURATIONS`.
        language: Presidio language code.
        min_confidence: Scores below this are not emitted. Presidio's Aadhaar
            patterns are scored 0.01 before validation and 1.0 after, so this
            floor is what separates "the pattern matched" from "the checksum
            passed" — and it is declared rather than tuned per configuration.
    """

    access_tier = AccessTier.T3_TEXT
    category = Category.PII

    def __init__(
        self,
        configuration: str = "stock",
        *,
        language: str = "en",
        min_confidence: float = 0.35,
    ) -> None:
        if configuration not in CONFIGURATIONS:
            raise ValueError(
                f"unknown Presidio configuration {configuration!r}; expected "
                f"one of {list(CONFIGURATIONS)}"
            )
        self.configuration = configuration
        self.language = language
        self.min_confidence = min_confidence
        self._analyzer = None

    @property
    def detector_id(self) -> str:
        """Configuration is part of the identity, not a runtime flag.

        Two configurations of Presidio have different measured bounds, so they
        are different detectors as far as the warrant key is concerned. Sharing
        an id would let a warrant measured on one be quoted for the other.
        """
        return f"presidio-{self.configuration}"

    @property
    def detector_version(self) -> str:
        """Presidio's own version, so a warrant is pinned to the code measured."""
        import importlib.metadata as metadata

        try:
            return metadata.version("presidio-analyzer")
        except metadata.PackageNotFoundError:  # pragma: no cover - defensive
            return "unknown"

    @property
    def analyzer(self):
        """The engine, built once on first use."""
        if self._analyzer is None:
            self._analyzer = _build_analyzer(self.configuration)
            names = sorted({r.name for r in self._analyzer.registry.recognizers})
            indian = [n for n in names if n in INDIAN_RECOGNIZERS]
            _LOG.info(
                "%s: %d recognizers registered, %d of them Indian (%s)",
                self.detector_id, len(names), len(indian),
                ", ".join(indian) or "none",
            )
        return self._analyzer

    # -- detection ---------------------------------------------------------- #

    def find(self, text: str) -> tuple[PiiMatch, ...]:
        """Every identifier Presidio reports in one message.

        Returns the same :class:`~src.detectors.pii_reference.PiiMatch` the
        reference detector returns, so the two are interchangeable everywhere
        downstream — which is what makes the Phase 8 comparison a comparison of
        detectors rather than of integrations.
        """
        results = self.analyzer.analyze(
            text=text, language=self.language, entities=None
        )
        matches = [
            PiiMatch(
                kind=result.entity_type,
                text=text[result.start : result.end],
                start=result.start,
                end=result.end,
                confidence=float(result.score),
                # Presidio reports validation through the score: a pattern-only
                # Aadhaar hit scores 0.01, a Verhoeff-validated one 1.0. There
                # is no separate flag, so this records what can actually be
                # known rather than inventing certainty.
                checksum_checked=result.entity_type in {"IN_AADHAAR", "IN_PAN"},
            )
            for result in results
            if result.entity_type in _IDENTIFIER_ENTITIES
            and result.score >= self.min_confidence
        ]
        matches.sort(key=lambda m: (-m.confidence, m.start))
        return tuple(matches)

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Highest-confidence identifier score per message, 0 when none.

        The message-level score is the maximum over matches rather than a count
        or a sum: the label is *"this message contains a personal identifier"*,
        so two identifiers in one message is still one positive, and summing
        would make verbose disclosures score higher than terse ones for no
        reason connected to the label.
        """
        out = np.zeros(len(texts), dtype=float)
        for index, text in enumerate(texts):
            matches = self.find(text)
            if matches:
                out[index] = max(m.confidence for m in matches)
        return out
