"""Custom recognizers for the third Presidio configuration. ``DECISIONS.md`` 008.

**These are fitted to the failure modes observed on ``hinglish-pii-200``, and
that is the point rather than a caveat.** The three configurations answer three
different questions and only the third involves any work on our side:

* ``stock`` — what a team gets on day one. Presidio registers no Indian
  recognizers.
* ``enabled`` — what a team gets after finding the six Indian recognizers
  Presidio already ships. Free, and the honest baseline for "does Presidio
  handle Indian identifiers".
* ``enabled_plus_custom`` — what it costs to close the remaining gap. This file
  is that cost, measured in code.

Reporting all three is required (``DECISIONS.md`` 008): reporting only the first
invites *"you crippled it"*, and reporting only the last hides the day-one
experience that is most of the argument.

**What is forbidden, and is not done here:** improving ``stock`` or ``enabled``
and reporting the improved number under those names. Each configuration builds
its registry in a separate branch of :func:`~controlplane.detectors.presidio_adapter._build_analyzer`,
and this module is reachable from exactly one of them.

## Why the built-in recognizers miss what they miss

Read from Presidio 2.2.364's source rather than inferred from behaviour:

``InAadhaarRecognizer`` matches ``\\b[0-9]{12}\\b`` or
``\\b[0-9]{4}[- :][0-9]{4}[- :][0-9]{4}\\b`` and sanitises by removing only
``-``, space and ``:``. An Aadhaar written ``99.99.48.54.32.83`` matches neither
pattern, so the Verhoeff validator never sees it. The miss is at the pattern
stage, not the validation stage — which matters, because it means the fix is a
looser pattern feeding the *same* checksum, not a weaker check.

Every recognizer here therefore keeps the original validation and widens only
the separator set. Nothing below accepts an identifier the built-in recognizer
would have rejected on its checksum.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from presidio_analyzer import EntityRecognizer

__all__ = ["custom_recognizers"]

#: Separators seen between digits of a deliberately broken-up identifier.
#: Kept explicit rather than ``\\W`` so the recognizer cannot swallow arbitrary
#: punctuation and join two unrelated numbers into a false twelve-digit match.
_SEPARATOR = r"[\s.\-:]"


def custom_recognizers() -> Sequence["EntityRecognizer"]:
    """Build the separator-tolerant recognizers for ``enabled_plus_custom``.

    Returns:
        Recognizers to add on top of the six built-in Indian ones.
    """
    from presidio_analyzer import Pattern, PatternRecognizer
    from presidio_analyzer.predefined_recognizers import (
        InAadhaarRecognizer,
        InPanRecognizer,
    )

    class SeparatorTolerantAadhaar(InAadhaarRecognizer):
        """Aadhaar with any of space, dot, dash or colon between digit groups.

        Subclasses the built-in so ``validate_result`` — first digit >= 2,
        Verhoeff, non-palindrome — is inherited unchanged. Only the pattern and
        the sanitiser's separator set are widened, so this cannot accept a
        number the original would have rejected.
        """

        PATTERNS = [
            Pattern(
                "AADHAAR separator-tolerant",
                r"\b[0-9]{4}" + _SEPARATOR + r"{0,3}[0-9]{4}" + _SEPARATOR
                + r"{0,3}[0-9]{4}\b",
                0.01,
            ),
            Pattern(
                "AADHAAR digit-pairs",
                r"\b(?:[0-9]{2}" + _SEPARATOR + r"{1,3}){5}[0-9]{2}\b",
                0.01,
            ),
        ]

        def __init__(self) -> None:
            super().__init__(
                patterns=self.PATTERNS,
                replacement_pairs=[("-", ""), (" ", ""), (":", ""), (".", "")],
                name="SeparatorTolerantAadhaar",
            )

    class SeparatorTolerantPan(InPanRecognizer):
        """PAN written ``YXC.PR7.606.R`` rather than ``YXCPR7606R``.

        PAN has a structural rule (five letters, four digits, one letter) and no
        checksum, so tolerance here is genuinely weaker evidence than for
        Aadhaar. It is scored accordingly by the base class.
        """

        PATTERNS = [
            Pattern(
                "PAN separator-tolerant",
                r"\b[A-Za-z]{3}" + _SEPARATOR + r"{0,3}[A-Za-z]{2}[0-9]"
                + _SEPARATOR + r"{0,3}[0-9]{3}" + _SEPARATOR + r"{0,3}[A-Za-z]\b",
                0.3,
            ),
        ]

        def __init__(self) -> None:
            super().__init__(patterns=self.PATTERNS, name="SeparatorTolerantPan")

    upi = PatternRecognizer(
        supported_entity="UPI_VPA",
        name="UpiVpaRecognizer",
        patterns=[
            Pattern("UPI VPA", r"\b[A-Za-z0-9._-]{3,}\s?@\s?[A-Za-z]{2,}\b", 0.6),
            # "arjun66 at the rate paytm" -- spelled-out separator, which is a
            # disclosure form a customer actually uses when a form field
            # rejects '@'.
            Pattern(
                "UPI VPA spelled",
                r"\b[A-Za-z0-9._-]{3,}\s+(?:at the rate|at)\s+[A-Za-z]{2,}\b",
                0.4,
            ),
        ],
        context=["upi", "vpa", "paytm", "gpay", "phonepe"],
    )

    ifsc = PatternRecognizer(
        supported_entity="IN_IFSC",
        name="IfscRecognizer",
        patterns=[
            Pattern("IFSC", r"\b[A-Za-z]{4}0[A-Za-z0-9]{6}\b", 0.6),
            Pattern(
                "IFSC separator-tolerant",
                r"\b[A-Za-z]{2}" + _SEPARATOR + r"{0,3}[A-Za-z]{2}"
                + _SEPARATOR + r"{0,3}0(?:" + _SEPARATOR + r"{0,3}[A-Za-z0-9]{1,6}){1,6}\b",
                0.35,
            ),
        ],
        context=["ifsc", "neft", "rtgs", "branch"],
    )

    return (SeparatorTolerantAadhaar(), SeparatorTolerantPan(), upi, ifsc)
