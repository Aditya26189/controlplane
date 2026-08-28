"""Regular expressions for Indian identifiers, deliberately loose.

Loose because the disclosure forms this project cares about are the messy ones.
A tight pattern anchored on ``\\d{4} \\d{4} \\d{4}`` reports excellent precision
on form-field data and misses most of a chat corpus, which is the failure mode
``hinglish-pii-200`` exists to measure. Tightening is the *validator's* job —
Verhoeff for Aadhaar, the structural rule for PAN — so the pattern casts wide
and the checksum decides confidence.

Kept in their own module so the pattern set is reviewable in one place, and so
Phase 8 can diff it against what Presidio's shipped recognisers actually match.
"""

from __future__ import annotations

import re

__all__ = [
    "AADHAAR_LOOSE",
    "IFSC_PATTERN",
    "PAN_PATTERN",
    "PHONE_LOOSE",
    "UPI_PATTERN",
    "digits_only",
]

#: Twelve digits in groups, separated by anything people actually type: spaces,
#: hyphens, dots, or nothing. The `(?<![\d])` guards stop a match starting or
#: ending mid-number, which would otherwise pull twelve digits out of a
#: sixteen-digit transaction reference and report an Aadhaar that is not there.
AADHAAR_LOOSE = re.compile(
    r"(?<![\d])"
    r"\d{2,4}(?:[\s.\-]{0,3}\d{2,4}){2,5}"
    r"(?![\d])"
)

#: Five letters, four digits, one letter, tolerating separators inside.
PAN_PATTERN = re.compile(
    r"(?<![A-Z0-9])"
    r"[A-Z]{3}[\s.\-]{0,2}[A-Z]{2}[\s.\-]{0,2}\d{4}[\s.\-]{0,2}[A-Z]"
    r"(?![A-Z0-9])",
    re.IGNORECASE,
)

#: A UPI virtual payment address. The handle list is open because banks add new
#: ones; the shape is what identifies it.
UPI_PATTERN = re.compile(
    r"\b[a-z0-9][a-z0-9._-]{1,}\s?@\s?[a-z][a-z0-9]{1,}\b",
    re.IGNORECASE,
)

#: An Indian mobile number, with or without country code, with messy separators.
PHONE_LOOSE = re.compile(
    r"(?<![\d])"
    r"(?:\+?91[\s.\-]{0,3})?"
    r"\d{2,5}(?:[\s.\-]{0,3}\d{2,5}){1,4}"
    r"(?![\d])"
)

#: IFSC: four bank letters, a zero, six branch characters.
IFSC_PATTERN = re.compile(
    r"\b[A-Z]{4}[\s.\-]{0,2}0[\s.\-]{0,2}[A-Z0-9]{6}\b",
    re.IGNORECASE,
)


def digits_only(text: str) -> str:
    """Strip everything but digits.

    Used before a checksum, because the separators are exactly what the
    ``spaced`` disclosure form varies and the checksum is defined over digits.
    """
    return "".join(character for character in text if character.isdigit())
