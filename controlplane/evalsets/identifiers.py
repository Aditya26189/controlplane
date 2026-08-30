"""Synthetic Indian identifiers, and the three forms they get disclosed in.

**Everything produced here is synthetic.** Aadhaar numbers are drawn from the
`999999xxxxxx` range that UIDAI reserves for testing and never issues, PANs use
a reserved-looking surname block, and phone numbers use the `+91 90000 0xxxx`
prefix. They carry a valid **Verhoeff** check digit where the real format has
one, because a checksum-validating recogniser must be given input that actually
passes the checksum or the evaluation measures nothing.

**The three disclosure forms** are the reason this module exists. A recogniser
tuned on verbatim identifiers looks excellent and then misses most of the real
traffic, because people do not paste identifiers cleanly:

* ``verbatim`` — ``4176 5623 9812``, as a form field would produce.
* ``spaced`` — irregular whitespace and separators: ``4176-5623 9812``,
  ``4176 . 5623 . 9812``. Extremely common in chat.
* ``obfuscated`` — digits spelled, partially masked, or interleaved with words:
  ``4176 5623 nine eight one two``, ``XXXX XXXX 9812``. What people write when
  they know they should not be sharing it but need to.

Presidio's published HIGH-sensitivity recall of 0.07 comes from a benchmark
that deliberately includes non-verbatim forms (``DECISIONS.md`` 009). Building
the same structure here is what makes our number comparable rather than
flattering — and describing the construction is what stops the comparison being
turned against us.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Iterator, Literal

__all__ = [
    "DISCLOSURE_FORMS",
    "GENERATORS",
    "DisclosureForm",
    "Identifier",
    "IdentifierKind",
    "VERBATIM",
    "SPACED",
    "OBFUSCATED",
    "aadhaar",
    "ifsc",
    "pan",
    "phone",
    "upi_vpa",
    "verhoeff_check_digit",
    "verhoeff_is_valid",
]

DisclosureForm = Literal["verbatim", "spaced", "obfuscated"]
IdentifierKind = Literal["IN_AADHAAR", "IN_PAN", "UPI_VPA", "IN_PHONE", "IN_IFSC"]

VERBATIM: DisclosureForm = "verbatim"
SPACED: DisclosureForm = "spaced"
OBFUSCATED: DisclosureForm = "obfuscated"

DISCLOSURE_FORMS: tuple[DisclosureForm, ...] = (VERBATIM, SPACED, OBFUSCATED)

#: UIDAI reserves 9999 xxxx xxxx for testing and never issues it. Using it means
#: no string in this repo can collide with a real person's Aadhaar number.
_SYNTHETIC_AADHAAR_PREFIX = "9999"

_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}

# Verhoeff tables. The dihedral group D5 multiplication table, its permutation
# table, and the inverse table. Aadhaar uses this rather than Luhn, which is why
# a Luhn implementation silently accepts invalid Aadhaar numbers.
_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def verhoeff_check_digit(payload: str) -> str:
    """Compute the Verhoeff check digit for a digit string.

    Aadhaar uses Verhoeff, not Luhn. A Luhn implementation accepts roughly 90%
    of invalid Aadhaar numbers, so a recogniser built on the wrong checksum
    reports a precision it does not have.

    Args:
        payload: The digits the check digit is computed over, most significant
            first, without the check digit.

    Returns:
        A single character, ``"0"``-``"9"``.
    """
    if not payload.isdigit():
        raise ValueError(f"Verhoeff payload must be digits, got {payload!r}")
    checksum = 0
    for position, digit in enumerate(reversed(payload)):
        checksum = _D[checksum][_P[(position + 1) % 8][int(digit)]]
    return str(_INV[checksum])


def verhoeff_is_valid(number: str) -> bool:
    """Whether a digit string carries a valid Verhoeff check digit.

    Separators are ignored, because the whole point of the ``spaced`` disclosure
    form is that people write them.
    """
    digits = "".join(c for c in number if c.isdigit())
    if len(digits) < 2:
        return False
    checksum = 0
    for position, digit in enumerate(reversed(digits)):
        checksum = _D[checksum][_P[position % 8][int(digit)]]
    return checksum == 0


@dataclasses.dataclass(frozen=True)
class Identifier:
    """One synthetic identifier, and how it appears in text.

    Args:
        kind: What it is, using Presidio's entity names where they exist so the
            comparison is like-for-like.
        canonical: The identifier in its clean form.
        rendered: How it appears in the message, after the disclosure form is
            applied.
        form: Which disclosure form produced ``rendered``.
        checksum_valid: Whether ``canonical`` passes its format's checksum. False
            entries exist deliberately: a recogniser that validates checksums
            should reject them, and one that only pattern-matches will not.
    """

    kind: IdentifierKind
    canonical: str
    rendered: str
    form: DisclosureForm
    checksum_valid: bool = True


#: Separators and chunk sizes the ``spaced`` form draws from. Declared as data
#: rather than inlined because ``DECISIONS.md`` 085 turns on being able to say
#: exactly what space a detector was fitted to: the custom Presidio recognizers
#: cover this inventory completely, so redrawing from it at a new seed tests
#: nothing about generalisation.
BASE_SEPARATORS = (" ", "-", " - ", ".", " . ", "  ")
BASE_CHUNKS = (2, 3, 4)

#: The extension used to build an out-of-sample set. Every separator here is
#: **outside** ``presidio_custom._SEPARATOR``, which is ``[\s.\-:]`` — so items
#: drawn from the extension are the ones that measure how much of a fitted
#: recogniser's recall was fitting. Declared before the set was built (085).
EXTENDED_SEPARATORS = BASE_SEPARATORS + ("/", "_", "|", ",", "")
EXTENDED_CHUNKS = BASE_CHUNKS + (5, 6)


def _apply_form(
    digits_or_text: str,
    form: DisclosureForm,
    rng: random.Random,
    *,
    extended: bool = False,
) -> str:
    """Render an identifier in one of the three disclosure forms.

    Args:
        digits_or_text: The canonical identifier.
        form: Which disclosure form.
        rng: Seeded RNG.
        extended: Draw the ``spaced`` form's separator and chunk size from the
            extended inventory. Used only to build an out-of-sample set; the
            base inventory is what ``hinglish-pii-200`` was built from and is
            left untouched so that set still reproduces.
    """
    if form == VERBATIM:
        return digits_or_text
    if form == SPACED:
        separators = EXTENDED_SEPARATORS if extended else BASE_SEPARATORS
        chunks = EXTENDED_CHUNKS if extended else BASE_CHUNKS
        separator = rng.choice(list(separators))
        chunk = rng.choice(list(chunks))
        cleaned = digits_or_text.replace(" ", "")
        pieces = [cleaned[i : i + chunk] for i in range(0, len(cleaned), chunk)]
        return separator.join(pieces)
    # obfuscated
    cleaned = digits_or_text.replace(" ", "")
    style = rng.choice(["words", "mask_prefix", "mask_middle", "interleave"])
    if style == "words" and cleaned.isdigit():
        split = len(cleaned) // 2
        head = cleaned[:split]
        tail = " ".join(_DIGIT_WORDS[c] for c in cleaned[split:])
        return f"{head} {tail}"
    if style == "mask_prefix":
        keep = 4
        return f"{'X' * (len(cleaned) - keep)}{cleaned[-keep:]}"
    if style == "mask_middle":
        return f"{cleaned[:2]}{'*' * (len(cleaned) - 4)}{cleaned[-2:]}"
    filler = rng.choice(["(space)", "-", " ka ", " wala "])
    mid = len(cleaned) // 2
    return f"{cleaned[:mid]}{filler}{cleaned[mid:]}"


def aadhaar(
    rng: random.Random, form: DisclosureForm, *,
    valid: bool = True, extended: bool = False,
) -> Identifier:
    """A synthetic Aadhaar number in the UIDAI test range, with a real check digit.

    Args:
        rng: Seeded RNG, so the set is reproducible.
        form: Disclosure form.
        valid: Whether the check digit should be correct. Invalid ones exist so
            the checksum-validating configuration can be shown to reject what a
            pure regex accepts.
    """
    body = _SYNTHETIC_AADHAAR_PREFIX + "".join(str(rng.randint(0, 9)) for _ in range(7))
    check = verhoeff_check_digit(body)
    if not valid:
        check = str((int(check) + 1) % 10)
    canonical = body + check
    grouped = f"{canonical[:4]} {canonical[4:8]} {canonical[8:]}"
    return Identifier(
        kind="IN_AADHAAR",
        canonical=canonical,
        rendered=_apply_form(grouped, form, rng),
        form=form,
        checksum_valid=valid,
    )


def pan(
    rng: random.Random, form: DisclosureForm, *,
    valid: bool = True, extended: bool = False,
) -> Identifier:
    """A synthetic PAN: five letters, four digits, one letter.

    The fourth character encodes holder type (``P`` for individual) and the
    fifth is the surname initial. ``valid=False`` breaks the structural rule so
    a structural validator can be shown to reject it.
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    head = "".join(rng.choice(letters) for _ in range(3))
    holder = "P" if valid else "1"
    surname = rng.choice(letters)
    digits = "".join(str(rng.randint(0, 9)) for _ in range(4))
    tail = rng.choice(letters)
    canonical = f"{head}{holder}{surname}{digits}{tail}"
    return Identifier(
        kind="IN_PAN",
        canonical=canonical,
        rendered=_apply_form(canonical, form, rng) if form != VERBATIM else canonical,
        form=form,
        checksum_valid=valid,
    )


def upi_vpa(
    rng: random.Random, form: DisclosureForm, *, extended: bool = False
) -> Identifier:
    """A synthetic UPI virtual payment address, e.g. ``name@okhdfcbank``."""
    names = ["rahul", "priya", "arjun", "sneha", "vikram", "anita", "farhan", "meera"]
    handles = ["okhdfcbank", "okaxis", "oksbi", "okicici", "ybl", "paytm", "upi"]
    canonical = f"{rng.choice(names)}{rng.randint(10, 99)}@{rng.choice(handles)}"
    if form == VERBATIM:
        rendered = canonical
    elif form == SPACED:
        rendered = canonical.replace("@", " @ ")
    else:
        rendered = canonical.replace("@", " at the rate ")
    return Identifier(
        kind="UPI_VPA", canonical=canonical, rendered=rendered, form=form
    )


def phone(
    rng: random.Random, form: DisclosureForm, *, extended: bool = False
) -> Identifier:
    """A synthetic Indian mobile number in a reserved test prefix."""
    canonical = "+919" + "0000" + "".join(str(rng.randint(0, 9)) for _ in range(5))
    display = f"+91 {canonical[3:8]} {canonical[8:]}"
    return Identifier(
        kind="IN_PHONE",
        canonical=canonical,
        rendered=_apply_form(display, form, rng),
        form=form,
    )


def ifsc(
    rng: random.Random, form: DisclosureForm, *, extended: bool = False
) -> Identifier:
    """A synthetic IFSC code: four bank letters, ``0``, six branch characters."""
    banks = ["HDFC", "ICIC", "SBIN", "UTIB", "KKBK", "PUNB"]
    canonical = f"{rng.choice(banks)}0{''.join(str(rng.randint(0, 9)) for _ in range(6))}"
    return Identifier(
        kind="IN_IFSC",
        canonical=canonical,
        rendered=canonical if form == VERBATIM else _apply_form(canonical, form, rng),
        form=form,
    )


#: The generators, keyed so a builder can vary identifier type systematically.
GENERATORS = {
    "IN_AADHAAR": aadhaar,
    "IN_PAN": pan,
    "UPI_VPA": upi_vpa,
    "IN_PHONE": phone,
    "IN_IFSC": ifsc,
}
