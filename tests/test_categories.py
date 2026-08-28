"""An envelope is a distribution plus a label definition. ``DECISIONS.md`` 089.

Found by trying to warrant a PII detector on the probe's envelope so that two
categories could compose on one input. It worked, and produced a number that was
arithmetically correct and about a different question.
"""

from __future__ import annotations

import pytest

from controlplane.evalsets.categories import (
    EVAL_SET_CATEGORY,
    LabelCategoryError,
    category_of,
    require_compatible,
)
from controlplane.model import Category


def test_a_pii_detector_cannot_be_warranted_against_hallucination_labels() -> None:
    """The exact pairing that produced 'pii-reference recall 0.0063 on TriviaQA'.

    That number is the fraction of *wrong answers* that contain an identifier,
    filed under a key that reads as a PII claim.
    """
    with pytest.raises(LabelCategoryError, match="cannot be warranted"):
        require_compatible(Category.PII, "triviaqa-2400-t960")


def test_a_hallucination_detector_cannot_be_warranted_against_pii_labels() -> None:
    with pytest.raises(LabelCategoryError, match="cannot be warranted"):
        require_compatible(Category.HALLUCINATION, "hinglish-pii-200")


def test_a_matching_pair_is_allowed() -> None:
    require_compatible(Category.PII, "hinglish-pii-200")
    require_compatible(Category.HALLUCINATION, "triviaqa-2400-t960")


def test_any_detector_may_be_measured_on_a_single_class_set() -> None:
    """The one case where cross-category measurement is meaningful.

    'How often does this detector fire on traffic that should never be flagged'
    is worth asking of any detector, whatever it detects — which is what a
    hard-negative set is for.
    """
    assert category_of("hard-negatives-200") is None
    require_compatible(Category.PII, "hard-negatives-200")
    require_compatible(Category.HALLUCINATION, "hard-negatives-200")


def test_a_detector_declaring_no_category_is_allowed_through() -> None:
    """The guard catches a mismatch. A detector making no claim about what it
    detects cannot mismatch."""
    require_compatible(None, "triviaqa-2400-t960")


def test_an_unmapped_set_is_refused_rather_than_defaulted() -> None:
    """A set whose label meaning nobody declared is the case that caused this."""
    with pytest.raises(LabelCategoryError, match="no declared label category"):
        category_of("some-new-set-nobody-declared")


def test_synthetic_fixtures_inherit_their_base_set_meaning() -> None:
    """A fixture stands in for one set; it is not a set with a meaning of its own."""
    assert category_of("triviaqa-600-synthetic") is Category.HALLUCINATION
    assert category_of("hinglish-pii-200-fixture") is Category.PII


def test_every_shipped_eval_set_has_a_declared_meaning() -> None:
    """A set in evalsets/ that is not in the table cannot be validated against,
    which would be discovered at run time rather than here."""
    from pathlib import Path

    shipped = {
        p.stem
        for p in (Path(__file__).resolve().parents[1] / "evalsets").glob("*.json")
        if p.stem != "manifest"
    }
    missing = sorted(shipped - set(EVAL_SET_CATEGORY))
    assert not missing, f"eval sets with no declared label category: {missing}"
