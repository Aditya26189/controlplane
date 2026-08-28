"""What each eval set's labels *mean*. ``DECISIONS.md`` 089.

An envelope is not only a distribution. It is a distribution **plus a label
definition**, and a detector can only be warranted on a set whose labels mean
what that detector detects.

This was found by trying to give a PII detector a warrant on the probe's
envelope so that two detector categories could compose on one input. The
validation harness accepted it and produced a number:

    pii-reference on triviaqa-2400-t960 — recall 0.0063 [0.0018, 0.0111]

That number is not a PII recall. TriviaQA's positive class is *"the model's
answer was incorrect"*, so what was actually measured is **the fraction of wrong
answers that happen to contain a personal identifier** — a quantity nobody
wants, filed under a warrant key that reads as though it were a PII claim. It is
the exact shape of failure this repo exists to prevent: plausible, well-formed,
and about a different question than its label suggests.

Nothing errored, because every part in isolation was doing its job. The set has
labels, the detector produces scores, the metrics are arithmetically correct.
Only the *meaning* was mismatched, and meaning was the one thing not represented
anywhere a check could reach.

## Why this is a separate registry rather than a field on the set

``construction["label_meaning"]`` already records it, in prose, and prose cannot
be checked. The obvious fix is a structured field on :class:`EvalSet` — but
construction notes are inside the content hash, so adding one would change the
identity of every frozen set and orphan every warrant keyed on it.

So the mapping lives here: explicit, auditable, versioned with the code, and
outside the hash. A set absent from this table cannot be validated against at
all, which is deliberate — an unmapped set is one whose labels nobody has
declared, and guessing is what produced the bad warrant above.
"""

from __future__ import annotations

from typing import Mapping, Optional

from ..model.enums import Category

__all__ = ["EVAL_SET_CATEGORY", "LabelCategoryError", "category_of", "require_compatible"]


class LabelCategoryError(ValueError):
    """A detector was pointed at a set whose labels mean something else."""


#: ``eval_set_id`` -> what its positive class means.
#:
#: ``None`` marks a **single-class** set: no positives, so no category claim is
#: made and any detector may be measured for its false-positive rate. That is
#: exactly what a hard-negative set is for, and it is the one case where
#: cross-category measurement is meaningful — "how often does this detector fire
#: on traffic that should never be flagged" is a question worth asking of any
#: detector, whatever it detects.
EVAL_SET_CATEGORY: Mapping[str, Optional[Category]] = {
    "triviaqa-600": Category.HALLUCINATION,
    "triviaqa-longctx-600": Category.HALLUCINATION,
    "triviaqa-2400-t960": Category.HALLUCINATION,
    "triviaqa-600-synthetic": Category.HALLUCINATION,
    "triviaqa-longctx-600-synthetic": Category.HALLUCINATION,
    "hinglish-pii-200": Category.PII,
    "hinglish-pii-200b": Category.PII,
    "hinglish-pii-200-longctx": Category.PII,
    "canary-20-pii": Category.PII,
    "canary-20-triviaqa": Category.HALLUCINATION,
    "hard-negatives-200": None,
}


def category_of(eval_set_id: str) -> Optional[Category]:
    """What this set's positive class means.

    Args:
        eval_set_id: The set.

    Returns:
        The category, or ``None`` for a declared single-class set.

    Raises:
        LabelCategoryError: If the set is not in the table. Unmapped is refused
            rather than defaulted: a set whose label meaning nobody has declared
            is precisely the case that produced a PII warrant measured against
            hallucination labels.
    """
    # Synthetic fixtures are suffixed rather than enumerated, so they resolve to
    # whatever their base set means. A fixture is a stand-in for one set, not a
    # set with a meaning of its own.
    key = eval_set_id
    for suffix in ("-synthetic", "-fixture"):
        if key.endswith(suffix) and key not in EVAL_SET_CATEGORY:
            key = key[: -len(suffix)]
    if key not in EVAL_SET_CATEGORY:
        raise LabelCategoryError(
            f"eval set {eval_set_id!r} has no declared label category. Add it to "
            "EVAL_SET_CATEGORY. Validating against a set whose labels nobody has "
            "declared is how a PII detector ends up with a warrant measured "
            "against hallucination labels (DECISIONS.md 089)."
        )
    return EVAL_SET_CATEGORY[key]


def require_compatible(detector_category: Optional[Category], eval_set_id: str) -> None:
    """Refuse a detector measured against labels that mean something else.

    Args:
        detector_category: What the detector detects. ``None`` for a detector
            that declares no category, which is allowed through — the guard
            exists to catch a *mismatch*, and a detector making no claim about
            what it detects cannot mismatch.
        eval_set_id: The set being measured on.

    Raises:
        LabelCategoryError: On a mismatch.
    """
    label_category = category_of(eval_set_id)
    if label_category is None or detector_category is None:
        return
    if detector_category is not label_category:
        raise LabelCategoryError(
            f"a {detector_category.value} detector cannot be warranted on "
            f"{eval_set_id!r}, whose positive class means "
            f"{label_category.value}. The measurement would be arithmetically "
            "correct and about a different question — for a PII detector on "
            "TriviaQA it is 'what fraction of wrong answers contain an "
            "identifier', filed under a key that reads as a PII claim. "
            "An envelope is a distribution plus a label definition "
            "(DECISIONS.md 089)."
        )
