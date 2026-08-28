"""Reallocate a frozen set's declared splits without touching its data.

``DECISIONS.md`` 079. Built for one specific, pre-registered need: a warrant's
test split was too small to support a calibration claim a profile makes, and the
activations for every item in the set were already extracted.

**This produces a new set, never a mutation.** ``split`` is inside
:meth:`EvalSet.content_hash`, so a reallocation is a different identity —
invariant 9, working exactly as intended. The source set keeps its id, its
envelope, its warrants and its published numbers. What comes out of here is a
new envelope that has never been measured, and measuring it is a fresh scoring
rather than a re-opening of the old one.

**The data is asserted identical, not assumed.** The new set's extraction
identity is compared against the source's, and a mismatch raises. That assertion
is what licenses reusing the extraction cache, so it is checked rather than
argued: if re-splitting ever altered an item, the activations would be filed
against the wrong rows and every number after it would be wrong and plausible.

Nothing here decides *what* the new proportions should be. A split size chosen
to make a failing check pass is selection on test wearing a different hat, and
the caller has to have written the number down first (``DECISIONS.md`` 079 does).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Optional

import numpy as np

from ..validation.evalsets import SPLITS, TEST, TRAIN, VALIDATION, EvalItem, EvalSet, EvalSetError

__all__ = ["cache_source_id", "resplit_by_question"]

_LOG = logging.getLogger(__name__)


def resplit_by_question(
    source: EvalSet,
    *,
    eval_set_id: str,
    fractions: tuple[float, float, float],
    seed: int,
    rationale: str,
    min_test_items: Optional[int] = None,
) -> EvalSet:
    """Reassign declared splits by question, leaving every item otherwise intact.

    Whole questions move together, as everywhere else in this repo: splitting by
    example lets a probe score itself on a rephrasing of what it was fitted on,
    and the resulting numbers are inflated in a way nothing errors about.

    Args:
        source: The frozen set to derive from.
        eval_set_id: Name for the new set. Must differ from the source's — two
            sets sharing a name and differing in content is the confusion
            invariant 9 exists to prevent, and the matrix keys its envelope axis
            by this string.
        fractions: ``(train, validation, test)`` proportions of *questions*.
        seed: Shuffling seed, so the allocation is reproducible.
        rationale: Why this was done, recorded in the new set's construction
            notes and therefore inside its content hash. A derived set that
            cannot say what it was derived for is an orphan.
        min_test_items: If given, the reallocation must produce at least this
            many test items or the call raises. Pass the number the
            pre-registration committed to, so that missing it is an error here
            rather than a disappointment three stages later.

    Returns:
        A new :class:`EvalSet`.

    Raises:
        EvalSetError: If the name is unchanged, the fractions are invalid, the
            data would not survive intact, or ``min_test_items`` is unmet.
    """
    if eval_set_id == source.eval_set_id:
        raise EvalSetError(
            f"a re-split must be a new set; {eval_set_id!r} is the source's own "
            "name. Two sets sharing a name and differing in content is exactly "
            "the confusion invariant 9 prevents."
        )
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise EvalSetError(f"fractions must be three values summing to 1, got {fractions}")
    if any(f <= 0 for f in fractions):
        raise EvalSetError(
            f"every split needs a positive share, got {fractions}. A zero share "
            "is a different design (a set with no train rows, say) and should "
            "be declared rather than fallen into."
        )

    questions = list(dict.fromkeys(item.question_id for item in source.items))
    order = np.random.default_rng(seed).permutation(len(questions))
    shuffled = [questions[i] for i in order]

    n_train = int(round(len(shuffled) * fractions[0]))
    n_validation = int(round(len(shuffled) * fractions[1]))
    assignment: dict[str, str] = {}
    for position, question in enumerate(shuffled):
        if position < n_train:
            assignment[question] = TRAIN
        elif position < n_train + n_validation:
            assignment[question] = VALIDATION
        else:
            assignment[question] = TEST

    items = tuple(
        dataclasses.replace(item, split=assignment[item.question_id])
        for item in source.items
    )
    counts = {name: sum(1 for i in items if i.split == name) for name in SPLITS}

    if min_test_items is not None and counts[TEST] < min_test_items:
        raise EvalSetError(
            f"re-split produced {counts[TEST]} test items, below the "
            f"{min_test_items} the caller committed to. Adjust the fractions "
            "deliberately and record it, rather than accepting a smaller "
            "held-out sample than was pre-registered."
        )

    construction: dict[str, Any] = {
        **dict(source.construction),
        "derived_from": source.eval_set_id,
        "derived_from_content_hash": source.content_hash,
        "derived_from_extraction_hash": source.extraction_hash,
        "resplit_fractions": list(fractions),
        "resplit_seed": seed,
        "resplit_counts": counts,
        "resplit_rationale": rationale,
    }

    new = EvalSet(
        eval_set_id=eval_set_id,
        items=items,
        data_source=source.data_source,
        construction=construction,
    )

    # The assertion that licenses reusing the extraction cache. Checked, not
    # argued: if a re-split ever altered an item, activations would be filed
    # against the wrong rows and every number after would be wrong and
    # plausible.
    if new.extraction_hash != source.extraction_hash:
        raise EvalSetError(
            f"{eval_set_id}: re-splitting changed the data. The extraction "
            "identity must be preserved — that is the whole basis for reusing "
            "the cache instead of re-extracting."
        )

    _LOG.info(
        "%s: re-split from %s into %s (extraction identity unchanged: %s)",
        eval_set_id, source.eval_set_id, counts, new.extraction_hash[:16],
    )
    return new


def cache_source_id(evalset: EvalSet) -> str:
    """Which set's extraction cache holds this set's activations.

    A re-split set has no extraction of its own and does not need one: its items
    are the source's items in the source's order, asserted so at construction.
    This reads the derivation recorded in the set's own construction notes
    rather than taking it as an argument, so the answer travels inside the
    frozen file and cannot be supplied wrongly at a call site.

    Args:
        evalset: The set whose activations are wanted.

    Returns:
        The ``eval_set_id`` whose cache to load. The set's own id when it was
        extracted directly.
    """
    return str(evalset.construction.get("derived_from") or evalset.eval_set_id)
