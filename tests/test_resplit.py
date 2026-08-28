"""Re-splitting a frozen set, and the cache reuse it licenses.

``DECISIONS.md`` 079. The whole operation rests on one claim — that reallocating
declared splits leaves the data untouched — so these tests are mostly about that
claim being *checked* rather than believed, and about the guard that was
deliberately loosened still catching everything it was there for.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.evalsets.resplit import cache_source_id, resplit_by_question
from src.validation.evalsets import (
    TEST,
    TRAIN,
    VALIDATION,
    EvalItem,
    EvalSet,
    EvalSetError,
    ExtractionCache,
    split_by_question,
)


def make_set(eval_set_id: str = "source-set", n_questions: int = 100) -> EvalSet:
    """A frozen set with two items per question and declared 50/25/25 splits."""
    items = []
    for q in range(n_questions):
        split = TRAIN if q < 50 else (VALIDATION if q < 75 else TEST)
        for k in range(2):
            items.append(
                EvalItem(
                    item_id=f"i-{q:03d}-{k}",
                    question_id=f"q-{q:03d}",
                    prompt=f"question {q}?",
                    response=f"answer {q}.{k}",
                    label=(q + k) % 2,
                    split=split,
                )
            )
    return EvalSet(
        eval_set_id=eval_set_id,
        items=tuple(items),
        construction={"method": "fixture"},
    )


def cache_for(evalset: EvalSet, eval_set_hash: str | None = None) -> ExtractionCache:
    return ExtractionCache(
        eval_set_id=evalset.eval_set_id,
        eval_set_hash=eval_set_hash or evalset.content_hash,
        model_name="fixture-model",
        layer=-1,
        data_source=evalset.data_source,
        features={"T1-mean_pool": np.zeros((len(evalset.items), 4))},
        labels=np.array([i.label for i in evalset.items]),
        question_ids=np.array([i.question_id for i in evalset.items], dtype=object),
        token_lengths=np.full(len(evalset.items), 32.0),
    )


# --------------------------------------------------------------------------- #
# The extraction identity
# --------------------------------------------------------------------------- #


def test_re_splitting_changes_the_content_hash_and_not_the_extraction_hash() -> None:
    """The two hashes exist to answer two different questions.

    Content hash: is this the same *set*? No — invariant 9, a new identity.
    Extraction hash: is this the same *data*? Yes — so the activations stand.
    """
    source = make_set()
    derived = resplit_by_question(
        source, eval_set_id="derived", fractions=(0.4, 0.2, 0.4), seed=1729,
        rationale="fixture",
    )
    assert derived.content_hash != source.content_hash
    assert derived.extraction_hash == source.extraction_hash


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(lambda i: dataclasses.replace(i, prompt="edited"), id="prompt"),
        pytest.param(lambda i: dataclasses.replace(i, response="edited"), id="response"),
        pytest.param(lambda i: dataclasses.replace(i, label=1 - i.label), id="label"),
        pytest.param(lambda i: dataclasses.replace(i, item_id="renamed"), id="item_id"),
        pytest.param(lambda i: dataclasses.replace(i, question_id="q-999"), id="question_id"),
    ],
)
def test_the_extraction_hash_still_catches_every_edit_to_the_data(mutation) -> None:
    """The loosened guard must still see everything it was loosened around."""
    source = make_set()
    edited = dataclasses.replace(
        source, items=(mutation(source.items[0]),) + source.items[1:]
    )
    assert edited.extraction_hash != source.extraction_hash


def test_the_extraction_hash_ignores_only_the_name_and_the_split() -> None:
    """The two fields that are not inputs to a forward pass."""
    source = make_set()
    renamed = dataclasses.replace(source, eval_set_id="a-different-name")
    resplit = dataclasses.replace(
        source, items=tuple(dataclasses.replace(i, split=TEST) for i in source.items)
    )
    assert renamed.extraction_hash == source.extraction_hash
    assert resplit.extraction_hash == source.extraction_hash


def test_reordering_the_items_changes_the_extraction_hash() -> None:
    """Order is an input: activations are filed by row."""
    source = make_set()
    shuffled = dataclasses.replace(source, items=tuple(reversed(source.items)))
    assert shuffled.extraction_hash != source.extraction_hash


# --------------------------------------------------------------------------- #
# The re-split itself
# --------------------------------------------------------------------------- #


def test_whole_questions_move_together() -> None:
    """Splitting by example lets a probe score itself on a rephrasing."""
    derived = resplit_by_question(
        make_set(), eval_set_id="derived", fractions=(0.4, 0.2, 0.4), seed=1729,
        rationale="fixture",
    )
    by_question: dict[str, set[str]] = {}
    for item in derived.items:
        by_question.setdefault(item.question_id, set()).add(item.split)
    assert all(len(splits) == 1 for splits in by_question.values())


def test_the_derived_splits_have_no_overlap_and_cover_everything() -> None:
    derived = resplit_by_question(
        make_set(), eval_set_id="derived", fractions=(0.4, 0.2, 0.4), seed=1729,
        rationale="fixture",
    )
    splits = split_by_question(derived, seed=1729)
    assert sum(len(v) for v in splits.values()) == len(derived.items)
    for a in splits:
        for b in splits:
            if a < b:
                assert not set(splits[a]) & set(splits[b])


def test_a_re_split_must_be_a_new_set() -> None:
    """Two sets sharing a name and differing in content is invariant 9's failure."""
    source = make_set()
    with pytest.raises(EvalSetError, match="must be a new set"):
        resplit_by_question(
            source, eval_set_id=source.eval_set_id, fractions=(0.4, 0.2, 0.4),
            seed=1729, rationale="fixture",
        )


def test_a_shortfall_against_the_committed_test_size_raises_here() -> None:
    """Missing a pre-registered number should be an error at the point of the
    decision, not a disappointment three stages later."""
    with pytest.raises(EvalSetError, match="below the"):
        resplit_by_question(
            make_set(n_questions=100), eval_set_id="derived",
            fractions=(0.8, 0.1, 0.1), seed=1729, rationale="fixture",
            min_test_items=673,
        )


def test_the_re_split_is_reproducible_at_one_seed() -> None:
    kwargs = dict(
        eval_set_id="derived", fractions=(0.4, 0.2, 0.4), seed=1729, rationale="fixture"
    )
    a = resplit_by_question(make_set(), **kwargs)
    b = resplit_by_question(make_set(), **kwargs)
    assert a.content_hash == b.content_hash


def test_the_derivation_is_recorded_inside_the_frozen_file() -> None:
    """A derived set that cannot say what it was derived from is an orphan.

    The notes are inside the content hash, so the provenance cannot be edited
    off without changing the set's identity.
    """
    source = make_set()
    derived = resplit_by_question(
        source, eval_set_id="derived", fractions=(0.4, 0.2, 0.4), seed=1729,
        rationale="because the test split was too small",
    )
    notes = derived.construction
    assert notes["derived_from"] == source.eval_set_id
    assert notes["derived_from_content_hash"] == source.content_hash
    assert notes["derived_from_extraction_hash"] == source.extraction_hash
    assert notes["resplit_rationale"]
    assert cache_source_id(derived) == source.eval_set_id
    assert cache_source_id(source) == source.eval_set_id


# --------------------------------------------------------------------------- #
# Cache reuse
# --------------------------------------------------------------------------- #


def test_a_cache_matches_a_re_split_of_the_set_it_came_from(tmp_path) -> None:
    source = make_set()
    derived = resplit_by_question(
        source, eval_set_id="derived", fractions=(0.4, 0.2, 0.4), seed=1729,
        rationale="fixture",
    )
    path = cache_for(source).save(tmp_path / "c.npz")
    loaded = ExtractionCache.load(path, expected_items=derived)
    assert loaded.n_items == len(derived.items)


def test_the_strict_hash_check_still_refuses_a_re_split(tmp_path) -> None:
    """The narrow check is opt-in. Callers expecting a byte-identical set keep
    the guard they had."""
    source = make_set()
    derived = resplit_by_question(
        source, eval_set_id="derived", fractions=(0.4, 0.2, 0.4), seed=1729,
        rationale="fixture",
    )
    path = cache_for(source).save(tmp_path / "c.npz")
    with pytest.raises(EvalSetError, match="set changed after extraction"):
        ExtractionCache.load(path, expected_hash=derived.content_hash)


def test_a_cache_from_a_different_set_is_refused_and_names_the_row(tmp_path) -> None:
    source = make_set()
    other = dataclasses.replace(
        source,
        items=(dataclasses.replace(source.items[0], question_id="q-999"),)
        + source.items[1:],
    )
    path = cache_for(source).save(tmp_path / "c.npz")
    with pytest.raises(EvalSetError, match="row 0"):
        ExtractionCache.load(path, expected_items=other)


def test_a_relabelled_set_is_refused(tmp_path) -> None:
    """Relabelling changes what every measured number means."""
    source = make_set()
    relabelled = dataclasses.replace(
        source,
        items=(dataclasses.replace(source.items[0], label=1 - source.items[0].label),)
        + source.items[1:],
    )
    path = cache_for(source).save(tmp_path / "c.npz")
    with pytest.raises(EvalSetError, match="label"):
        ExtractionCache.load(path, expected_items=relabelled)


def test_a_cache_of_the_wrong_length_is_refused(tmp_path) -> None:
    source = make_set()
    shorter = dataclasses.replace(source, items=source.items[:-2])
    path = cache_for(source).save(tmp_path / "c.npz")
    with pytest.raises(EvalSetError, match="different sets"):
        ExtractionCache.load(path, expected_items=shorter)


def test_validate_refuses_a_cache_from_an_unrelated_set(tmp_path) -> None:
    """The re-split exemption is narrow: it applies only to a set that declares
    this cache's set as its source, and only after the items are compared."""
    from src.config import load_config
    from pathlib import Path as _Path

    config = load_config(str(_Path(__file__).resolve().parents[1] / "config.yaml"))
    source = make_set("source-set")
    unrelated = make_set("unrelated-set")
    # Same items, but it does not declare a derivation, so the exemption must
    # not apply even though the data would compare equal.
    assert "derived_from" not in unrelated.construction
    with pytest.raises(ValueError, match="Re-extract"):
        from src.validation.runner import validate

        validate(
            config,
            unrelated,
            cache_for(source),
            variant="T1-mean_pool",
            detector_id="fixture",
            detector_version="0.0.1",
        )
