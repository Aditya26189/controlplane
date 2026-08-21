"""Split integrity: no question leaks across train/val/test (invariant 3).

These run on a synthetic frame rather than on TriviaQA so they need no network
and no cache, and so a leak is provoked deliberately rather than waited for.
"""

import pandas as pd
import pytest

from src.data import (
    SPLIT_NAMES,
    assert_split_integrity,
    deduplicate_questions,
    drop_empty,
    normalize_question,
    split_by_question,
)


def make_frame(n: int = 300) -> pd.DataFrame:
    """Build a synthetic frame shaped like the loaded dataset."""
    return pd.DataFrame(
        {
            "question_id": [f"q{i:04d}" for i in range(n)],
            "question": [f"Who invented gadget number {i}?" for i in range(n)],
            "answer_value": [f"Person {i}" for i in range(n)],
            "aliases": [[f"Person {i}", f"P{i} the inventor"] for i in range(n)],
        }
    ).assign(question_norm=lambda f: f["question"].map(normalize_question))


def test_splits_are_pairwise_disjoint_on_question_id():
    frame = split_by_question(make_frame(), 0.6, 0.2, seed=1729)
    assert_split_integrity(frame)
    groups = {
        name: set(frame.loc[frame["split"] == name, "question_id"])
        for name in SPLIT_NAMES
    }
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]


def test_splits_are_pairwise_disjoint_on_normalised_question():
    """Disjoint ids are not enough: a re-asked question would still leak."""
    frame = split_by_question(make_frame(), 0.6, 0.2, seed=1729)
    groups = {
        name: set(frame.loc[frame["split"] == name, "question_norm"])
        for name in SPLIT_NAMES
    }
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]


def test_split_sizes_are_60_20_20_within_one_example():
    frame = split_by_question(make_frame(1000), 0.6, 0.2, seed=1729)
    counts = frame["split"].value_counts()
    assert abs(counts["train"] - 600) <= 1
    assert abs(counts["val"] - 200) <= 1
    assert abs(counts["test"] - 200) <= 1
    assert counts.sum() == 1000


def test_every_row_receives_exactly_one_split():
    frame = split_by_question(make_frame(97), 0.6, 0.2, seed=1729)
    assert frame["split"].isna().sum() == 0
    assert set(frame["split"].unique()) <= set(SPLIT_NAMES)
    assert len(frame) == 97


def test_split_is_deterministic_under_a_fixed_seed():
    """Same seed, same assignment: reproducibility starts here."""
    first = split_by_question(make_frame(), 0.6, 0.2, seed=1729)
    second = split_by_question(make_frame(), 0.6, 0.2, seed=1729)
    assert first["split"].tolist() == second["split"].tolist()


def test_split_changes_with_the_seed():
    first = split_by_question(make_frame(), 0.6, 0.2, seed=1729)
    other = split_by_question(make_frame(), 0.6, 0.2, seed=7)
    assert first["split"].tolist() != other["split"].tolist()


def test_repeated_question_id_lands_in_one_split():
    """Grouping is by id, so duplicate ids cannot straddle a split boundary."""
    frame = make_frame(50)
    doubled = pd.concat([frame, frame], ignore_index=True)
    out = split_by_question(doubled, 0.6, 0.2, seed=1729)
    per_id = out.groupby("question_id")["split"].nunique()
    assert (per_id == 1).all()


def test_integrity_check_catches_an_injected_id_leak():
    """The assertion must actually fire; a check that cannot fail proves nothing."""
    frame = split_by_question(make_frame(), 0.6, 0.2, seed=1729)
    leaked = frame.copy()
    train_row = leaked[leaked["split"] == "train"].iloc[0]
    test_index = leaked.index[leaked["split"] == "test"][0]
    leaked.loc[test_index, "question_id"] = train_row["question_id"]
    with pytest.raises(AssertionError, match="question_id overlap"):
        assert_split_integrity(leaked)


def test_integrity_check_catches_an_injected_question_leak():
    """Distinct ids with the same question text are still a leak."""
    frame = split_by_question(make_frame(), 0.6, 0.2, seed=1729)
    leaked = frame.copy()
    train_row = leaked[leaked["split"] == "train"].iloc[0]
    test_index = leaked.index[leaked["split"] == "test"][0]
    leaked.loc[test_index, "question_norm"] = train_row["question_norm"]
    with pytest.raises(AssertionError, match="question_norm overlap"):
        assert_split_integrity(leaked)


# --- the filters that run before the split --------------------------------- #


def test_deduplicate_collapses_normalised_duplicates():
    frame = pd.DataFrame(
        {
            "question_id": ["a", "b", "c"],
            "question": [
                "What is the capital of France?",
                "what is the  capital of france",
                "Who wrote the Iliad?",
            ],
            "answer_value": ["Paris", "Paris", "Homer"],
            "aliases": [["Paris"], ["Paris"], ["Homer"]],
        }
    )
    out, dropped = deduplicate_questions(frame)
    assert dropped == 1
    assert len(out) == 2
    assert out.iloc[0]["question_id"] == "a"  # first occurrence is kept


def test_drop_empty_removes_unanswerable_rows():
    """A row with no usable alias can never be correct and would skew the base rate."""
    frame = pd.DataFrame(
        {
            "question_id": ["a", "b", "c"],
            "question": ["Real question?", "", "Another real one?"],
            "question_norm": ["real question", "", "another real one"],
            "answer_value": ["Paris", "Paris", ""],
            "aliases": [["Paris"], ["Paris"], ["", "!!!"]],
        }
    )
    out, dropped = drop_empty(frame)
    assert dropped == 2
    assert out["question_id"].tolist() == ["a"]
