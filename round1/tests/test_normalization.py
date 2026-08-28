"""Answer/question normalisation and the lenient matching rule (SPEC.md §2)."""

import pytest

from src.data import (
    is_abstention,
    is_correct,
    is_exact_match,
    normalize_answer,
    normalize_question,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The Beatles", "beatles"),
        ("the  beatles.", "beatles"),
        ("A Tale of Two Cities", "tale of two cities"),
        ("An Apple", "apple"),
        ("  MIXED   Case,  Punctuation!  ", "mixed case punctuation"),
        ("", ""),
        ("!!!", ""),
        ("the", ""),
    ],
)
def test_normalize_answer(raw, expected):
    """Case, punctuation, articles and whitespace all normalise away."""
    assert normalize_answer(raw) == expected


def test_normalize_answer_handles_non_strings():
    """A missing alias must normalise to empty rather than raise mid-run."""
    assert normalize_answer(None) == ""


def test_normalize_question_keeps_articles():
    """Question dedup keeps articles; only the answer rule drops them."""
    assert normalize_question("Who wrote The Iliad?") == "who wrote the iliad"
    assert normalize_question("who  wrote the iliad") == "who wrote the iliad"


def test_normalize_question_collapses_near_duplicates():
    """Punctuation and casing differences must collapse to one dedup key."""
    a = normalize_question("What is the capital of France?")
    b = normalize_question("what is the capital of france")
    assert a == b


# --- the lenient rule ------------------------------------------------------ #


def test_lenient_match_accepts_alias_inside_a_sentence():
    """The model answers in a sentence; the gold answer is a span."""
    assert is_correct("The answer is the Beatles, I believe.", ["Beatles"])


def test_lenient_match_is_case_and_punctuation_insensitive():
    assert is_correct("paris!", ["Paris"])


def test_lenient_match_rejects_a_wrong_answer():
    assert not is_correct("The answer is Rome.", ["Paris", "City of Paris"])


def test_empty_prediction_is_never_correct():
    """A truncated or empty generation must not be labelled correct."""
    assert not is_correct("", ["Paris"])
    assert not is_correct("   ", ["Paris"])


def test_alias_list_with_only_empty_aliases_is_never_correct():
    assert not is_correct("anything at all", ["", "!!!", "the"])


# --- the short-alias guard ------------------------------------------------- #


@pytest.mark.parametrize(
    "prediction",
    ["The emperor was Augustus.", "Because of the trust involved", "just useful"],
)
def test_short_alias_rejected_as_substring(prediction):
    """An alias of "US" inside "Augustus" must not count as a match.

    This is the guard from CLAUDE.md's pitfall list: a two-character alias
    appears as a substring inside thousands of unrelated generations, and every
    one of them would be mislabelled correct (DECISIONS.md 010).
    """
    assert not is_correct(prediction, ["US"], min_alias_len_for_substring=3)


def test_short_alias_accepted_as_whole_token():
    """The same alias must still match when it stands as a token of its own."""
    assert is_correct("the us", ["US"], min_alias_len_for_substring=3)
    assert is_correct("It was the US, I think.", ["US"], min_alias_len_for_substring=3)


def test_short_alias_guard_is_configurable():
    """The threshold comes from config, so the guard can be audited by moving it.

    With the guard disabled, substring containment returns and "Augustus"
    matches "US" -- which is exactly the mislabelling the default prevents.
    """
    assert is_correct("The emperor was Augustus.", ["US"], min_alias_len_for_substring=1)


def test_long_alias_still_matches_as_substring():
    """Only short aliases are restricted; the lenient rule survives for the rest."""
    assert is_correct(
        "I think it was the united states of america.",
        ["United States of America"],
        min_alias_len_for_substring=3,
    )


# --- strict exact match ---------------------------------------------------- #


def test_exact_match_requires_the_whole_string():
    assert is_exact_match("The Beatles", ["beatles"])
    assert not is_exact_match("The answer is the Beatles", ["beatles"])


def test_exact_match_is_stricter_than_lenient():
    """The audit column must be a subset of the lenient one, never a superset."""
    prediction = "The answer is Paris."
    aliases = ["Paris"]
    assert is_correct(prediction, aliases)
    assert not is_exact_match(prediction, aliases)


# --- abstention detection (SPEC.md §9) ------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "I don't know.",
        "I DON'T KNOW",
        "Sorry, I'm not sure who that is.",
        "I cannot answer that",
        "There is no information available.",
    ],
)
def test_abstention_detected(text, config):
    assert is_abstention(text, config.abstention.patterns)


@pytest.mark.parametrize("text", ["Paris.", "The Beatles", "", "Nobody knows for sure"])
def test_non_abstention_not_detected(text, config):
    assert not is_abstention(text, config.abstention.patterns)
