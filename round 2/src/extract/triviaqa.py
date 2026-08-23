"""TriviaQA loading, deduplication, question-level splitting, and labelling.

Three things here are correctness-critical and each fails silently:

* **Deduplicate before splitting.** TriviaQA ships near-duplicate question
  strings. An example-level split, or a question-level split over
  un-deduplicated questions, puts the same question on both sides and the probe
  scores itself on what it memorised.
* **Short-alias matching.** A gold alias of ``"US"`` appears inside thousands of
  unrelated generations, so substring containment marks wrong answers correct
  and the base rate collapses. Aliases under three characters require an exact
  token match.
* **Label polarity.** ``1`` means *incorrect* — the thing the probe should fire
  on. Inverting it yields ``1 - AUROC``, which reads as a strong negative
  result and misdirects debugging for hours.
"""

from __future__ import annotations

import logging
import re
import string
import unicodedata
from typing import Any, Iterable, Optional, Sequence

__all__ = [
    "TriviaItem",
    "is_correct",
    "load_triviaqa",
    "normalise_answer",
    "split_questions",
]

_LOG = logging.getLogger(__name__)

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCTUATION = set(string.punctuation)

#: Aliases shorter than this require an exact token match rather than substring
#: containment. "US", "UK" and "CD" appear inside ordinary words and inside
#: thousands of unrelated generations; matching them by containment marks wrong
#: answers correct, which lowers the measured base rate and inflates every
#: downstream number without raising anything.
SHORT_ALIAS_CHARS = 3


class TriviaItem:
    """One question, its gold aliases, and the model's answer once generated.

    A plain class rather than a frozen dataclass because ``response`` and
    ``label`` are filled in after generation, and the alternative is
    constructing the object twice.
    """

    __slots__ = ("question_id", "question", "aliases", "response", "label")

    def __init__(
        self,
        question_id: str,
        question: str,
        aliases: Sequence[str],
        response: str = "",
        label: Optional[int] = None,
    ) -> None:
        self.question_id = question_id
        self.question = question
        self.aliases = tuple(aliases)
        self.response = response
        self.label = label


def normalise_answer(text: str) -> str:
    """Standard TriviaQA normalisation: lowercase, strip articles and punctuation.

    NFKC first so that full-width and composed forms compare equal, which
    matters because generations occasionally contain them and gold aliases
    never do.
    """
    lowered = unicodedata.normalize("NFKC", text).lower()
    without_punctuation = "".join(
        " " if character in _PUNCTUATION else character for character in lowered
    )
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def is_correct(response: str, aliases: Iterable[str]) -> tuple[bool, str]:
    """Whether a generated answer matches any gold alias.

    Args:
        response: The model's generation.
        aliases: Gold answer aliases from the dataset.

    Returns:
        ``(correct, how)`` — the second naming which alias matched and by which
        rule, so a disputed label can be examined rather than argued about.

    The short-alias guard is the whole point of this function. TriviaQA's alias
    lists include two-character entries, and ``"us" in normalise_answer(...)``
    is true for any generation containing "thus", "focus" or "must".
    """
    normalised_response = normalise_answer(response)
    if not normalised_response:
        return False, "empty generation"
    response_tokens = set(normalised_response.split())

    for alias in aliases:
        normalised_alias = normalise_answer(alias)
        if not normalised_alias:
            continue
        if len(normalised_alias) < SHORT_ALIAS_CHARS:
            # Exact token match only. Containment on a two-character alias marks
            # unrelated generations correct.
            if normalised_alias in response_tokens:
                return True, f"exact token match on short alias {normalised_alias!r}"
            continue
        if normalised_alias in normalised_response:
            return True, f"substring match on {normalised_alias!r}"
    return False, "no alias matched"


def load_triviaqa(
    *,
    n_questions: int,
    seed: int,
    split: str = "validation",
    subset: str = "rc.nocontext",
    cache_dir: Optional[str] = None,
) -> list[TriviaItem]:
    """Load and deduplicate TriviaQA questions.

    ``rc.nocontext`` because the probe reads the model's *own* knowledge state
    at question time. Supplying the evidence document would measure reading
    comprehension instead, and the whole claim is about what the model knows
    before it starts generating.

    Args:
        n_questions: How many distinct questions to keep after deduplication.
        seed: Sampling seed.
        split: Dataset split to draw from.
        subset: Dataset configuration.
        cache_dir: HuggingFace cache directory, for offline runs.

    Returns:
        Deduplicated items, in a stable order.

    Raises:
        RuntimeError: If fewer distinct questions survive deduplication than
            were asked for, rather than silently returning a smaller set — a
            short set changes every interval downstream.
    """
    import numpy as np
    from datasets import load_dataset

    _LOG.info("loading TriviaQA %s/%s", subset, split)
    dataset = load_dataset("trivia_qa", subset, split=split, cache_dir=cache_dir)

    seen: dict[str, str] = {}
    items: list[TriviaItem] = []
    duplicates = 0
    for row in dataset:
        question = row["question"].strip()
        key = normalise_answer(question)
        if not key:
            continue
        if key in seen:
            duplicates += 1
            continue
        seen[key] = row["question_id"]
        aliases = list(row["answer"].get("aliases") or [])
        aliases += list(row["answer"].get("normalized_aliases") or [])
        value = row["answer"].get("value")
        if value:
            aliases.append(value)
        if not aliases:
            continue
        items.append(
            TriviaItem(
                question_id=str(row["question_id"]),
                question=question,
                aliases=sorted(set(aliases)),
            )
        )
        if len(items) >= n_questions * 3:
            # Enough to sample from without walking the whole split.
            break

    _LOG.info(
        "%d distinct questions after dedup (%d near-duplicates dropped)",
        len(items),
        duplicates,
    )
    if len(items) < n_questions:
        raise RuntimeError(
            f"asked for {n_questions} distinct questions but only {len(items)} "
            "survived deduplication. Returning a smaller set would change every "
            "interval downstream without saying so; widen the split instead."
        )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(items), size=n_questions, replace=False)
    return [items[int(i)] for i in sorted(chosen)]


def split_questions(
    items: Sequence[TriviaItem],
    *,
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, list[int]]:
    """Assign whole questions to train, validation and test.

    By question, never by example, and the assertion at the end is the point:
    an overlapping split inflates every downstream number and produces no error
    of its own.

    Args:
        items: Deduplicated questions.
        fractions: Train, validation, test proportions.
        seed: Shuffling seed.

    Returns:
        Mapping of split name to indices into ``items``.

    Raises:
        RuntimeError: If any question lands in two splits, or a split is empty.
    """
    import numpy as np

    if abs(sum(fractions) - 1.0) > 1e-9:
        raise RuntimeError(f"fractions must sum to 1, got {fractions}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    n_train = int(round(fractions[0] * len(items)))
    n_validation = int(round(fractions[1] * len(items)))

    splits = {
        "train": sorted(int(i) for i in order[:n_train]),
        "validation": sorted(int(i) for i in order[n_train : n_train + n_validation]),
        "test": sorted(int(i) for i in order[n_train + n_validation :]),
    }
    for name, indices in splits.items():
        if not indices:
            raise RuntimeError(f"split {name!r} is empty at fractions {fractions}")

    all_indices = [i for indices in splits.values() for i in indices]
    if len(set(all_indices)) != len(all_indices):
        raise RuntimeError("splits overlap: a question appears in more than one")
    if len(all_indices) != len(items):
        raise RuntimeError(
            f"splits cover {len(all_indices)} of {len(items)} questions"
        )
    _LOG.info("split by question: %s", {k: len(v) for k, v in splits.items()})
    return splits
