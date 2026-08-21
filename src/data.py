"""Dataset loading, question deduplication, question-level splitting, labelling.

Two responsibilities that both have to be exactly right or the headline number
is meaningless:

1. **Splitting** (CLAUDE.md invariant 3). Splits are by ``question_id``, taken
   after deduplicating normalised question strings. TriviaQA ships near-
   duplicate phrasings; an example-level split leaks them across train and test
   and inflates AUROC by an unknown amount.
2. **Labelling** (SPEC.md §2). The model writes a sentence, the gold answer is
   a short span, so matching is lenient normalised containment -- with a guard
   that stops two-character aliases like "US" matching inside unrelated text.

Nothing here needs a GPU or a model; this stage runs on a laptop.
"""

import logging
import re
import string
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import Config

LOGGER = logging.getLogger(__name__)

_PUNCTUATION = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")

SPLIT_NAMES = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def normalize_answer(s: str) -> str:
    """Normalise an answer string the SQuAD/TriviaQA way.

    Lowercase, strip punctuation, drop articles, collapse whitespace. Used on
    both sides of every correctness comparison so that "The Beatles.",
    "the beatles" and "Beatles" compare equal (SPEC.md §2).

    Args:
        s: Raw prediction or gold alias.

    Returns:
        The normalised string, possibly empty.
    """
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCTUATION)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def normalize_question(s: str) -> str:
    """Normalise a question string for deduplication only.

    Lowercase, strip punctuation, collapse whitespace -- and, unlike
    :func:`normalize_answer`, keep articles (SPEC.md §1 step 2). Articles carry
    meaning in questions ("who wrote the Iliad" is not "who wrote Iliad" in the
    same way that "the Beatles" is "Beatles"), so dropping them here would merge
    questions that are not duplicates and quietly shrink the dataset.

    Args:
        s: Raw question text.

    Returns:
        Lowercased, punctuation-stripped, whitespace-collapsed key.
    """
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCTUATION)
    return " ".join(s.split())


# --------------------------------------------------------------------------- #
# Labelling
# --------------------------------------------------------------------------- #


def is_correct(
    prediction: str,
    aliases: Sequence[str],
    min_alias_len_for_substring: int = 3,
) -> bool:
    """Lenient correctness: does any gold alias appear in the generation?

    Strict exact match would label almost everything incorrect, because the
    model answers in a sentence and the gold answer is a span (DECISIONS.md
    007). The guard matters: an alias shorter than
    ``min_alias_len_for_substring`` ("US", "UK") appears inside thousands of
    unrelated strings, so short aliases require a whole-token match instead of
    substring containment.

    Args:
        prediction: The model's generated text.
        aliases: Gold answer aliases, un-normalised.
        min_alias_len_for_substring: Normalised aliases shorter than this need a
            whole-token match. From ``config.labeling``.

    Returns:
        True if the generation counts as correct under the lenient rule.
    """
    pred = normalize_answer(prediction)
    if not pred:
        return False
    tokens = set(pred.split())
    for alias in aliases:
        a = normalize_answer(alias)
        if not a:
            continue
        if len(a) < min_alias_len_for_substring:
            if a in tokens:
                return True
        elif a in pred:
            return True
    return False


def is_exact_match(prediction: str, aliases: Sequence[str]) -> bool:
    """Strict exact match, recorded as the audit column beside the lenient rule.

    Reported alongside the lenient base rate so a reviewer can see how much of
    the label distribution the matching rule is responsible for (SPEC.md §2).

    Args:
        prediction: The model's generated text.
        aliases: Gold answer aliases, un-normalised.

    Returns:
        True if the whole normalised generation equals a normalised alias.
    """
    pred = normalize_answer(prediction)
    if not pred:
        return False
    return any(pred == normalize_answer(alias) for alias in aliases)


def is_abstention(text: str, patterns: Iterable[str]) -> bool:
    """Does the generation express that the model does not know?

    Used for the secondary validation in SPEC.md §9: if the probe direction
    also tracks the model's own expressed uncertainty, that is independent
    evidence it reads something real rather than a dataset artifact.

    Args:
        text: The model's generated text.
        patterns: Lowercase abstention phrases from ``config.abstention``.

    Returns:
        True if any pattern occurs in the lowercased, whitespace-collapsed text.
    """
    if not isinstance(text, str) or not text:
        return False
    # Curly apostrophes are folded to ASCII: models emit both forms, and the
    # configured patterns are written with the ASCII one.
    haystack = " ".join(text.lower().replace("’", "'").split())
    return any(pattern in haystack for pattern in patterns)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _collect_aliases(answer: Any) -> list[str]:
    """Flatten a TriviaQA answer struct into a deduplicated alias list.

    Uses ``value``, ``aliases`` and ``normalized_aliases`` together (SPEC.md
    §1). They overlap heavily, but each occasionally carries a surface form the
    others miss, and every one of them is re-normalised before comparison.
    """
    out: list[str] = []
    if not isinstance(answer, dict):
        return out
    value = answer.get("value")
    if isinstance(value, str) and value.strip():
        out.append(value)
    for key in ("aliases", "normalized_aliases"):
        entries = answer.get(key)
        if entries is None:
            continue
        for alias in entries:
            if isinstance(alias, str) and alias.strip():
                out.append(alias)
    seen: set[str] = set()
    unique: list[str] = []
    for alias in out:
        if alias not in seen:
            seen.add(alias)
            unique.append(alias)
    return unique


def load_raw(config: Config) -> pd.DataFrame:
    """Load the configured dataset split into a tidy frame.

    Only the fields SPEC.md §1 lists are kept. ``rc.nocontext`` still carries
    empty ``entity_pages``/``search_results`` columns; dropping them before
    conversion keeps memory flat.

    Args:
        config: Resolved experiment config.

    Returns:
        Frame with columns ``question_id``, ``question``, ``answer_value``,
        ``aliases``.

    Raises:
        RuntimeError: if the dataset cannot be loaded, with the offline-cache
            hint from CLAUDE.md attached.
    """
    from datasets import load_dataset  # imported lazily: heavy, and GPU-free

    LOGGER.info(
        "loading %s (%s) split=%s",
        config.data.dataset,
        config.data.config,
        config.data.split,
    )
    try:
        ds = load_dataset(
            config.data.dataset, config.data.config, split=config.data.split
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with actionable context
        raise RuntimeError(
            f"could not load {config.data.dataset}/{config.data.config}. On an "
            "offline compute node, pre-download into HF_HOME on a login node and "
            "set HF_DATASETS_OFFLINE=1 (CLAUDE.md, Environment notes)."
        ) from exc

    keep = [c for c in ("question_id", "question", "answer") if c in ds.column_names]
    missing = {"question_id", "question", "answer"} - set(keep)
    if missing:
        raise RuntimeError(
            f"dataset is missing expected column(s) {sorted(missing)}; "
            f"available: {ds.column_names}"
        )
    frame = ds.select_columns(keep).to_pandas()
    frame["aliases"] = frame["answer"].map(_collect_aliases)
    frame["answer_value"] = frame["answer"].map(
        lambda a: a.get("value", "") if isinstance(a, dict) else ""
    )
    frame = frame.drop(columns=["answer"])
    LOGGER.info("loaded %d rows", len(frame))
    return frame


# --------------------------------------------------------------------------- #
# Deduplication, subsampling, splitting
# --------------------------------------------------------------------------- #


def deduplicate_questions(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop repeated normalised question strings, keeping the first occurrence.

    Runs before splitting, not after: deduplicating within splits would still
    leave a near-duplicate of a training question sitting in test (DECISIONS.md
    003).

    Args:
        frame: Frame with a ``question`` column.

    Returns:
        The deduplicated frame and the number of rows dropped.
    """
    frame = frame.copy()
    frame["question_norm"] = frame["question"].map(normalize_question)
    before = len(frame)
    frame = frame.drop_duplicates(subset="question_norm", keep="first")
    dropped = before - len(frame)
    LOGGER.info("dedup on normalised question: dropped %d of %d rows", dropped, before)
    return frame.reset_index(drop=True), dropped


def drop_empty(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows with no question text or no usable gold alias.

    An example with no alias can never be labelled correct, so keeping it would
    inject guaranteed positives into the label distribution.

    Args:
        frame: Frame with ``question_norm`` and ``aliases`` columns.

    Returns:
        The filtered frame and the number of rows dropped.
    """
    before = len(frame)
    has_question = frame["question_norm"].str.len() > 0
    has_alias = frame["aliases"].map(
        lambda aliases: any(normalize_answer(a) for a in aliases)
    )
    frame = frame[has_question & has_alias]
    dropped = before - len(frame)
    LOGGER.info("dropped %d rows with an empty question or no usable alias", dropped)
    return frame.reset_index(drop=True), dropped


def split_by_question(
    frame: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> pd.DataFrame:
    """Assign each ``question_id`` to exactly one split (CLAUDE.md invariant 3).

    Splitting on unique ids rather than rows means that if a question ever
    appears twice, both copies land in the same split. The shuffle uses a local
    ``RandomState`` rather than the global seed so the split does not depend on
    how much other code drew from the global generator first.

    Args:
        frame: Deduplicated frame with ``question_id``.
        train_frac: Fraction of question ids for train.
        val_frac: Fraction for validation; the remainder is test.
        seed: Split seed.

    Returns:
        A copy of ``frame`` with a ``split`` column of train/val/test.
    """
    ids = frame["question_id"].unique()
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(ids))
    shuffled = ids[order]

    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    assignment: dict[Any, str] = {}
    for qid in shuffled[:n_train]:
        assignment[qid] = "train"
    for qid in shuffled[n_train : n_train + n_val]:
        assignment[qid] = "val"
    for qid in shuffled[n_train + n_val :]:
        assignment[qid] = "test"

    out = frame.copy()
    out["split"] = out["question_id"].map(assignment)
    if out["split"].isna().any():
        raise AssertionError("every question_id must receive a split assignment")
    return out


def assert_split_integrity(frame: pd.DataFrame) -> None:
    """Assert splits are pairwise disjoint on id *and* on normalised question.

    Enforces CLAUDE.md invariant 3 in the pipeline, not only in the test suite:
    a leak here inflates the headline number and nothing else would raise.

    Args:
        frame: Frame with ``split``, ``question_id`` and ``question_norm``.

    Raises:
        AssertionError: on any overlap, or a split name outside train/val/test.
    """
    unexpected = set(frame["split"].unique()) - set(SPLIT_NAMES)
    if unexpected:
        raise AssertionError(f"unexpected split label(s): {sorted(unexpected)}")
    for key in ("question_id", "question_norm"):
        groups = {
            name: set(frame.loc[frame["split"] == name, key]) for name in SPLIT_NAMES
        }
        for i, left in enumerate(SPLIT_NAMES):
            for right in SPLIT_NAMES[i + 1 :]:
                overlap = groups[left] & groups[right]
                if overlap:
                    raise AssertionError(
                        f"{key} overlap between {left} and {right}: "
                        f"{len(overlap)} shared value(s), e.g. {sorted(overlap)[:3]}"
                    )


def prepare_dataset(config: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the whole data pipeline of SPEC.md §1 and return the split frame.

    Order is load, deduplicate, drop empties, shuffle, subsample, split. The
    subsample happens *before* the split so that changing ``n_examples`` does
    not silently change which questions are in test relative to a smaller run's
    train.

    Args:
        config: Resolved experiment config.

    Returns:
        The frame with a ``split`` column, and a stats dict for the artifacts.
    """
    frame = load_raw(config)
    rows_loaded = len(frame)

    if config.data.dedup_questions:
        frame, duplicates_dropped = deduplicate_questions(frame)
    else:
        frame = frame.copy()
        frame["question_norm"] = frame["question"].map(normalize_question)
        duplicates_dropped = 0

    frame, empties_dropped = drop_empty(frame)

    rng = np.random.RandomState(config.seed)
    frame = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)

    requested = config.data.n_examples
    if requested > len(frame):
        LOGGER.warning(
            "requested n_examples=%d but only %d rows survive filtering; using all",
            requested,
            len(frame),
        )
    frame = frame.iloc[: min(requested, len(frame))].reset_index(drop=True)

    frame = split_by_question(
        frame, config.data.train_frac, config.data.val_frac, config.seed
    )
    assert_split_integrity(frame)

    sizes = frame["split"].value_counts().to_dict()
    stats: dict[str, Any] = {
        "dataset": config.data.dataset,
        "dataset_config": config.data.config,
        "split": config.data.split,
        "rows_loaded": int(rows_loaded),
        "duplicates_dropped": int(duplicates_dropped),
        "empty_or_aliasless_dropped": int(empties_dropped),
        "n_examples_requested": int(requested),
        "n_final": int(len(frame)),
        "split_sizes": {name: int(sizes.get(name, 0)) for name in SPLIT_NAMES},
        "seed": config.seed,
    }
    LOGGER.info(
        "prepared %d examples: train=%d val=%d test=%d",
        stats["n_final"],
        stats["split_sizes"]["train"],
        stats["split_sizes"]["val"],
        stats["split_sizes"]["test"],
    )
    return frame, stats


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

SPLIT_COLUMNS = ["question_id", "question", "question_norm", "answer_value", "aliases", "split"]


def save_splits(frame: pd.DataFrame, path: str | Path) -> Path:
    """Persist the split assignment so every later stage reads the same one.

    Args:
        frame: The frame returned by :func:`prepare_dataset`.
        path: Destination parquet path.

    Returns:
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame[SPLIT_COLUMNS].to_parquet(out, index=False)
    LOGGER.info("wrote %s (%d rows)", out, len(frame))
    return out


def load_splits(path: str | Path) -> pd.DataFrame:
    """Read the persisted splits and re-assert their integrity.

    The re-assert is not paranoia about parquet: it catches a stale
    ``splits.parquet`` left over from a different config, which would otherwise
    let a later stage train on a split that does not match its own settings.

    Args:
        path: Parquet path written by :func:`save_splits`.

    Returns:
        The split frame.
    """
    frame = pd.read_parquet(path)
    missing = set(SPLIT_COLUMNS) - set(frame.columns)
    if missing:
        raise AssertionError(f"{path} is missing column(s) {sorted(missing)}")
    frame["aliases"] = frame["aliases"].map(
        lambda a: list(a) if a is not None else []
    )
    assert_split_integrity(frame)
    return frame


def label_frame(
    frame: pd.DataFrame,
    completions: Sequence[str],
    config: Config,
    abstention_patterns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Attach generations, correctness, polarity label, and abstention flags.

    The probe's positive class is *incorrect* (DECISIONS.md 004), so
    ``label = 1 - correct``. Stated here because getting it backwards produces
    ``1 - AUROC``, which reads as a strong negative result rather than a bug.

    Args:
        frame: Split frame, in the same order as ``completions``.
        completions: Generated text, one per row.
        config: Resolved config, for the labelling rule.
        abstention_patterns: Defaults to ``config.abstention.patterns``.

    Returns:
        A copy of ``frame`` with ``completion``, ``correct``, ``exact_match``,
        ``label`` and ``abstained`` columns.
    """
    if len(completions) != len(frame):
        raise AssertionError(
            f"got {len(completions)} completions for {len(frame)} rows"
        )
    patterns = list(
        abstention_patterns
        if abstention_patterns is not None
        else config.abstention.patterns
    )
    out = frame.copy()
    out["completion"] = list(completions)
    out["correct"] = [
        is_correct(text, aliases, config.labeling.min_alias_len_for_substring)
        for text, aliases in zip(out["completion"], out["aliases"])
    ]
    if config.labeling.record_strict_em:
        out["exact_match"] = [
            is_exact_match(text, aliases)
            for text, aliases in zip(out["completion"], out["aliases"])
        ]
    else:
        # NaN, not False. "We did not compute this" and "no answer matched
        # strictly" are different facts, and conflating them makes the strict-vs-
        # lenient audit report a 100-point gap that never happened.
        out["exact_match"] = np.nan
    out["label"] = (~out["correct"]).astype(int)
    out["abstained"] = [is_abstention(text, patterns) for text in out["completion"]]
    return out
