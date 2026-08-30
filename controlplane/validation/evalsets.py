"""Evaluation sets and the cached extraction ``/validate`` runs from.

**Eval sets are frozen and content-hashed** (``CLAUDE.md`` invariant 9). The
hash is the envelope id, which is the third element of the warrant key, so
changing a set's contents changes its identity and every warrant measured on
the old contents keeps pointing at the old envelope. Nothing silently inherits.

That identity rule does one more job that is worth stating, because it is doing
real work rather than being a nicety: **a synthetic fixture cannot masquerade as
a measured eval set.** Its contents differ, so its hash differs, so it occupies
a different cell in the matrix. There is no path by which numbers measured on a
fixture end up filed under ``triviaqa-600``. ``DECISIONS.md`` 027.

**Splits are by question, never by example.** TriviaQA ships several examples
per question and occasionally near-duplicate question strings, so an
example-level split puts the same question on both sides of the line and the
probe scores itself on what it memorised. :func:`split_by_question` groups
first, deduplicates on the normalised question string, and asserts zero overlap.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from ..model import content_hash

__all__ = [
    "EvalItem",
    "EvalSet",
    "EvalSetError",
    "ExtractionCache",
    "PaddingEvidence",
    "TRAIN",
    "TEST",
    "VALIDATION",
    "normalise_question",
    "split_by_question",
]

_LOG = logging.getLogger(__name__)

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
SPLITS = (TRAIN, VALIDATION, TEST)

#: Marks data that was generated rather than measured. Carried through every
#: artifact so a synthetic run can never be read as a measured one.
SOURCE_MEASURED = "measured"
SOURCE_SYNTHETIC = "synthetic"


class EvalSetError(ValueError):
    """Raised when an eval set or its cache is inconsistent."""


def normalise_question(text: str) -> str:
    """Normalise a question string for deduplication.

    Unicode NFKC, case-folded, punctuation stripped, whitespace collapsed. The
    aim is to catch *near*-duplicates — "Who wrote the Iliad?" and "who wrote
    The Iliad" are the same question for splitting purposes, and letting them
    land on opposite sides of the split lets the probe score itself on a
    question it was fitted on.

    NFKC before case-folding because Devanagari and full-width Latin both
    normalise under it, and the Hinglish set contains both.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    stripped = re.sub(r"[^\w\s]", " ", folded, flags=re.UNICODE)
    return re.sub(r"\s+", " ", stripped).strip()


@dataclasses.dataclass(frozen=True)
class EvalItem:
    """One scorable item.

    Args:
        item_id: Unique within the set.
        question_id: The grouping unit for splitting. Several items may share it.
        prompt: What the model was asked.
        response: What it answered. Empty for sets scored on the prompt alone.
        label: 1 means *incorrect* — the positive class. See ``stats.py``.
        split: Declared split, when the set ships one. A frozen eval set built as
            a held-out sample — ``triviaqa-600`` is exactly that — declares every
            item ``"test"`` and draws train and validation rows from a separate
            set, rather than being cut to a third of its size by a derived split.
            When no item declares one, splits are derived by question.
        meta: Anything a detector or an eval needs, e.g. gold aliases, PII spans,
            or the reason an item is a hard negative.
    """

    item_id: str
    question_id: str
    prompt: str
    response: str
    label: int
    split: Optional[str] = None
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.item_id or not self.question_id:
            raise EvalSetError("item_id and question_id are required")
        if self.split is not None and self.split not in SPLITS:
            raise EvalSetError(
                f"{self.item_id}: split must be one of {list(SPLITS)}, got "
                f"{self.split!r}"
            )
        if self.label not in (0, 1):
            raise EvalSetError(
                f"{self.item_id}: label must be 0 or 1 with 1 meaning 'incorrect', "
                f"got {self.label!r}"
            )


@dataclasses.dataclass(frozen=True)
class EvalSet:
    """A frozen, content-hashed collection of items.

    Args:
        eval_set_id: Human-readable name, e.g. ``"triviaqa-600"``.
        items: The items, in a fixed order.
        data_source: ``"measured"`` or ``"synthetic"``. Part of the hashed
            identity, so a fixture and a real set can never collide.
        construction: How the set was built — enough for a reader to rebuild or
            challenge it. Also hashed.
    """

    eval_set_id: str
    items: tuple[EvalItem, ...]
    data_source: str = SOURCE_MEASURED
    construction: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.items:
            raise EvalSetError(f"{self.eval_set_id}: an eval set must have items")
        if self.data_source not in (SOURCE_MEASURED, SOURCE_SYNTHETIC):
            raise EvalSetError(
                f"{self.eval_set_id}: data_source must be {SOURCE_MEASURED!r} or "
                f"{SOURCE_SYNTHETIC!r}, got {self.data_source!r}"
            )
        ids = [item.item_id for item in self.items]
        if len(set(ids)) != len(ids):
            raise EvalSetError(f"{self.eval_set_id}: duplicate item_id")

    @property
    def content_hash(self) -> str:
        """SHA-256 over the set's contents — the envelope id.

        Covers the items, the ``data_source`` and the construction notes.
        Changing any of them creates a new identity, which is invariant 9: a
        modified set is a *different* set and cannot inherit the old one's
        warrants.
        """
        return content_hash(
            {
                "eval_set_id": self.eval_set_id,
                "data_source": self.data_source,
                "construction": self.construction,
                "items": [
                    {
                        "item_id": i.item_id,
                        "question_id": i.question_id,
                        "prompt": i.prompt,
                        "response": i.response,
                        "label": i.label,
                        "split": i.split,
                        "meta": i.meta,
                    }
                    for i in self.items
                ],
            }
        )

    @property
    def extraction_hash(self) -> str:
        """Identity of the data an extraction actually read. ``DECISIONS.md`` 079.

        Narrower than :meth:`content_hash` on purpose, and the difference is
        exactly two fields: the set's **name** and each item's **declared
        split**. Neither is an input to extraction — a forward pass reads a
        prompt, not a collection's label for it — so a set that is renamed or
        re-split has activations that remain, item for item, exactly correct.

        Everything that *could* invalidate an activation is still covered: the
        item ids, the question ids, the prompts, the responses, the labels and
        their order. Any edit to those produces a different hash and the cache
        is refused, which is the failure the check exists for.

        This is a deliberate weakening of a guard and is recorded as one. It
        exists so that enlarging a held-out split does not require re-running a
        GPU hour to reproduce identical numbers.
        """
        return content_hash(
            {
                "items": [
                    {
                        "item_id": i.item_id,
                        "question_id": i.question_id,
                        "prompt": i.prompt,
                        "response": i.response,
                        "label": i.label,
                    }
                    for i in self.items
                ]
            }
        )

    @property
    def envelope_id(self) -> str:
        """The warrant key's third element: ``sha256:<16 hex>``.

        Truncated for legibility on screen. Sixteen hex characters is 64 bits,
        which is ample against accidental collision among a handful of eval sets
        and is not being used as a security boundary.
        """
        return f"sha256:{self.content_hash[:16]}"

    @property
    def labels(self) -> np.ndarray:
        """0/1 labels in item order, 1 meaning incorrect."""
        return np.array([i.label for i in self.items], dtype=int)

    @property
    def question_ids(self) -> np.ndarray:
        """Question id per item — the bootstrap's resampling unit."""
        return np.array([i.question_id for i in self.items], dtype=object)

    @property
    def base_rate(self) -> float:
        """Prevalence of the positive class.

        Reported beside AUROC always. At an 85% correct rate a probe that always
        predicts "correct" scores 0.85 accuracy and 0.5 AUROC, and the base rate
        is what stops that reading as signal.
        """
        return float(self.labels.mean())

    def __len__(self) -> int:
        return len(self.items)


def split_by_question(
    evalset: EvalSet,
    *,
    fractions: tuple[float, float, float] = (0.5, 0.25, 0.25),
    seed: int,
) -> dict[str, np.ndarray]:
    """Split into train/validation/test by question, with zero overlap.

    Deduplicates on the normalised question string **before** splitting, so
    near-duplicate phrasings of one question cannot land on opposite sides. Then
    assigns whole questions, so every item sharing a question goes to the same
    split.

    Asserts the result rather than trusting the construction: overlapping splits
    inflate every downstream number and produce no error of their own.

    A set that **declares** its splits is honoured rather than re-split. A frozen
    held-out sample is a decision already made; re-deriving it would both shrink
    it and silently disagree with whatever the declaration was for.

    Args:
        evalset: The set to split.
        fractions: Train, validation, test proportions of *questions*. Used only
            when the set declares no splits of its own.
        seed: Shuffling seed.

    Returns:
        Mapping of split name to row indices. A declared-split set may legally
        return an empty array for a split it does not contain — a pure test set
        has no train rows, and the caller supplies those from elsewhere.

    Raises:
        EvalSetError: If the fractions are invalid, a derived split is empty, or
            any question appears in two splits.
    """
    declared = [item.split for item in evalset.items]
    if all(s is not None for s in declared):
        indices = {
            name: np.flatnonzero([s == name for s in declared]) for name in SPLITS
        }
        groups = evalset.question_ids
        for a in SPLITS:
            for b in SPLITS:
                if a < b and set(groups[indices[a]]) & set(groups[indices[b]]):
                    raise EvalSetError(
                        f"{evalset.eval_set_id}: declared splits put the same "
                        f"question in both {a} and {b}. Splits are by question, "
                        "never by example (CLAUDE.md); an overlap lets the probe "
                        "score itself on what it was fitted on."
                    )
        _LOG.info(
            "%s: using declared splits %s",
            evalset.eval_set_id,
            {k: int(v.size) for k, v in indices.items()},
        )
        return indices
    if any(s is not None for s in declared):
        raise EvalSetError(
            f"{evalset.eval_set_id}: {sum(s is not None for s in declared)} of "
            f"{len(declared)} items declare a split. Either all do or none do; a "
            "partial declaration would silently mix a declared holdout with a "
            "derived one."
        )

    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9:
        raise EvalSetError(f"fractions must be three values summing to 1, got {fractions}")
    if any(f <= 0 for f in fractions):
        raise EvalSetError(f"every split must be non-empty, got fractions {fractions}")

    # Collapse near-duplicate questions onto one grouping key.
    canonical: dict[str, str] = {}
    group_of_item: list[str] = []
    for item in evalset.items:
        key = normalise_question(item.prompt) or item.question_id
        canonical.setdefault(key, item.question_id)
        group_of_item.append(canonical[key])
    groups = np.array(group_of_item, dtype=object)

    unique = np.array(sorted(set(group_of_item)), dtype=object)
    collapsed = len(set(evalset.question_ids.tolist())) - unique.size
    if collapsed > 0:
        _LOG.info(
            "%s: %d question(s) collapsed as near-duplicates before splitting",
            evalset.eval_set_id,
            collapsed,
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(unique.size)
    shuffled = unique[order]
    n_train = int(round(fractions[0] * unique.size))
    n_val = int(round(fractions[1] * unique.size))
    assignment = {
        TRAIN: set(shuffled[:n_train].tolist()),
        VALIDATION: set(shuffled[n_train : n_train + n_val].tolist()),
        TEST: set(shuffled[n_train + n_val :].tolist()),
    }

    indices = {
        name: np.flatnonzero([g in members for g in groups])
        for name, members in assignment.items()
    }
    for name, idx in indices.items():
        if idx.size == 0:
            raise EvalSetError(
                f"{evalset.eval_set_id}: split {name!r} is empty at fractions "
                f"{fractions} with {unique.size} questions"
            )
    for a in SPLITS:
        for b in SPLITS:
            if a < b and set(groups[indices[a]]) & set(groups[indices[b]]):
                raise EvalSetError(
                    f"{evalset.eval_set_id}: question overlap between {a} and {b}"
                )
    total = sum(idx.size for idx in indices.values())
    if total != len(evalset):
        raise EvalSetError(
            f"{evalset.eval_set_id}: splits cover {total} of {len(evalset)} items"
        )
    return indices


@dataclasses.dataclass(frozen=True)
class PaddingEvidence:
    """Activations captured three ways, so the padding control can compare them.

    Captured at extraction, when the model is loaded, and checked at validation,
    which runs from cache. The control's job is to prove the features it is
    validating were produced with left padding, and comparing these three
    tensors does exactly that:

    * ``unbatched`` — one sequence at a time, no padding at all. The reference.
    * ``left_padded`` — the batched path actually used. Must match the reference.
    * ``right_padded`` — the deliberate fault. Must **not** match, and the run
      must reject it.

    Args:
        unbatched: ``(n, hidden_dim)``.
        left_padded: ``(n, hidden_dim)``.
        right_padded: ``(n, hidden_dim)``.
        n_prompts: How many prompts were used.
        max_pad_tokens: Largest number of pad tokens in the batch. A batch of
            equal-length prompts pads nothing and would pass either way, so the
            evidence is worthless unless this is greater than zero.
    """

    unbatched: np.ndarray
    left_padded: np.ndarray
    right_padded: np.ndarray
    n_prompts: int
    max_pad_tokens: int

    def __post_init__(self) -> None:
        shapes = {
            "unbatched": self.unbatched.shape,
            "left_padded": self.left_padded.shape,
            "right_padded": self.right_padded.shape,
        }
        if len(set(shapes.values())) != 1:
            raise EvalSetError(f"padding evidence shapes disagree: {shapes}")
        if self.max_pad_tokens <= 0:
            raise EvalSetError(
                "padding evidence was captured on a batch with no padding, so it "
                "proves nothing. Left and right padding are identical when every "
                "prompt is the same length; the batch must contain prompts of "
                "differing lengths."
            )


@dataclasses.dataclass(frozen=True)
class ExtractionCache:
    """Everything ``/validate`` needs, without loading a model.

    This is what makes the gate's "under a minute from cache" achievable and
    what makes the demo's *Prove it* button possible: extraction is a GPU hour,
    validation is a few seconds of linear algebra, and they are separate stages
    that meet on disk.

    Args:
        eval_set_id: Which set this was extracted from.
        eval_set_hash: Its content hash. Checked on load — a cache whose set has
            changed underneath it is stale, and validating against it would file
            numbers under an envelope that no longer describes the data.
        model_name: The model the activations came from. A probe is pinned to
            it; a model change invalidates every activation-tier warrant
            (``SPEC.md`` §5.4).
        layer: Absolute hidden-state index the activations were taken from.
        data_source: ``"measured"`` or ``"synthetic"``.
        features: Variant name → ``(n_items, n_features)``. Variants are
            ``"T1-mean_pool"``, ``"T1-max_rolling_means"``, ``"T2-logprob"``,
            ``"T3-judge"``.
        labels: ``(n_items,)`` 0/1, 1 meaning incorrect.
        question_ids: ``(n_items,)`` grouping for splits and the bootstrap.
        token_lengths: ``(n_items,)`` prompt lengths, the envelope's
            highest-priority feature.
        padding_evidence: Captured at extraction; consumed by the control.
        extra: Extraction metadata worth keeping beside the numbers.
    """

    eval_set_id: str
    eval_set_hash: str
    model_name: str
    layer: int
    data_source: str
    features: dict[str, np.ndarray]
    labels: np.ndarray
    question_ids: np.ndarray
    token_lengths: np.ndarray
    padding_evidence: Optional[PaddingEvidence] = None
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.features:
            raise EvalSetError(f"{self.eval_set_id}: cache carries no features")
        n = self.labels.size
        for name, matrix in self.features.items():
            if matrix.ndim != 2:
                raise EvalSetError(f"{name}: features must be 2-D, got {matrix.shape}")
            if matrix.shape[0] != n:
                raise EvalSetError(
                    f"{name}: {matrix.shape[0]} feature rows but {n} labels"
                )
            if not np.all(np.isfinite(matrix)):
                raise EvalSetError(
                    f"{name}: features contain NaN or infinity. A probe fitted on "
                    "them either crashes in the solver or silently learns the "
                    "pattern of missingness."
                )
        if self.question_ids.size != n or self.token_lengths.size != n:
            raise EvalSetError(
                f"{self.eval_set_id}: labels, question_ids and token_lengths must "
                "all have one entry per item"
            )
        if self.data_source not in (SOURCE_MEASURED, SOURCE_SYNTHETIC):
            raise EvalSetError(f"unknown data_source {self.data_source!r}")

    @property
    def n_items(self) -> int:
        """Number of items in the cache."""
        return int(self.labels.size)

    @property
    def variants(self) -> tuple[str, ...]:
        """Feature variant names, sorted — the rungs of the tier ladder."""
        return tuple(sorted(self.features))

    def matrix(self, variant: str) -> np.ndarray:
        """Feature matrix for one variant.

        Raises:
            EvalSetError: If the variant is absent, naming what is present. A
                missing tier silently skipped would leave a hole in the ladder
                that reads as "we didn't measure that" rather than as a bug.
        """
        if variant not in self.features:
            raise EvalSetError(
                f"{self.eval_set_id}: no features for {variant!r}; cache holds "
                f"{list(self.variants)}"
            )
        return self.features[variant]

    def save(self, path: str | Path) -> Path:
        """Write the cache to a compressed ``.npz``.

        Not committed — it is large and regenerable (``CONTRIBUTING.md``). The
        artifact that *is* committed is the validation run this cache produced.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "__eval_set_id": np.array(self.eval_set_id),
            "__eval_set_hash": np.array(self.eval_set_hash),
            "__model_name": np.array(self.model_name),
            "__layer": np.array(self.layer),
            "__data_source": np.array(self.data_source),
            "__labels": self.labels,
            "__question_ids": self.question_ids.astype(str),
            "__token_lengths": self.token_lengths,
            "__variants": np.array(self.variants),
        }
        for name, matrix in self.features.items():
            payload[f"feat::{name}"] = matrix
        if self.padding_evidence is not None:
            payload["__pad_unbatched"] = self.padding_evidence.unbatched
            payload["__pad_left"] = self.padding_evidence.left_padded
            payload["__pad_right"] = self.padding_evidence.right_padded
            payload["__pad_n"] = np.array(self.padding_evidence.n_prompts)
            payload["__pad_max"] = np.array(self.padding_evidence.max_pad_tokens)
        np.savez_compressed(out, **payload)
        _LOG.info("wrote extraction cache %s (%d items)", out, self.n_items)
        return out

    @classmethod
    def load(
        cls,
        path: str | Path,
        expected_hash: Optional[str] = None,
        *,
        expected_items: Optional["EvalSet"] = None,
    ) -> "ExtractionCache":
        """Read a cache back, optionally checking it still matches its eval set.

        Two checks, and a caller uses one or the other.

        ``expected_hash`` is the strict one: the set's full content hash, which
        covers its name and its declared splits as well as its data. Use it
        whenever the set is expected to be byte-identical to the one extracted.

        ``expected_items`` is for the case ``DECISIONS.md`` 079 introduces — a
        set that was **re-split** and so has a different content hash while
        holding the same items in the same order. It compares the data the cache
        actually stores, item for item: the count, the question ids and the
        labels. That is a direct comparison rather than a digest, so it is the
        stronger check on everything it covers.

        What it does not cover: a prompt or response edited while its
        ``question_id`` and ``label`` stayed the same, because the cache stores
        activations rather than text. That case is caught upstream instead —
        eval sets are frozen and ``verify_manifest`` re-hashes every file
        against the manifest, so an edited set is found before anything loads a
        cache for it.

        Args:
            path: The ``.npz`` written by :meth:`save`.
            expected_hash: The set's current content hash.
            expected_items: The set whose items the cache must match.

        Raises:
            EvalSetError: If the cache was extracted from different contents than
                the set now has. Validating against a stale cache files numbers
                under an envelope that no longer describes the data, which is
                invariant 9's failure and produces no error of its own.
        """
        with np.load(Path(path), allow_pickle=False) as data:
            variants = [str(v) for v in data["__variants"]]
            features = {v: data[f"feat::{v}"] for v in variants}
            evidence = None
            if "__pad_unbatched" in data:
                evidence = PaddingEvidence(
                    unbatched=data["__pad_unbatched"],
                    left_padded=data["__pad_left"],
                    right_padded=data["__pad_right"],
                    n_prompts=int(data["__pad_n"]),
                    max_pad_tokens=int(data["__pad_max"]),
                )
            cache = cls(
                eval_set_id=str(data["__eval_set_id"]),
                eval_set_hash=str(data["__eval_set_hash"]),
                model_name=str(data["__model_name"]),
                layer=int(data["__layer"]),
                data_source=str(data["__data_source"]),
                features=features,
                labels=data["__labels"],
                question_ids=data["__question_ids"].astype(object),
                token_lengths=data["__token_lengths"],
                padding_evidence=evidence,
            )
        if expected_hash is not None and cache.eval_set_hash != expected_hash:
            raise EvalSetError(
                f"cache at {path} was extracted from {cache.eval_set_id} with hash "
                f"{cache.eval_set_hash[:16]}, but that set now hashes to "
                f"{expected_hash[:16]}. The set changed after extraction; re-extract "
                "rather than validating against a cache that describes different data."
            )
        if expected_items is not None:
            cache._require_same_items(expected_items, path)
        return cache

    def _require_same_items(self, evalset: "EvalSet", path: str | Path) -> None:
        """Assert this cache holds the given set's items, in order.

        Compared elementwise rather than by hash, so the error can name the row
        that diverged. A cache silently describing different rows produces
        plausible numbers under the wrong envelope and raises nothing of its own.
        """
        if self.n_items != len(evalset.items):
            raise EvalSetError(
                f"cache at {path} holds {self.n_items} items but "
                f"{evalset.eval_set_id} has {len(evalset.items)}. These are "
                "different sets, whatever their names."
            )
        expected_questions = np.array(
            [i.question_id for i in evalset.items], dtype=object
        )
        mismatched = np.nonzero(self.question_ids != expected_questions)[0]
        if mismatched.size:
            row = int(mismatched[0])
            raise EvalSetError(
                f"cache at {path} disagrees with {evalset.eval_set_id} at row "
                f"{row}: cache has question {self.question_ids[row]!r}, the set "
                f"has {expected_questions[row]!r}. The item order changed, so "
                "every activation is filed against the wrong row."
            )
        expected_labels = np.array([i.label for i in evalset.items])
        mismatched = np.nonzero(self.labels != expected_labels)[0]
        if mismatched.size:
            row = int(mismatched[0])
            raise EvalSetError(
                f"cache at {path} disagrees with {evalset.eval_set_id} at row "
                f"{row}: cache label {self.labels[row]}, set label "
                f"{expected_labels[row]}. Relabelling changes what every "
                "measured number means."
            )
