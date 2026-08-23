"""Synthetic eval sets and caches, for exercising the harness without a GPU.

**These are fixtures, not measurements.** Every set built here carries
``data_source="synthetic"``, which is part of the hashed identity, so its
envelope id differs from the real set's and it occupies a *different cell* in
the warrant matrix. There is no path by which a number measured on a fixture
gets filed under ``triviaqa-600``. That is not a convention to remember; it
falls out of invariant 9 and is asserted in ``test_synthetic_cannot_masquerade``.
``DECISIONS.md`` 027.

What the fixtures are for:

* running ``/validate``, all five controls, and the refusal path end to end on a
  laptop, so the logic is testable now and the GPU run only supplies numbers;
* the smoke test, which must pass on a clean checkout with no model cache;
* the demo's fallback, so a rehearsal does not need a GPU.

What they are **not** for: any number that reaches ``RESULTS.md``, the README or
a slide. The signal strengths below are *parameters chosen by us*, so a tier
ladder measured on them would be a picture of this function, not a finding. The
real ladder needs the real extraction.

The one thing the fixtures reproduce honestly is *mechanism*: sequences carry a
localised signal in a small subspace and are pooled through the real
:mod:`src.detectors.aggregation` code, so what pooling does to that signal as
context grows is arithmetic we can observe rather than a value we typed in.

Note that this does **not** mean the fixture demonstrates the expected
long-context result. It does not, and it is not tuned to. See
``DECISIONS.md`` 028 for what the fixture revealed about max-of-rolling-means
that we did not expect, and why that is a question for the real run rather than
something to tune away here.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..detectors.aggregation import aggregate
from .evalsets import (
    SOURCE_SYNTHETIC,
    EvalItem,
    EvalSet,
    ExtractionCache,
    PaddingEvidence,
)

__all__ = ["DEFAULT_SIGNAL_BY_TIER", "synthetic_cache", "synthetic_evalset"]

_LOG = logging.getLogger(__name__)

#: Signal strength per tier. **Parameters, not findings.** Calibrated only so a
#: short-context fixture lands in a usable band (roughly 0.70-0.80 AUROC) rather
#: than saturating at 1.0. Saturation is the problem being avoided: when every
#: tier scores perfectly, threshold selection, interval width and the refusal
#: criteria all become trivial and the fixture exercises none of them.
#:
#: **No tier ordering is claimed or engineered here.** At these values the
#: logprob variant happens to score above both activation variants, which is not
#: a prediction about the real ladder in either direction. Whether access to
#: activations buys anything over logprobs is exactly what the real ablation
#: measures, and pre-baking the expected answer into a fixture is how a demo ends
#: up showing a result the data never produced.
DEFAULT_SIGNAL_BY_TIER = {
    "T1-mean_pool": 0.92,
    "T1-max_rolling_means": 0.92,
    "T2-logprob": 0.66,
    "T3-judge": 0.48,
}


def synthetic_evalset(
    *,
    eval_set_id: str,
    n_items: int,
    base_rate: float,
    seed: int,
    items_per_question: int = 1,
    long_context: bool = False,
) -> EvalSet:
    """Build a synthetic eval set with a declared base rate.

    Args:
        eval_set_id: Name. Convention is to suffix ``-synthetic`` so it is
            obvious on screen as well as in the hash.
        n_items: How many items.
        base_rate: Fraction labelled 1 (*incorrect*).
        seed: RNG seed.
        items_per_question: Items sharing one question id, so the grouped
            bootstrap and the question-level split have something to group.
        long_context: Whether prompts are long. Recorded in ``construction`` and
            used by :func:`synthetic_cache` to choose sequence lengths.

    Returns:
        A frozen :class:`EvalSet` marked synthetic.
    """
    if not 0.0 < base_rate < 1.0:
        raise ValueError(f"base_rate must be in (0, 1), got {base_rate}")
    rng = np.random.default_rng(seed)
    n_positive = int(round(base_rate * n_items))
    labels = np.zeros(n_items, dtype=int)
    labels[rng.choice(n_items, size=n_positive, replace=False)] = 1

    items = []
    for i in range(n_items):
        question_index = i // max(1, items_per_question)
        # Distinct prompt text per question so the near-duplicate collapse in
        # split_by_question has real strings to work on.
        filler = "context " * (400 if long_context else 4)
        items.append(
            EvalItem(
                item_id=f"{eval_set_id}-{i:05d}",
                question_id=f"q{question_index:05d}",
                prompt=f"{filler}synthetic question {question_index} variant {i}?",
                response=f"synthetic response {i}",
                label=int(labels[i]),
                meta={"synthetic": True},
            )
        )
    return EvalSet(
        eval_set_id=eval_set_id,
        items=tuple(items),
        data_source=SOURCE_SYNTHETIC,
        construction={
            "generator": "src.validation.synthetic.synthetic_evalset",
            "seed": seed,
            "n_items": n_items,
            "requested_base_rate": base_rate,
            "items_per_question": items_per_question,
            "long_context": long_context,
            "warning": (
                "Synthetic fixture. Exercises the harness; is not a measurement. "
                "Numbers from this set must not reach RESULTS.md or the README."
            ),
        },
    )


def synthetic_cache(
    evalset: EvalSet,
    *,
    seed: int,
    hidden_dim: int = 32,
    window: int,
    stride: int,
    signal_by_tier: Optional[dict[str, float]] = None,
    short_len: int = 96,
    long_len: int = 1536,
    signal_span: int = 32,
    signal_dims: int = 4,
    amplitude_spread: float = 0.75,
) -> ExtractionCache:
    """Generate a cache by pooling synthetic sequences through the real code.

    Sequences are Gaussian noise with a localised bump added for items labelled
    *incorrect*. The bump is pooled by the actual
    :func:`~src.detectors.aggregation.aggregate` functions, so the difference
    between mean pooling and max-of-rolling-means on long context is arithmetic
    rather than something asserted here.

    ``signal_by_tier`` sets how strong the bump is for each access tier. **Those
    are parameters, not findings.** A ladder measured on this cache is a picture
    of these numbers; the real ladder requires the real extraction.

    Args:
        evalset: The synthetic set to build a cache for. Must be synthetic.
        seed: RNG seed.
        hidden_dim: Feature width.
        window: Rolling-mean window, from ``config.probe.rolling_window``.
        stride: Rolling-mean stride, from ``config.probe.rolling_stride``.
        signal_by_tier: Bump magnitude per variant.
        short_len: Sequence length for a short-context set.
        long_len: Sequence length for a long-context set.
        signal_span: How many positions the bump covers. Held fixed as context
            grows, which is precisely why mean pooling dilutes it.
        signal_dims: Width of the subspace the signal lives in. A signal spread
            over every dimension is trivially separable however diluted, which
            is not the situation a probe is in.
        amplitude_spread: Per-item amplitude noise as a fraction of ``strength``.
            Controls how much the classes overlap, and therefore where AUROC
            lands. Without overlap every tier scores 1.0 and the fixture
            exercises none of the paths that matter.

    Returns:
        An :class:`ExtractionCache` marked synthetic.

    Raises:
        ValueError: If asked to build a synthetic cache for a measured set,
            which would put fixture numbers under a real envelope.
    """
    if evalset.data_source != SOURCE_SYNTHETIC:
        raise ValueError(
            f"{evalset.eval_set_id} is a {evalset.data_source} set. Refusing to "
            "attach synthetic features to it: the cache would carry the real "
            "set's content hash and its numbers would be filed under the real "
            "envelope."
        )
    signal_by_tier = signal_by_tier or dict(DEFAULT_SIGNAL_BY_TIER)
    rng = np.random.default_rng(seed)
    labels = evalset.labels
    long_context = bool(evalset.construction.get("long_context"))
    seq_len = long_len if long_context else short_len

    n = len(evalset)
    pooled: dict[str, list[np.ndarray]] = {name: [] for name in signal_by_tier}
    token_lengths = np.full(n, seq_len, dtype=int)

    # One fixed direction per variant. A signal spread over every dimension is
    # trivially separable however diluted, which is not the situation a probe is
    # in: the readable direction is a small subspace of the residual stream and
    # the probe has to find it.
    directions = {}
    for variant in signal_by_tier:
        raw = rng.normal(0.0, 1.0, size=hidden_dim)
        raw[signal_dims:] = 0.0
        directions[variant] = raw / np.linalg.norm(raw)

    for i in range(n):
        start = int(rng.integers(0, max(1, seq_len - signal_span)))
        for variant, strength in signal_by_tier.items():
            hidden = rng.normal(0.0, 1.0, size=(seq_len, hidden_dim))
            # Amplitude overlaps between classes, which is what makes AUROC land
            # below 1.0. A deterministic bump would be perfectly separable and
            # would exercise none of the paths that matter.
            centre = strength if labels[i] == 1 else 0.0
            amplitude = rng.normal(centre, strength * amplitude_spread)
            hidden[start : start + signal_span] += amplitude * directions[variant]
            if variant.startswith("T1-"):
                strategy = variant.split("-", 1)[1]
                pooled[variant].append(
                    aggregate(hidden, strategy, None, window=window, stride=stride)
                )
            elif variant == "T2-logprob":
                # A logprob family: mean, min, spread, and the last position.
                summary = hidden @ directions[variant]
                pooled[variant].append(
                    np.array(
                        [summary.mean(), summary.min(), summary.std(), summary[-1]]
                    )
                )
            else:
                # A judge returns one score, not a vector.
                pooled[variant].append(np.array([float(hidden.mean(axis=0) @ directions[variant])]))

    features = {name: np.vstack(rows) for name, rows in pooled.items()}

    # Padding evidence: the left-padded batch reproduces the unbatched reference
    # to floating-point noise; the right-padded one reads position -1 out of the
    # pad region and is unrelated to it. Generated rather than extracted, but the
    # *relationship* is the real one, and the control's arithmetic is identical.
    n_pad_prompts = 4
    reference = rng.normal(0.0, 1.0, size=(n_pad_prompts, hidden_dim))
    evidence = PaddingEvidence(
        unbatched=reference,
        left_padded=reference + rng.normal(0.0, 1e-6, size=reference.shape),
        right_padded=rng.normal(0.0, 1.0, size=reference.shape),
        n_prompts=n_pad_prompts,
        max_pad_tokens=37,
    )

    _LOG.info(
        "synthetic cache for %s: %d items, %d variants, seq_len=%d",
        evalset.eval_set_id,
        n,
        len(features),
        seq_len,
    )
    return ExtractionCache(
        eval_set_id=evalset.eval_set_id,
        eval_set_hash=evalset.content_hash,
        model_name="synthetic-fixture (no model was loaded)",
        layer=-1,
        data_source=SOURCE_SYNTHETIC,
        features=features,
        labels=labels,
        question_ids=evalset.question_ids,
        token_lengths=token_lengths,
        padding_evidence=evidence,
        extra={
            "generator": "src.validation.synthetic.synthetic_cache",
            "seed": seed,
            "signal_by_tier": dict(signal_by_tier),
            "sequence_length": seq_len,
            "signal_span": signal_span,
            "warning": (
                "Synthetic fixture. Signal strengths are parameters chosen by us, "
                "not measurements. A tier ladder computed from this cache "
                "describes this function and nothing else."
            ),
        },
    )
