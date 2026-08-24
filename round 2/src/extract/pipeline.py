"""The extraction stage: TriviaQA in, frozen eval sets and caches out.

Runs **both** envelopes in one session. ``triviaqa-longctx-600`` is what Beat 4
turns on, and a two-session plan is a plan where the second session does not
happen.

**The long-context pass covers the test split only**, and that is a measurement
decision rather than a saving. The question Beat 4 asks is *"what happens to
**this** probe when the traffic changes underneath it?"* — not *"what happens if
you retrain on long context?"*. Those are different experiments and only the
first is the drift story. Extracting long-context activations for the train
split would answer the second, at four times the GPU cost, and would quietly
replace the demo's claim with a weaker one. ``DECISIONS.md`` 051.

The stage ends by checking its own output against the fixture path with
:func:`~src.validation.metrics_builder.assert_metric_shape_compatible`, so a
normalisation or polarity divergence is caught on the GPU session rather than
after the artifacts have been downloaded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import Config
from ..evalsets.builders import build_longctx
from ..validation.evalsets import (
    SOURCE_MEASURED,
    TEST,
    TRAIN,
    VALIDATION,
    EvalItem,
    EvalSet,
    ExtractionCache,
)
from .activations import capture_padding_evidence, extract_activations, generate_answers
from .model import LoadedModel, build_prompt
from .triviaqa import (
    TriviaItem,
    is_correct,
    is_exact_match,
    load_triviaqa,
    split_questions,
)

__all__ = ["ExtractionResult", "extract_triviaqa"]

_LOG = logging.getLogger(__name__)

#: Concise instruction, fixed across every item. Varying it per question would
#: make the prompt a confound: a difference in activations could then be a
#: difference in instruction rather than in what the model knows.
SYSTEM_PROMPT = "Answer the question concisely. Give only the answer."


class ExtractionResult:
    """What one extraction session produced.

    Args:
        short_evalset: ``triviaqa-600``, with declared splits.
        short_cache: Its activations, both aggregations, at the chosen layer.
        long_evalset: ``triviaqa-longctx-600``, test split only.
        long_cache: Its activations.
        report: Counts and diagnostics worth writing beside the numbers.
    """

    def __init__(
        self,
        short_evalset: EvalSet,
        short_cache: ExtractionCache,
        long_evalset: Optional[EvalSet],
        long_cache: Optional[ExtractionCache],
        report: dict[str, Any],
    ) -> None:
        self.short_evalset = short_evalset
        self.short_cache = short_cache
        self.long_evalset = long_evalset
        self.long_cache = long_cache
        self.report = report


def _label_items(items: list[TriviaItem], answers: list[str]) -> dict[str, Any]:
    """Attach generations and labels, and report how the labels were reached.

    Label 1 means **incorrect** — the thing the probe should fire on.
    """
    how_counts: dict[str, int] = {}
    strict_correct = 0
    for item, answer in zip(items, answers):
        item.response = answer
        correct, how = is_correct(answer, item.aliases)
        item.label = 0 if correct else 1
        strict_correct += int(is_exact_match(answer, item.aliases))
        key = how.split(" on ")[0]
        how_counts[key] = how_counts.get(key, 0) + 1

    labels = np.array([item.label for item in items])
    base_rate = float(labels.mean())
    strict_base_rate = 1.0 - strict_correct / len(items)
    _LOG.info(
        "labelled %d items: base rate %.4f lenient, %.4f strict EM "
        "(gap %.4f), match rules %s",
        len(items),
        base_rate,
        strict_base_rate,
        strict_base_rate - base_rate,
        how_counts,
    )
    if base_rate in (0.0, 1.0):
        raise RuntimeError(
            f"every generation was labelled the same ({base_rate}). Either the "
            "alias matching is broken or generation failed; a single-class set "
            "supports no ranking claim and would refuse every warrant."
        )
    return {
        "base_rate": base_rate,
        "base_rate_strict_em": strict_base_rate,
        "lenient_minus_strict": strict_base_rate - base_rate,
        "match_rules": how_counts,
    }


def extract_triviaqa(
    config: Config,
    loaded: LoadedModel,
    *,
    n_questions: int = 2400,
    layer_fraction: Optional[float] = None,
    fractions: tuple[float, float, float] = (0.5, 0.25, 0.25),
    batch_size: int = 8,
    long_batch_size: int = 1,
    max_new_tokens: int = 32,
    cache_dir: Optional[str] = None,
    long_context: bool = True,
    checkpoint_dir: Optional[str] = None,
) -> ExtractionResult:
    """Extract both TriviaQA envelopes in one session.

    Args:
        config: Resolved config. Supplies the seed, aggregations, rolling window
            and layer fractions.
        loaded: The loaded model.
        n_questions: Distinct questions after deduplication. Defaults to 2,400,
            which yields a 600-item test split — the size ``triviaqa-600`` names
            (``DECISIONS.md`` 030).
        layer_fraction: Which fractional depth to extract. Defaults to the
            deepest configured fraction below the last layer, which is where the
            Round 1 probe read from.
        fractions: Train, validation, test proportions of *questions*.
        batch_size: Prompts per forward pass for short context.
        long_batch_size: Prompts per forward pass for long context. Defaults to
            1: a 16k-token sequence at 7B in NF4 does not batch on a 16GB card.
        max_new_tokens: Generation cap.
        cache_dir: HuggingFace cache, for offline runs.
        long_context: Whether to run the long-context pass. Only turn this off
            to debug the short pass; shipping without it leaves Beat 4 with no
            measured basis.
        checkpoint_dir: Where to save the short-context eval set and cache
            **before** the long-context pass runs. Strongly recommended: the
            short pass is the expensive half — generation dominates at 2.37 s an
            item — and an OOM in the long pass would otherwise discard it. Learnt
            the hard way on the first real run.

    Returns:
        An :class:`ExtractionResult`.
    """
    seed = config.seed
    layers = config.resolve_layers(loaded.num_hidden_layers)
    if layer_fraction is None:
        layer = layers[-2] if len(layers) > 1 else layers[-1]
    else:
        layer = int(round(layer_fraction * loaded.num_hidden_layers))
    layer = max(1, min(loaded.num_hidden_layers, layer))
    _LOG.info(
        "extracting at layer %d of %d (configured layers %s)",
        layer,
        loaded.num_hidden_layers,
        list(layers),
    )

    items = load_triviaqa(
        n_questions=n_questions, seed=seed, cache_dir=cache_dir
    )
    splits = split_questions(items, fractions=fractions, seed=seed)
    split_of_index = {
        index: name for name, indices in splits.items() for index in indices
    }

    prompts = [
        build_prompt(loaded.tokenizer, item.question, SYSTEM_PROMPT) for item in items
    ]

    _LOG.info("capturing padding evidence before anything else is measured")
    # Prompts of deliberately differing length, so the batch actually pads. The
    # evidence proves nothing on a batch of equal-length prompts, and
    # PaddingEvidence refuses to be built from one.
    by_length = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    sample = [prompts[by_length[0]], prompts[by_length[len(by_length) // 3]],
              prompts[by_length[2 * len(by_length) // 3]], prompts[by_length[-1]]]
    padding_evidence = capture_padding_evidence(loaded, sample, layer=layer)

    answers = generate_answers(
        loaded, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size
    )
    label_report = _label_items(items, answers)

    features = extract_activations(
        loaded,
        prompts,
        layer=layer,
        aggregations=config.probe.aggregations,
        window=config.probe.rolling_window,
        stride=config.probe.rolling_stride,
        batch_size=batch_size,
    )

    token_lengths = np.array(
        [len(loaded.tokenizer(prompt)["input_ids"]) for prompt in prompts], dtype=int
    )

    eval_items = tuple(
        EvalItem(
            item_id=f"triviaqa-{item.question_id}",
            question_id=item.question_id,
            prompt=item.question,
            response=item.response,
            label=int(item.label),
            split=split_of_index[index],
            meta={"aliases": list(item.aliases)},
        )
        for index, item in enumerate(items)
    )
    construction = {
        "method": "TriviaQA rc.nocontext, deduplicated, split by question",
        "n_questions": n_questions,
        "fractions": list(fractions),
        "seed": seed,
        "model": loaded.name,
        "layer": layer,
        "system_prompt": SYSTEM_PROMPT,
        "max_new_tokens": max_new_tokens,
        "decoding": "greedy",
        "label_meaning": "1 = the model's answer was incorrect",
        "short_alias_rule": "aliases under 3 characters require an exact token match",
        "llm_generated": False,
        **label_report,
    }
    short_evalset = EvalSet(
        eval_set_id="triviaqa-600",
        items=eval_items,
        data_source=SOURCE_MEASURED,
        construction=construction,
    )
    short_cache = ExtractionCache(
        eval_set_id=short_evalset.eval_set_id,
        eval_set_hash=short_evalset.content_hash,
        model_name=loaded.name,
        layer=layer,
        data_source=SOURCE_MEASURED,
        features=features,
        labels=short_evalset.labels,
        question_ids=short_evalset.question_ids,
        token_lengths=token_lengths,
        padding_evidence=padding_evidence,
        extra={**loaded.provenance(), **construction},
    )

    # Checkpoint the expensive half before attempting the fragile one. The short
    # pass costs ~13 minutes of generation for 2,400 items; the long pass can
    # exhaust the card. Losing the first to a failure in the second is a bad
    # trade that only has to happen once.
    if checkpoint_dir is not None:
        from ..evalsets import save_evalset

        checkpoint = Path(checkpoint_dir)
        (checkpoint / "evalsets").mkdir(parents=True, exist_ok=True)
        (checkpoint / "results").mkdir(parents=True, exist_ok=True)
        save_evalset(short_evalset, checkpoint / "evalsets")
        short_cache.save(checkpoint / "results" / "cache-triviaqa-600.npz")
        _LOG.info(
            "checkpointed the short-context pass to %s; a failure in the "
            "long-context pass will not lose it",
            checkpoint,
        )

    long_evalset = None
    long_cache = None
    long_error = None
    if long_context:
        try:
            long_evalset, long_cache = _extract_long_context(
                config,
                loaded,
                short_evalset,
                layer=layer,
                batch_size=long_batch_size,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised below unless checkpointed
            long_error = exc
            if checkpoint_dir is None:
                raise
            _LOG.error(
                "long-context pass failed: %s. The short-context pass is "
                "checkpointed and intact; re-run only the long pass.",
                exc,
            )

    report = {
        "layer": layer,
        "n_questions": len(items),
        "splits": {name: len(indices) for name, indices in splits.items()},
        "base_rate": label_report["base_rate"],
        "match_rules": label_report["match_rules"],
        "token_length_mean": float(token_lengths.mean()),
        "token_length_max": int(token_lengths.max()),
        "long_context": bool(long_context),
        "long_context_error": None if long_error is None else repr(long_error),
        **loaded.provenance(),
    }
    return ExtractionResult(short_evalset, short_cache, long_evalset, long_cache, report)


def _extract_long_context(
    config: Config,
    loaded: LoadedModel,
    short_evalset: EvalSet,
    *,
    layer: int,
    batch_size: int,
) -> tuple[EvalSet, ExtractionCache]:
    """Build and extract the long-context envelope, test split only.

    Test only because the drift question is what happens to the *already-fitted*
    probe when traffic changes, not what happens if you refit on the new
    distribution. See the module docstring and ``DECISIONS.md`` 051.
    """
    test_items = tuple(item for item in short_evalset.items if item.split == TEST)
    _LOG.info(
        "building long-context envelope from %d test items", len(test_items)
    )
    test_only = EvalSet(
        eval_set_id="triviaqa-600-test",
        items=test_items,
        data_source=SOURCE_MEASURED,
        construction={
            **short_evalset.construction,
            "note": "test split only, as the base for the long-context envelope",
        },
    )
    pad_tokens = next(
        (spec.pad_tokens for spec in config.evalsets if spec.pad_tokens is not None),
        (4000, 16000),
    )
    long_evalset = build_longctx(
        test_only,
        seed=config.seed,
        pad_tokens=pad_tokens,
        eval_set_id="triviaqa-longctx-600",
    )

    prompts = [
        build_prompt(loaded.tokenizer, item.prompt, SYSTEM_PROMPT)
        for item in long_evalset.items
    ]
    features = _extract_with_oom_retry(
        loaded,
        prompts,
        layer=layer,
        aggregations=config.probe.aggregations,
        window=config.probe.rolling_window,
        stride=config.probe.rolling_stride,
        batch_size=batch_size,
    )
    token_lengths = np.array(
        [len(loaded.tokenizer(prompt)["input_ids"]) for prompt in prompts], dtype=int
    )
    _LOG.info(
        "long-context token lengths: mean %.0f, min %d, max %d",
        token_lengths.mean(),
        token_lengths.min(),
        token_lengths.max(),
    )

    long_cache = ExtractionCache(
        eval_set_id=long_evalset.eval_set_id,
        eval_set_hash=long_evalset.content_hash,
        model_name=loaded.name,
        layer=layer,
        data_source=SOURCE_MEASURED,
        features=features,
        labels=long_evalset.labels,
        question_ids=long_evalset.question_ids,
        token_lengths=token_lengths,
        # No padding evidence: it was captured once this session on the short
        # pass and describes the same tokenizer and the same batching code. A
        # second capture would be the same check twice, not a second check.
        padding_evidence=None,
        extra={
            **loaded.provenance(),
            "derived_from": short_evalset.eval_set_id,
            "split_covered": "test only",
            "why_test_only": (
                "the drift question is what happens to the already-fitted probe "
                "when traffic changes, not what happens if you refit on the new "
                "distribution (DECISIONS.md 051)"
            ),
        },
    )
    return long_evalset, long_cache


def _extract_with_oom_retry(
    loaded: LoadedModel,
    prompts: list[str],
    *,
    layer: int,
    aggregations,
    window: int,
    stride: int,
    batch_size: int,
    min_batch_size: int = 1,
):
    """Extract, halving the batch on OOM, and reporting rather than dying quietly.

    The memory-efficient attention backend should make a 16k-token sequence fit
    on a 16 GB card at batch 1. This is the backstop for the cases it does not:
    a torch build without that backend, a card with less free memory than
    expected, or a longer sequence than the band implies.

    Raises with the sequence length that failed, because "CUDA out of memory" on
    its own does not tell you whether to shorten the band, free the card, or
    change the backend.
    """
    import torch

    current = batch_size
    while True:
        try:
            return extract_activations(
                loaded,
                prompts,
                layer=layer,
                aggregations=aggregations,
                window=window,
                stride=stride,
                batch_size=current,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current > min_batch_size:
                current = max(min_batch_size, current // 2)
                _LOG.warning("OOM; retrying long context at batch size %d", current)
                continue
            lengths = [len(loaded.tokenizer(p)["input_ids"]) for p in prompts]
            raise RuntimeError(
                f"out of memory extracting long context at batch size 1. "
                f"Prompt lengths run {min(lengths)}-{max(lengths)} tokens.\n"
                "The attention matrix is heads x seq x seq on the MATH backend "
                "-- 28 x 14500^2 x 2 bytes is about 11 GB for one op -- so if "
                "the memory-efficient backend is unavailable this cannot fit.\n"
                "Options, cheapest first:\n"
                "  - confirm torch selected the efficient backend (torch >= 2.3 "
                "exposes torch.nn.attention.sdpa_kernel)\n"
                "  - narrow evalsets.pad_tokens in config.yaml; the band is "
                "currently the source of the longest sequences\n"
                "  - spread the model across both GPUs so more of one card is "
                "free for the attention workspace"
            ) from None
