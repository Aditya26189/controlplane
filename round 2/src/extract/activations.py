"""Question-time activation extraction, and the padding evidence that proves it.

**Question-time means after the prompt, before any generated token exists.** The
whole claim rests on reading the model's state at the moment it has finished
reading the question and has not yet produced anything. So activations come from
a **separate prefill forward pass**, never from inside ``generate()``.

That separation is not stylistic. ``output_hidden_states=True`` inside
``generate()`` retains hidden states for *every generated step* — a 7B model at
32 new tokens over a batch of 8 holds tens of gigabytes and dies on a 16GB T4.
Prefill once for activations, then generate separately for the answer.

**The padding evidence** captured here is what the validation control consumes
(``SPEC.md`` §2.1). The same prompts are scored three ways — unbatched,
left-padded, right-padded — and the control asserts the left batch reproduces
the unbatched reference *and that the right batch does not*. The second half is
what makes it a fault injection rather than an assertion: a check that passes
whatever you feed it proves nothing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from ..detectors.aggregation import aggregate
from ..validation.evalsets import PaddingEvidence
from .model import LoadedModel, assert_left_padding

__all__ = [
    "capture_padding_evidence",
    "extract_activations",
    "generate_answers",
]

_LOG = logging.getLogger(__name__)


def _hidden_states_at(
    loaded: LoadedModel, prompts: Sequence[str], layer: int, *, padding: str = "left"
) -> tuple[np.ndarray, np.ndarray]:
    """One prefill pass; returns ``(hidden, mask)`` for the requested layer.

    Args:
        loaded: The model and tokenizer.
        prompts: Rendered prompts, already through the chat template.
        layer: 1-based index into ``hidden_states``, where 0 is the embedding
            output and L is the output of transformer block L.
        padding: ``"left"`` normally. ``"right"`` **only** for the deliberate
            fault injection, and the caller has to ask for it by name.

    Returns:
        ``hidden`` of shape ``(batch, seq, hidden)`` and ``mask`` of shape
        ``(batch, seq)``, both as float/bool numpy on CPU.
    """
    import torch

    tokenizer = loaded.tokenizer
    original_side = tokenizer.padding_side
    tokenizer.padding_side = padding
    if padding == "left":
        assert_left_padding(tokenizer, where="_hidden_states_at")
    try:
        batch = tokenizer(
            list(prompts), return_tensors="pt", padding=True, truncation=False
        ).to(loaded.model.device)
        with torch.no_grad():
            outputs = loaded.model(
                **batch, output_hidden_states=True, use_cache=False
            )
        hidden = outputs.hidden_states[layer].to(torch.float32).cpu().numpy()
        mask = batch["attention_mask"].cpu().numpy().astype(bool)
    finally:
        tokenizer.padding_side = original_side
    return hidden, mask


def extract_activations(
    loaded: LoadedModel,
    prompts: Sequence[str],
    *,
    layer: int,
    aggregations: Sequence[str],
    window: int,
    stride: int,
    batch_size: int = 8,
    progress: bool = True,
) -> dict[str, np.ndarray]:
    """Pool question-time activations for every prompt, per aggregation.

    Args:
        loaded: Model and tokenizer.
        prompts: Rendered prompts.
        layer: Absolute hidden-state index, resolved from a fractional depth.
        aggregations: Which pooling strategies to compute, from
            ``config.probe.aggregations``. All are computed from the *same*
            forward pass, so a difference between them is a difference in
            pooling and never in the activations underneath.
        window: Rolling-mean window for ``max_rolling_means``.
        stride: Rolling-mean stride.
        batch_size: Prompts per forward pass. Long-context extraction needs a
            small one; the caller sizes it.
        progress: Show a tqdm bar.

    Returns:
        Mapping of ``"T1-<aggregation>"`` to ``(n_prompts, hidden_size)``.
    """
    assert_left_padding(loaded.tokenizer, where="extract_activations")
    pooled: dict[str, list[np.ndarray]] = {
        f"T1-{name}": [] for name in aggregations
    }

    iterator = range(0, len(prompts), batch_size)
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc=f"activations L{layer}", unit="batch")
        except ImportError:  # pragma: no cover
            pass

    for start in iterator:
        chunk = prompts[start : start + batch_size]
        hidden, mask = _hidden_states_at(loaded, chunk, layer, padding="left")
        for row in range(hidden.shape[0]):
            real = mask[row]
            sequence = hidden[row]
            for name in aggregations:
                pooled[f"T1-{name}"].append(
                    aggregate(sequence, name, real, window=window, stride=stride)
                )

    features = {name: np.vstack(rows) for name, rows in pooled.items()}
    for name, matrix in features.items():
        if not np.all(np.isfinite(matrix)):
            raise RuntimeError(
                f"{name}: extracted activations contain NaN or infinity. A probe "
                "fitted on them either crashes in the solver or silently learns "
                "the pattern of missingness."
            )
    _LOG.info(
        "extracted %s at layer %d for %d prompts",
        list(features),
        layer,
        len(prompts),
    )
    return features


def capture_padding_evidence(
    loaded: LoadedModel, prompts: Sequence[str], *, layer: int
) -> PaddingEvidence:
    """Score the same prompts three ways so the control has something to check.

    Uses the **last real position** of each sequence in every variant, which is
    the position the whole method reads. Under left padding that is index ``-1``
    for every row; under right padding index ``-1`` is a pad token for every row
    but the longest, which is exactly the fault being injected.

    The prompts must differ in length or the evidence proves nothing — left and
    right padding are identical when nothing is padded — and
    :class:`PaddingEvidence` refuses to be constructed without padding present.

    Args:
        loaded: Model and tokenizer.
        prompts: A handful of prompts of **differing lengths**.
        layer: Which layer to compare at.

    Returns:
        A :class:`PaddingEvidence` for the extraction cache.
    """
    unbatched_rows = []
    for prompt in prompts:
        hidden, mask = _hidden_states_at(loaded, [prompt], layer, padding="left")
        real = np.flatnonzero(mask[0])
        unbatched_rows.append(hidden[0][real[-1]])
    unbatched = np.vstack(unbatched_rows)

    left_hidden, left_mask = _hidden_states_at(loaded, prompts, layer, padding="left")
    left_rows = [
        left_hidden[row][np.flatnonzero(left_mask[row])[-1]]
        for row in range(left_hidden.shape[0])
    ]
    left_padded = np.vstack(left_rows)

    # The deliberate fault: take position -1 unconditionally, exactly as a
    # careless implementation would, so the pad tokens are read as content.
    right_hidden, right_mask = _hidden_states_at(
        loaded, prompts, layer, padding="right"
    )
    right_padded = np.vstack([right_hidden[row][-1] for row in range(right_hidden.shape[0])])

    lengths = right_mask.sum(axis=1)
    max_pad = int(right_mask.shape[1] - lengths.min())
    _LOG.info(
        "padding evidence: %d prompts, lengths %s, max pad %d tokens",
        len(prompts),
        lengths.tolist(),
        max_pad,
    )
    return PaddingEvidence(
        unbatched=unbatched,
        left_padded=left_padded,
        right_padded=right_padded,
        n_prompts=len(prompts),
        max_pad_tokens=max_pad,
    )


def generate_answers(
    loaded: LoadedModel,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 32,
    batch_size: int = 8,
    progress: bool = True,
) -> list[str]:
    """Greedy generation, separate from the activation pass.

    Greedy because the label has to be reproducible: a sampled answer makes the
    same question correct on one run and incorrect on the next, and every
    measured number would move between runs for reasons unrelated to the probe.

    ``output_hidden_states`` is **not** set here. Retaining hidden states across
    generated steps is what exhausts a 16GB card, and the activations were
    already taken at question time by :func:`extract_activations`.

    Args:
        loaded: Model and tokenizer.
        prompts: Rendered prompts.
        max_new_tokens: Answer length cap. TriviaQA answers are short; a long
            cap wastes compute and increases the chance a gold alias appears
            incidentally in surrounding prose.
        batch_size: Prompts per call.
        progress: Show a tqdm bar.

    Returns:
        The generated continuations, with the prompt stripped.
    """
    import torch

    assert_left_padding(loaded.tokenizer, where="generate_answers")
    tokenizer = loaded.tokenizer
    answers: list[str] = []

    iterator = range(0, len(prompts), batch_size)
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc="generating", unit="batch")
        except ImportError:  # pragma: no cover
            pass

    for start in iterator:
        chunk = list(prompts[start : start + batch_size])
        batch = tokenizer(chunk, return_tensors="pt", padding=True).to(
            loaded.model.device
        )
        with torch.no_grad():
            generated = loaded.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Left padding means every sequence's prompt ends at the same index, so
        # the continuation starts at the input width for all rows.
        continuation = generated[:, batch["input_ids"].shape[1] :]
        answers.extend(
            tokenizer.batch_decode(continuation, skip_special_tokens=True)
        )
    return [answer.strip() for answer in answers]
