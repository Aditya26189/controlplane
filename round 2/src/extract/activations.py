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

import contextlib
import logging
from typing import Any, Iterator, Optional, Sequence

import numpy as np

from ..detectors.aggregation import aggregate
from ..validation.evalsets import PaddingEvidence
from .model import LoadedModel, assert_left_padding

__all__ = [
    "capture_padding_evidence",
    "efficient_attention",
    "extract_activations",
    "generate_answers",
]

_LOG = logging.getLogger(__name__)


@contextlib.contextmanager
def efficient_attention() -> Iterator[None]:
    """Select SDPA's memory-efficient backend for one forward pass.

    Left to itself, ``scaled_dot_product_attention`` uses the *math* backend
    whenever it is handed an explicit attention mask, and math materialises the
    full ``heads x seq x seq`` matrix — 28 x 14500^2 x 2 bytes, about 11 GB for
    a single op. The memory-efficient backend is O(seq) and works on Turing
    (sm_75), which a T4 is. Flash-Attention-2 needs sm_80 and does not apply.

    This is **one of two** large allocations, and it was not the larger one. See
    :func:`_base_model` for the vocabulary-sized logits, which are bigger at
    every sequence length in the band. Fixing attention alone did not make the
    pass fit; both were needed.

    In practice transformers hands SDPA ``attn_mask=None, is_causal=True``
    whenever the padding mask is all ones, so on the supported versions the math
    backend is never reached at batch size 1 anyway. This manager stays as a
    guard for versions that behave differently; it is not load-bearing.

    The manager is *constructed* inside the try, and the body runs outside it.
    The first version wrapped ``yield`` in ``except RuntimeError`` — and
    ``torch.cuda.OutOfMemoryError`` subclasses ``RuntimeError``, so an OOM in the
    body was caught by the fallback handler, which then yielded a second time.
    That produced ``RuntimeError: generator didn't stop after throw()`` and hid
    the real error underneath it, including from the retry logic that was
    watching for OutOfMemoryError.
    """
    import torch

    manager = None
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        manager = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    except (ImportError, AttributeError):
        try:
            manager = torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=True, enable_math=True
            )
        except AttributeError:
            manager = None

    if manager is None:
        _LOG.warning(
            "no way to select the attention backend on this torch; long-context "
            "extraction will pay the math backend's seq^2 cost"
        )
        yield
        return

    with manager:
        yield


def _decoder_layers(model: Any) -> Any:
    """Find the list of transformer blocks, whatever the architecture calls it.

    Qwen2, Llama and Mistral all expose ``model.model.layers``; some wrappers
    nest differently. Raises rather than guessing, because a wrong list means
    the hook attaches to the wrong module and the activations come from a depth
    nobody chose.
    """
    for path in ("model.layers", "transformer.h", "model.decoder.layers"):
        target = model
        for attribute in path.split("."):
            target = getattr(target, attribute, None)
            if target is None:
                break
        if target is not None:
            return target
    raise RuntimeError(
        f"cannot locate the decoder layers on {type(model).__name__}. The hook "
        "needs them to capture one layer without materialising all of them."
    )


def _base_model(model: Any) -> Any:
    """The transformer trunk, without the vocabulary projection on top.

    A ``*ForCausalLM`` wrapper runs ``lm_head`` over **every position**, and
    transformers then upcasts the result to float32 with both copies alive. At
    Qwen2.5-7B's vocabulary of 152,064 that is ``seq x 152064 x 6`` bytes: 9.43
    GiB at an 11k-token sequence, against 6.43 GiB for even the math backend's
    attention matrix. It is the largest allocation in the pass and every byte of
    it is discarded — the probe reads a hidden state from one layer.

    Returns the wrapper itself if no trunk is recognisable, which costs memory
    but is never wrong. Calling the trunk is safe because nothing here reads the
    return value: the activation arrives through a forward hook.
    """
    for path in ("model.decoder", "model", "transformer"):
        target = model
        for attribute in path.split("."):
            target = getattr(target, attribute, None)
            if target is None:
                break
        if target is not None and hasattr(target, "forward"):
            return target
    _LOG.warning(
        "cannot find the transformer trunk on %s; running the full causal LM, "
        "which materialises vocabulary-sized logits that are then discarded",
        type(model).__name__,
    )
    return model


def _hidden_states_at(
    loaded: LoadedModel, prompts: Sequence[str], layer: int, *, padding: str = "left"
) -> tuple[np.ndarray, np.ndarray]:
    """One prefill pass; returns ``(hidden, mask)`` for the requested layer.

    Captures the layer with a **forward hook** rather than
    ``output_hidden_states=True``. That flag returns all ``L+1`` hidden-state
    tensors and we use exactly one, which is 28/29 of the memory thrown away.
    On this model that is 3.1 GB at a 16k-token sequence — against 0.11 GB for
    the single layer — and at 16k on a 16 GB card the difference is the
    headroom attention needs. Measured from Round 1's recorded shapes:
    28 layers, hidden 3584, bfloat16.

    ``layer`` keeps its ``hidden_states`` meaning: 0 is the embedding output and
    L is the output of transformer block L, so block ``layer - 1``'s output is
    what the hook captures. Preserving that indexing matters because the layer
    was chosen by fractional depth against the same convention.

    Args:
        loaded: The model and tokenizer.
        prompts: Rendered prompts, already through the chat template.
        layer: 1-based index into ``hidden_states``.
        padding: ``"left"`` normally. ``"right"`` **only** for the deliberate
            fault injection, and the caller has to ask for it by name.

    Returns:
        ``hidden`` of shape ``(batch, seq, hidden)`` and ``mask`` of shape
        ``(batch, seq)``, both as float/bool numpy on CPU.
    """
    import torch

    if layer < 1:
        raise ValueError(
            f"layer must be >= 1; layer 0 is the embedding output, which carries "
            f"no question-time state. Got {layer}."
        )
    blocks = _decoder_layers(loaded.model)
    if layer > len(blocks):
        raise ValueError(f"layer {layer} exceeds the model's {len(blocks)} blocks")

    tokenizer = loaded.tokenizer
    original_side = tokenizer.padding_side
    tokenizer.padding_side = padding
    if padding == "left":
        assert_left_padding(tokenizer, where="_hidden_states_at")

    captured: dict[str, Any] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        # A decoder block returns a tuple whose first element is the hidden
        # state; some return the tensor directly.
        tensor = output[0] if isinstance(output, tuple) else output
        # .cpu() before .float(): upcasting on the device would allocate
        # seq x hidden x 4 bytes of GPU memory (229 MB at 16k tokens) purely to
        # copy it off again.
        captured["hidden"] = tensor.detach().cpu().to(torch.float32).numpy()

    handle = blocks[layer - 1].register_forward_hook(hook)
    try:
        batch = tokenizer(
            list(prompts), return_tensors="pt", padding=True, truncation=False
        ).to(loaded.model.device)
        mask = batch["attention_mask"].cpu().numpy().astype(bool)

        # A single sequence is never padded, so its mask is all ones and carries
        # no information. Transformers already drops such a mask -- measured, in
        # test_all_ones_mask_is_already_skipped -- and passes is_causal=True to
        # SDPA, so this is insurance against a version that stops doing so, not
        # a fix for anything observed. An earlier comment here claimed it was
        # "what actually makes 16k tokens fit"; that was wrong (DECISIONS 057).
        forward_inputs = dict(batch)
        if len(prompts) == 1:
            forward_inputs.pop("attention_mask", None)

        with torch.no_grad(), efficient_attention():
            _base_model(loaded.model)(**forward_inputs, use_cache=False)
        if "hidden" not in captured:
            raise RuntimeError(
                f"the forward hook on block {layer - 1} never fired. The model "
                "did not run the layer the probe is pinned to, so no activation "
                "was captured."
            )
        hidden = captured["hidden"]
    finally:
        handle.remove()
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
