"""Question-time activation extraction, greedy generation, and labelling.

This is the core of the experiment (SPEC.md §4). Two passes per batch:

1. **Prefill only**, a plain forward with ``output_hidden_states=True``. Every
   layer arrives in one call, which makes the layer sweep essentially free.
2. **Generation**, a standard ``generate()`` call, for the labels.

They are deliberately not combined. Passing ``output_hidden_states=True`` into
``generate()`` retains hidden states for every decoding step and exhausts a T4;
the extra prefill is one forward pass against ~32 decode steps.

The activation is taken at the last position of the prompt, before any
generated token exists (CLAUDE.md invariant 1), which is only the true final
prompt token if the batch is left-padded (invariant 4). That is asserted before
every batched call, and verified numerically once per run by
:func:`check_left_padding_equivalence`.
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import Config
from src.data import label_frame
from src.model import assert_left_padding, build_prompts

LOGGER = logging.getLogger(__name__)


class BaseRateError(RuntimeError):
    """Raised when the labelled base rate falls outside the configured band.

    A degenerate label distribution makes every downstream number meaningless,
    so TASKS.md Stage 3 says stop and report rather than continue.
    """


class EquivalenceCheckError(RuntimeError):
    """Raised when batched and unbatched activations disagree.

    Its own type because this is the check that catches the padding failure,
    and the padding failure is silent.
    """


# --------------------------------------------------------------------------- #
# Activation reads
# --------------------------------------------------------------------------- #


def last_token_activations(
    outputs: Any, layers: Sequence[int], expected_hidden_layers: int
) -> dict[int, np.ndarray]:
    """Read the final-position hidden state at each requested layer.

    ``outputs.hidden_states`` is a tuple of length ``n_layers + 1`` where index
    0 is the embedding output and index L is the output of transformer block L
    (SPEC.md §4). Position ``-1`` is the true final prompt token only under left
    padding; the caller is responsible for having asserted that.

    Args:
        outputs: The return value of a forward pass with hidden states on.
        layers: Absolute layer indices to read.
        expected_hidden_layers: ``model.config.num_hidden_layers``, for the
            shape assertion.

    Returns:
        Layer index -> ``(batch, hidden_size)`` float32 array.

    Raises:
        AssertionError: on an unexpected hidden-state count, an out-of-range
            layer, or a non-finite value.
    """
    hidden_states = outputs.hidden_states
    if len(hidden_states) != expected_hidden_layers + 1:
        raise AssertionError(
            f"expected {expected_hidden_layers + 1} hidden-state tensors "
            f"(embeddings + {expected_hidden_layers} blocks), got {len(hidden_states)}"
        )
    out: dict[int, np.ndarray] = {}
    for layer in layers:
        if not 1 <= layer <= expected_hidden_layers:
            raise AssertionError(
                f"layer {layer} out of range for a {expected_hidden_layers}-layer model"
            )
        vector = hidden_states[layer][:, -1, :].float().detach().cpu().numpy()
        if not np.isfinite(vector).all():
            raise AssertionError(
                f"layer {layer}: activation contains NaN or Inf, which would "
                "silently poison the probe's standardisation"
            )
        out[layer] = vector
    return out


# --------------------------------------------------------------------------- #
# The left-padding equivalence check (TASKS.md Stage 3, item 1)
# --------------------------------------------------------------------------- #


def batched_unbatched_deviation(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    layers: Sequence[int],
) -> tuple[float, dict[int, float]]:
    """Max absolute deviation between batched and unbatched last-token states.

    Computes only; it deliberately does not assert the padding side, so a test
    can point it at a right-padded tokenizer and confirm the deviation really
    does blow up. :func:`check_left_padding_equivalence` is the asserting
    wrapper used in the pipeline.

    Args:
        model: A loaded causal LM.
        tokenizer: Its tokenizer.
        prompts: Prompts of differing lengths -- equal lengths would make the
            comparison vacuous, since there would be no padding to get wrong.
        layers: Absolute layer indices to compare.

    Returns:
        ``(max deviation over all layers, per-layer max deviation)``.
    """
    import torch

    n_layers = int(model.config.num_hidden_layers)

    enc = tokenizer(list(prompts), return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        batched_out = model(**enc, output_hidden_states=True, use_cache=False)
    batched = last_token_activations(batched_out, layers, n_layers)

    singles: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    for prompt in prompts:
        one = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            single_out = model(**one, output_hidden_states=True, use_cache=False)
        for layer, vector in last_token_activations(single_out, layers, n_layers).items():
            singles[layer].append(vector[0])

    per_layer: dict[int, float] = {}
    for layer in layers:
        stacked = np.stack(singles[layer], axis=0)
        per_layer[layer] = float(np.max(np.abs(batched[layer] - stacked)))
    return max(per_layer.values()), per_layer


def check_left_padding_equivalence(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    layers: Sequence[int],
    tolerance: float = 1e-2,
) -> dict[str, Any]:
    """Verify batched activations equal unbatched ones, and fail hard if not.

    The single highest-value check in the repo. Attention masking makes the two
    identical in principle and near-identical in half precision, so a deviation
    beyond tolerance means the batch is being read at the wrong position --
    almost always right padding, in which case every activation in the run is a
    pad token's and the resulting AUROC of ~0.5 reads as a negative finding
    rather than as a bug.

    Args:
        model: A loaded causal LM.
        tokenizer: Its left-padded tokenizer.
        prompts: A small batch of ragged-length prompts (4 is the configured
            default in TASKS.md).
        layers: Absolute layer indices to check.
        tolerance: Maximum permitted absolute deviation.

    Returns:
        A dict with the max deviation, the per-layer deviations and the
        tolerance, for the run's metadata artifact.

    Raises:
        EquivalenceCheckError: if any layer exceeds the tolerance.
    """
    assert_left_padding(tokenizer)
    lengths = {len(tokenizer(p)["input_ids"]) for p in prompts}
    if len(lengths) < 2:
        raise EquivalenceCheckError(
            "equivalence check needs prompts of differing token lengths; with "
            "equal lengths there is no padding and the check proves nothing"
        )

    max_deviation, per_layer = batched_unbatched_deviation(
        model, tokenizer, prompts, layers
    )
    LOGGER.info(
        "left-padding equivalence: max deviation %.3e across layers %s (tolerance %.1e)",
        max_deviation,
        list(layers),
        tolerance,
    )
    if max_deviation > tolerance:
        raise EquivalenceCheckError(
            f"batched and unbatched activations differ by {max_deviation:.3e}, "
            f"above tolerance {tolerance:.1e}. Per-layer: "
            f"{ {k: round(v, 6) for k, v in per_layer.items()} }. "
            "This almost always means the batch is right-padded, so position -1 "
            "is a pad token and every extracted activation is meaningless "
            "(CLAUDE.md invariant 4)."
        )
    return {
        "max_deviation": max_deviation,
        "per_layer_deviation": {str(k): v for k, v in per_layer.items()},
        "tolerance": tolerance,
        "n_prompts": len(prompts),
        "distinct_prompt_lengths": len(lengths),
    }


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def select_equivalence_prompts(
    tokenizer: Any, prompts: Sequence[str], n: int = 4
) -> list[str]:
    """Pick the ``n`` prompts that maximise padding in one batch.

    The check's sensitivity scales with how much padding there is: batching four
    prompts of near-identical length pads by a handful of tokens and would still
    look fine if position -1 were slightly wrong. Taking the shortest and the
    longest, plus an even spread between them, puts the maximum available
    padding into the batch -- which is the condition most likely to expose a
    padding fault.

    Args:
        tokenizer: Tokenizer used to measure prompt length.
        prompts: Candidate prompts, ideally the whole run's.
        n: How many to select.

    Returns:
        ``n`` prompts spanning the length distribution, shortest first.
    """
    if len(prompts) <= n:
        return list(prompts)
    lengths = np.array([len(tokenizer(p)["input_ids"]) for p in prompts])
    ranked = np.argsort(lengths, kind="stable")
    picks = np.unique(np.linspace(0, len(ranked) - 1, n).round().astype(int))
    return [prompts[int(ranked[i])] for i in picks]


def _length_sorted_order(tokenizer: Any, prompts: Sequence[str]) -> np.ndarray:
    """Indices that sort prompts by token length, to cut padding waste.

    Stable sort so the order is reproducible for equal lengths. The original
    order is restored afterwards and the restoration is asserted by carrying
    ``question_id`` through (TASKS.md Stage 3 gate).
    """
    lengths = np.array([len(tokenizer(p)["input_ids"]) for p in prompts])
    return np.argsort(lengths, kind="stable")


def extract_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    layers: Sequence[int],
    config: Config,
) -> tuple[dict[int, np.ndarray], list[str], dict[str, float]]:
    """Run the two passes for one batch: prefill for activations, then generate.

    Args:
        model: A loaded causal LM.
        tokenizer: Its left-padded tokenizer.
        prompts: The batch's rendered prompts.
        layers: Absolute layer indices to extract.
        config: Resolved experiment config.

    Returns:
        ``(layer -> (batch, hidden) activations, completions, timings in seconds)``.
    """
    import torch

    assert_left_padding(tokenizer)  # invariant 4, re-checked per batch
    n_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)

    enc = tokenizer(list(prompts), return_tensors="pt", padding=True).to(model.device)

    # Pass 1 -- activations. use_cache=False: nothing downstream needs the KV
    # cache and keeping it wastes memory the generate() call is about to want.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True, use_cache=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - t0

    activations = last_token_activations(outputs, layers, n_layers)
    for layer, array in activations.items():
        if array.shape != (len(prompts), hidden_size):
            raise AssertionError(
                f"layer {layer}: expected shape {(len(prompts), hidden_size)}, "
                f"got {array.shape}"
            )
    del outputs

    # Pass 2 -- labels.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **enc,
            max_new_tokens=config.generation.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generate_seconds = time.perf_counter() - t1

    prompt_len = enc["input_ids"].shape[1]
    completions = tokenizer.batch_decode(
        generated[:, prompt_len:], skip_special_tokens=True
    )
    if len(completions) != len(prompts):
        raise AssertionError(
            f"decoded {len(completions)} completions for {len(prompts)} prompts"
        )

    timings = {
        "prefill_seconds": prefill_seconds,
        "generate_seconds": generate_seconds,
        "batch_size": float(len(prompts)),
        "padded_length": float(prompt_len),
    }
    return activations, [c.strip() for c in completions], timings


def run_extraction(
    model: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    config: Config,
    layers: Sequence[int],
    progress: bool = True,
) -> tuple[dict[int, np.ndarray], pd.DataFrame, dict[str, Any]]:
    """Extract activations and generate completions for a whole split frame.

    Prompts are processed in length-sorted batches to cut padding waste, then
    restored to the input order. The restoration is asserted by carrying
    ``question_id`` through the permutation and checking it round-trips -- an
    off-by-one there would silently pair every activation with the wrong label.

    Args:
        model: A loaded causal LM.
        tokenizer: Its left-padded tokenizer.
        frame: Split frame from ``data.prepare_dataset``.
        config: Resolved experiment config.
        layers: Absolute layer indices to extract.
        progress: Show a tqdm bar.

    Returns:
        ``(layer -> (n, hidden) activations in frame order, labelled frame, meta)``.
    """
    from tqdm.auto import tqdm

    prompts = build_prompts(tokenizer, frame["question"].tolist(), config)
    n = len(prompts)
    hidden_size = int(model.config.hidden_size)

    if config.generation.sort_by_length:
        order = _length_sorted_order(tokenizer, prompts)
    else:
        order = np.arange(n)

    sorted_prompts = [prompts[i] for i in order]
    sorted_qids = [frame["question_id"].iloc[i] for i in order]

    batch_size = config.generation.batch_size
    collected: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    completions_sorted: list[str] = []
    batch_timings: list[dict[str, float]] = []

    started = time.perf_counter()
    batches = range(0, n, batch_size)
    iterator = tqdm(batches, desc="extract", unit="batch", disable=not progress)
    for start in iterator:
        chunk = sorted_prompts[start : start + batch_size]
        activations, completions, timings = extract_batch(
            model, tokenizer, chunk, layers, config
        )
        for layer in layers:
            collected[layer].append(activations[layer])
        completions_sorted.extend(completions)
        batch_timings.append(timings)
    total_seconds = time.perf_counter() - started

    inverse = np.argsort(order, kind="stable")
    acts: dict[int, np.ndarray] = {}
    for layer in layers:
        stacked = np.concatenate(collected[layer], axis=0)
        if stacked.shape != (n, hidden_size):
            raise AssertionError(
                f"layer {layer}: stacked shape {stacked.shape} != {(n, hidden_size)}"
            )
        acts[layer] = stacked[inverse]

    completions = [completions_sorted[i] for i in inverse]

    # Order restoration, asserted rather than trusted.
    restored_qids = [sorted_qids[i] for i in inverse]
    if restored_qids != frame["question_id"].tolist():
        raise AssertionError(
            "question_id did not round-trip through length-sorted batching; "
            "activations would be paired with the wrong labels"
        )

    labelled = label_frame(frame, completions, config)

    prefill_per_response = [
        t["prefill_seconds"] / t["batch_size"] for t in batch_timings
    ]
    generate_per_response = [
        t["generate_seconds"] / t["batch_size"] for t in batch_timings
    ]
    meta: dict[str, Any] = {
        "n_examples": n,
        "layers": list(layers),
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "sort_by_length": config.generation.sort_by_length,
        "total_seconds": total_seconds,
        "examples_per_second": n / total_seconds if total_seconds > 0 else None,
        "median_prefill_seconds_per_response": float(np.median(prefill_per_response)),
        "median_generate_seconds_per_response": float(np.median(generate_per_response)),
        "max_padded_length": float(max(t["padded_length"] for t in batch_timings)),
    }
    return acts, labelled, meta


# --------------------------------------------------------------------------- #
# Base rate gate
# --------------------------------------------------------------------------- #


def base_rate_summary(labelled: pd.DataFrame) -> dict[str, float]:
    """Summarise the label distribution under both matching rules.

    ``base_rate`` throughout this repo means the fraction of responses that are
    *incorrect* -- the positive class (DECISIONS.md 004). ``accuracy`` is its
    complement and is reported too, because SPEC.md §2's sanity band is stated
    in terms of the fraction correct.
    """
    n = len(labelled)
    correct = float(labelled["correct"].mean()) if n else 0.0

    # exact_match is NaN throughout when labeling.record_strict_em is off. The
    # strict figures are then None rather than 0.0: reporting "not computed" as
    # "nothing matched" would invent a 100-point labelling gap.
    strict_column = labelled["exact_match"]
    strict_recorded = bool(n) and not strict_column.isna().all()
    strict = float(strict_column.mean()) if strict_recorded else None

    return {
        "n": n,
        "accuracy_lenient": correct,
        "accuracy_strict_em": strict,
        "strict_em_recorded": strict_recorded,
        "base_rate_incorrect": 1.0 - correct,
        "base_rate_incorrect_strict_em": None if strict is None else 1.0 - strict,
        "lenient_minus_strict_accuracy": None if strict is None else correct - strict,
        "abstention_rate": float(labelled["abstained"].mean()) if n else 0.0,
    }


def assert_base_rate(labelled: pd.DataFrame, config: Config) -> dict[str, float]:
    """Stop the run if the label distribution is degenerate.

    SPEC.md §2 puts the expected accuracy for a 7B on TriviaQA no-context at
    roughly 0.45-0.70 and the abort band at 0.25-0.85. Outside it, the likely
    causes are a malformed prompt, truncated generation, or a broken matching
    rule -- all of which produce a probe result that means nothing, so training
    on them would waste the GPU hour that just ran.

    Args:
        labelled: Frame from :func:`run_extraction`.
        config: Resolved experiment config.

    Returns:
        The base-rate summary.

    Raises:
        BaseRateError: if lenient accuracy falls outside the configured band.
    """
    summary = base_rate_summary(labelled)
    accuracy = summary["accuracy_lenient"]
    low, high = config.labeling.base_rate_min, config.labeling.base_rate_max
    strict = summary["accuracy_strict_em"]
    LOGGER.info(
        "labels: accuracy %.3f (lenient) / %s (strict EM); base rate incorrect "
        "%.3f; abstention %.3f",
        accuracy,
        "not recorded" if strict is None else f"{strict:.3f}",
        summary["base_rate_incorrect"],
        summary["abstention_rate"],
    )
    if not low <= accuracy <= high:
        raise BaseRateError(
            f"lenient accuracy {accuracy:.3f} is outside the sanity band "
            f"[{low}, {high}]. Do not train a probe on this: the likely causes "
            "are a malformed prompt, truncated generation, or a broken matching "
            "rule, and a probe fitted to a degenerate label distribution is "
            "meaningless (TASKS.md Stage 3 gate)."
        )
    return summary


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_activations(
    acts: dict[int, np.ndarray], question_ids: Sequence[str], path: str | Path
) -> Path:
    """Write activations to a compressed npz in fp16, keyed by layer.

    fp16 halves a ~150MB file at a precision cost far below the noise floor of
    a logistic regression on standardised features. ``question_id`` travels in
    the same file so a later stage can assert the rows still line up with
    ``labels.parquet`` rather than assuming it.

    Args:
        acts: Layer index -> ``(n, hidden)`` array.
        question_ids: One id per row, in the same order.
        path: Destination ``.npz`` path.

    Returns:
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        f"layer_{layer}": array.astype(np.float16) for layer, array in acts.items()
    }
    payload["question_id"] = np.asarray(list(question_ids), dtype=object)
    np.savez_compressed(out, **payload)
    size_mb = out.stat().st_size / (1024**2)
    LOGGER.info("wrote %s (%.1f MB, %d layers)", out, size_mb, len(acts))
    return out


def load_activations(
    path: str | Path, expected_question_ids: Optional[Sequence[str]] = None
) -> dict[int, np.ndarray]:
    """Read activations back as float32, optionally asserting row alignment.

    Args:
        path: The ``.npz`` written by :func:`save_activations`.
        expected_question_ids: If given, must match the stored ids exactly.

    Returns:
        Layer index -> ``(n, hidden)`` float32 array.

    Raises:
        AssertionError: if the stored ids do not match ``expected_question_ids``.
    """
    with np.load(path, allow_pickle=True) as data:
        acts = {
            int(key.split("_")[1]): data[key].astype(np.float32)
            for key in data.files
            if key.startswith("layer_")
        }
        stored_ids = [str(q) for q in data["question_id"]] if "question_id" in data.files else None
    if expected_question_ids is not None:
        if stored_ids is None:
            raise AssertionError(f"{path} has no question_id array to verify against")
        if stored_ids != [str(q) for q in expected_question_ids]:
            raise AssertionError(
                f"{path} question_ids do not match the labels frame; the "
                "activations and labels are not aligned"
            )
    return acts
