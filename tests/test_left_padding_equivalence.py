"""Batched last-token activations must equal unbatched ones (invariant 4).

Runs on the tiny offline Qwen2, so it needs no GPU and no download.

The criteria are scale-invariant — relative L2 error and cosine similarity —
because an absolute tolerance cannot tell "the batch is read at the wrong
position" apart from "the two forward passes rounded differently". On a 7B in
bfloat16 the latter reaches an absolute deviation of 3.0 in the late layers
purely from arithmetic, which an absolute 1e-2 tolerance rejects as a fault
(DECISIONS.md 014).

So these tests pin both directions and, crucially, that the criteria still
*discriminate*: right padding must be rejected by a wide margin.
"""

import numpy as np
import pytest

from src.extract import (
    EquivalenceCheckError,
    check_left_padding_equivalence,
    compare_batched_unbatched,
    equivalence_passes,
    last_token_activations,
)
from src.model import PaddingSideError, configure_tokenizer

RAGGED_PROMPTS = [
    "who wrote the iliad ?",
    "what is the capital of france ?",
    "what is paris",
    "who wrote the iliad of homer in the capital of france ?",
]

LAYERS = (1, 2, 3, 4)


@pytest.fixture
def left_padded(tiny_tokenizer):
    """The tokenizer, explicitly left-padded for this test."""
    configure_tokenizer(tiny_tokenizer)
    return tiny_tokenizer


def test_batched_matches_unbatched_under_left_padding(tiny_model, left_padded, config):
    result = check_left_padding_equivalence(
        tiny_model,
        left_padded,
        RAGGED_PROMPTS,
        LAYERS,
        relative_tolerance=config.equivalence_check.relative_tolerance,
        min_cosine=config.equivalence_check.min_cosine,
    )
    assert result["max_relative_l2"] < config.equivalence_check.relative_tolerance
    assert result["min_cosine_observed"] > config.equivalence_check.min_cosine
    assert result["distinct_prompt_lengths"] > 1


def test_right_padding_is_rejected_by_a_wide_margin(tiny_model, tiny_tokenizer, config):
    """The criteria must catch the failure they exist for.

    A relaxed-looking threshold is only defensible if the fault still fails it,
    so this measures the actual separation rather than asserting one.
    """
    configure_tokenizer(tiny_tokenizer)
    left = compare_batched_unbatched(tiny_model, tiny_tokenizer, RAGGED_PROMPTS, LAYERS)

    tiny_tokenizer.padding_side = "right"
    right = compare_batched_unbatched(tiny_model, tiny_tokenizer, RAGGED_PROMPTS, LAYERS)

    tol = config.equivalence_check.relative_tolerance
    cos = config.equivalence_check.min_cosine
    assert equivalence_passes(left, tol, cos)
    assert not equivalence_passes(right, tol, cos)
    assert right["max_relative_l2"] > 100 * left["max_relative_l2"]


def test_positive_control_runs_and_restores_the_padding_side(
    tiny_model, left_padded, config
):
    """The control must leave the tokenizer left-padded for the real run."""
    result = check_left_padding_equivalence(
        tiny_model,
        left_padded,
        RAGGED_PROMPTS,
        LAYERS,
        relative_tolerance=config.equivalence_check.relative_tolerance,
        min_cosine=config.equivalence_check.min_cosine,
        positive_control=True,
    )
    assert result["positive_control_ran"] is True
    assert result["positive_control_rejected"] is True
    assert result["right_padding_control"]["padding_side"] == "right"
    assert left_padded.padding_side == "left", "control must restore left padding"


def test_positive_control_raises_when_criteria_cannot_discriminate(
    tiny_model, left_padded
):
    """Thresholds so loose that right padding passes are worthless.

    This is the guard against "fix" the failing check by relaxing it: if the
    relaxation also accepts the fault, the run stops.
    """
    with pytest.raises(EquivalenceCheckError, match="positive control PASSED"):
        check_left_padding_equivalence(
            tiny_model,
            left_padded,
            RAGGED_PROMPTS,
            LAYERS,
            relative_tolerance=5.0,  # loose enough to accept the fault
            min_cosine=0.0,
            positive_control=True,
        )


def test_checker_refuses_a_right_padded_tokenizer(tiny_model, tiny_tokenizer):
    """The guard fires before any numerical comparison runs."""
    tiny_tokenizer.padding_side = "right"
    with pytest.raises(PaddingSideError):
        check_left_padding_equivalence(tiny_model, tiny_tokenizer, RAGGED_PROMPTS, LAYERS)


def test_checker_rejects_equal_length_prompts(tiny_model, left_padded):
    """Equal lengths mean no padding, so the comparison would prove nothing."""
    same = ["what is paris", "what is homer"]
    with pytest.raises(EquivalenceCheckError, match="differing token lengths"):
        check_left_padding_equivalence(tiny_model, left_padded, same, LAYERS)


def test_checker_raises_above_tolerance(tiny_model, left_padded):
    """An impossibly tight tolerance must raise rather than warn."""
    with pytest.raises(EquivalenceCheckError, match="invariant 4"):
        check_left_padding_equivalence(
            tiny_model,
            left_padded,
            RAGGED_PROMPTS,
            LAYERS,
            relative_tolerance=1e-12,
            min_cosine=0.999,
            positive_control=False,
        )


def test_criteria_are_scale_invariant():
    """The whole point: magnitude must not move the verdict.

    Scaling both sides by 1000 leaves relative error and cosine untouched, while
    an absolute deviation would grow by 1000x — which is exactly how a healthy
    7B run got rejected.
    """
    small = {
        "max_relative_l2": 1e-3,
        "min_cosine": 0.9999999,
        "max_abs": 1e-3,
    }
    large = {
        "max_relative_l2": 1e-3,
        "min_cosine": 0.9999999,
        "max_abs": 1.0,  # same relative error, 1000x the magnitude
    }
    assert equivalence_passes(small, 0.02, 0.999)
    assert equivalence_passes(large, 0.02, 0.999)


def test_report_carries_per_layer_detail(tiny_model, left_padded):
    """A reviewer should be able to see the curve, not just the maximum."""
    report = compare_batched_unbatched(tiny_model, left_padded, RAGGED_PROMPTS, LAYERS)
    assert set(report["per_layer"]) == {str(layer) for layer in LAYERS}
    for stats in report["per_layer"].values():
        assert {"max_abs", "max_relative_l2", "min_cosine", "reference_norm_median"} <= set(stats)
    assert report["padding_side"] == "left"


# --- shape and finiteness assertions --------------------------------------- #


def test_last_token_activations_shapes(tiny_model, left_padded):
    import torch

    enc = left_padded(RAGGED_PROMPTS, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = tiny_model(**enc, output_hidden_states=True, use_cache=False)
    acts = last_token_activations(out, LAYERS, tiny_model.config.num_hidden_layers)
    for layer in LAYERS:
        assert acts[layer].shape == (len(RAGGED_PROMPTS), tiny_model.config.hidden_size)
        assert np.isfinite(acts[layer]).all()


def test_last_token_activations_rejects_out_of_range_layer(tiny_model, left_padded):
    """Index 0 is the embedding output, not a transformer block."""
    import torch

    enc = left_padded(RAGGED_PROMPTS, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = tiny_model(**enc, output_hidden_states=True, use_cache=False)
    n_layers = tiny_model.config.num_hidden_layers
    with pytest.raises(AssertionError, match="out of range"):
        last_token_activations(out, [0], n_layers)
    with pytest.raises(AssertionError, match="out of range"):
        last_token_activations(out, [n_layers + 1], n_layers)


def test_last_token_activations_rejects_wrong_hidden_state_count(tiny_model, left_padded):
    """A mismatch here means the layer indices no longer mean what we think."""
    import torch

    enc = left_padded(RAGGED_PROMPTS, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = tiny_model(**enc, output_hidden_states=True, use_cache=False)
    with pytest.raises(AssertionError, match="hidden-state tensors"):
        last_token_activations(out, LAYERS, tiny_model.config.num_hidden_layers + 3)
