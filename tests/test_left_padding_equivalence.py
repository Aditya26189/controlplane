"""Batched last-token activations must equal unbatched ones (invariant 4).

Runs on the tiny offline Qwen2, so it needs no GPU and no download. Two
directions matter equally:

* under left padding the deviation is at the numerical floor, and
* under right padding it is enormous --

because a check that has never been seen to fail is not evidence of anything.
"""

import numpy as np
import pytest

from src.extract import (
    EquivalenceCheckError,
    batched_unbatched_deviation,
    check_left_padding_equivalence,
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


def test_batched_matches_unbatched_under_left_padding(tiny_model, left_padded):
    """The production tolerance is 1e-2; in fp32 the real deviation is far below."""
    result = check_left_padding_equivalence(
        tiny_model, left_padded, RAGGED_PROMPTS, LAYERS, tolerance=1e-2
    )
    assert result["max_deviation"] < 1e-2
    assert result["distinct_prompt_lengths"] > 1
    assert set(result["per_layer_deviation"]) == {str(layer) for layer in LAYERS}


def test_right_padding_produces_a_large_deviation(tiny_model, tiny_tokenizer):
    """The check must actually catch the failure it exists for.

    With right padding, position -1 of a short sequence is a pad token, so the
    batched read is of padding rather than of the final prompt token. Nothing
    raises anywhere in transformers; only this comparison notices.
    """
    tiny_tokenizer.padding_side = "right"
    max_deviation, _ = batched_unbatched_deviation(
        tiny_model, tiny_tokenizer, RAGGED_PROMPTS, LAYERS
    )
    assert max_deviation > 1e-2, (
        "right padding must move the activations far enough for the 1e-2 "
        "tolerance to reject it"
    )


def test_checker_refuses_a_right_padded_tokenizer(tiny_model, tiny_tokenizer):
    """The guard fires before the numerical comparison even runs."""
    tiny_tokenizer.padding_side = "right"
    with pytest.raises(PaddingSideError):
        check_left_padding_equivalence(
            tiny_model, tiny_tokenizer, RAGGED_PROMPTS, LAYERS
        )


def test_checker_rejects_equal_length_prompts(tiny_model, left_padded):
    """Equal lengths mean no padding, so the comparison would prove nothing."""
    same = ["what is paris", "what is homer"]
    with pytest.raises(EquivalenceCheckError, match="differing token lengths"):
        check_left_padding_equivalence(tiny_model, left_padded, same, LAYERS)


def test_checker_raises_above_tolerance(tiny_model, left_padded):
    """An impossibly tight tolerance must raise rather than warn."""
    with pytest.raises(EquivalenceCheckError, match="invariant 4"):
        check_left_padding_equivalence(
            tiny_model, left_padded, RAGGED_PROMPTS, LAYERS, tolerance=1e-12
        )


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
