"""Left padding is forced and asserted (CLAUDE.md invariant 4).

The failure this guards against is silent: right padding puts a pad token at
position -1, so every "last prompt token" activation is read off padding, the
pipeline completes normally, and the result looks like a negative finding.
"""

import pytest

from src.model import (
    PaddingSideError,
    assert_left_padding,
    build_prompt,
    build_quantization_config,
    configure_tokenizer,
    resolve_dtype,
)


class FakeTokenizer:
    """Minimal stand-in exposing the attributes the guard touches."""

    def __init__(self, padding_side="right", pad_token=None, eos_token="<eos>"):
        self.padding_side = padding_side
        self.pad_token = pad_token
        self.eos_token = eos_token


def test_assert_left_padding_accepts_left():
    assert_left_padding(FakeTokenizer(padding_side="left"))


@pytest.mark.parametrize("side", ["right", None, "both", ""])
def test_assert_left_padding_rejects_everything_else(side):
    """Anything other than 'left' must raise, including a missing attribute."""
    with pytest.raises(PaddingSideError, match="invariant 4"):
        assert_left_padding(FakeTokenizer(padding_side=side))


def test_assert_left_padding_rejects_object_without_attribute():
    with pytest.raises(PaddingSideError):
        assert_left_padding(object())


def test_configure_forces_left_padding_on_a_right_padded_tokenizer():
    """The loader corrects the padding side rather than trusting the default."""
    tokenizer = FakeTokenizer(padding_side="right")
    configure_tokenizer(tokenizer)
    assert tokenizer.padding_side == "left"


def test_configure_sets_pad_token_from_eos_when_missing():
    """Qwen2.5 ships no distinct pad token, and batching needs one."""
    tokenizer = FakeTokenizer(padding_side="right", pad_token=None, eos_token="<eos>")
    configure_tokenizer(tokenizer)
    assert tokenizer.pad_token == "<eos>"


def test_configure_leaves_an_existing_pad_token_alone():
    tokenizer = FakeTokenizer(padding_side="right", pad_token="<pad>")
    configure_tokenizer(tokenizer)
    assert tokenizer.pad_token == "<pad>"


def test_real_tokenizer_is_left_padded_after_configure(tiny_tokenizer):
    """The same guard, exercised on an actual transformers tokenizer."""
    tiny_tokenizer.padding_side = "right"
    configure_tokenizer(tiny_tokenizer)
    assert_left_padding(tiny_tokenizer)

    batch = tiny_tokenizer(["who wrote the iliad ?", "what is the capital"], padding=True)
    lengths = [sum(mask) for mask in batch["attention_mask"]]
    assert lengths[0] != lengths[1], "fixture must produce ragged lengths to be a test"
    # Left padding means the final position is real content for every row.
    assert all(mask[-1] == 1 for mask in batch["attention_mask"])


def test_prompt_ends_with_the_assistant_header(tiny_tokenizer, config):
    """add_generation_prompt is what makes the probed position the right one.

    Without the assistant turn header the last prompt token is the end of the
    user's question, not the token the model would generate from (SPEC.md §3).
    """
    configure_tokenizer(tiny_tokenizer)
    prompt = build_prompt(tiny_tokenizer, "who wrote the iliad ?", config)
    assert prompt.rstrip().endswith("assistant")
    assert config.prompt.system in prompt
    assert "who wrote the iliad ?" in prompt


def test_nf4_without_cuda_raises_rather_than_falling_back(config, monkeypatch):
    """A silent CPU fallback would change the condition without changing the hash."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="quantization: none"):
        build_quantization_config(config)


def test_quantization_none_returns_no_config(config):
    from dataclasses import replace

    cpu_config = replace(config, model=replace(config.model, quantization="none"))
    assert build_quantization_config(cpu_config) is None


def test_resolve_dtype_maps_config_names():
    import torch

    assert resolve_dtype("bfloat16") is torch.bfloat16
    assert resolve_dtype("float16") is torch.float16
    assert resolve_dtype("float32") is torch.float32
    with pytest.raises(ValueError):
        resolve_dtype("int8")
