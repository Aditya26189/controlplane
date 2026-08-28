"""The real loading path, exercised offline against a saved tiny model.

Stage 3 costs a GPU hour. These tests run ``load_model_and_tokenizer``,
``describe_model`` and ``sanity_generate`` end to end on a 41k-parameter Qwen2
saved to a temp directory, so a trivial breakage in the loading path -- a
renamed keyword, a missing pad token, a template that drops the assistant
header -- surfaces on a laptop in seconds instead of on a T4 an hour in.
"""

from dataclasses import replace

import pytest

from src.model import (
    assert_left_padding,
    describe_model,
    load_model_and_tokenizer,
    resolve_layers,
    sanity_generate,
)


@pytest.fixture(scope="module")
def saved_model_dir(tmp_path_factory, tiny_model, tiny_tokenizer):
    """Persist the tiny model and tokenizer so from_pretrained has a real target."""
    path = tmp_path_factory.mktemp("tiny-qwen2")
    tiny_model.save_pretrained(path)
    tiny_tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def local_config(saved_model_dir):
    """Config pointing at the saved tiny model, unquantised for CPU."""
    from src.config import load_config

    base = load_config("config.yaml")
    return replace(
        base,
        model=replace(
            base.model,
            name=str(saved_model_dir),
            quantization="none",
            dtype="float32",
            device_map="cpu",
        ),
    )


def test_loader_returns_a_left_padded_tokenizer(local_config):
    """The invariant holds through the real loader, not just the helper."""
    model, tokenizer = load_model_and_tokenizer(local_config)
    assert_left_padding(tokenizer)
    assert tokenizer.pad_token is not None
    assert model.training is False, "model must be in eval mode: dropout would make greedy decoding non-deterministic"


def test_describe_model_reports_layers_and_template(local_config):
    """describe_model is what the Stage 2 gate prints; it must be complete."""
    model, tokenizer = load_model_and_tokenizer(local_config)
    described = describe_model(model, tokenizer, local_config)

    assert described["num_hidden_layers"] == 4
    assert described["hidden_size"] == 32
    assert described["padding_side"] == "left"
    assert all(1 <= layer <= 4 for layer in described["probe_layers"])
    # SPEC.md §3: the exact template must land in the artifacts.
    assert "<QUESTION>" in described["example_prompt"]
    assert described["system_prompt"] == local_config.prompt.system


def test_resolved_layers_are_valid_hidden_state_indices(local_config):
    """Index 0 is the embedding output, so a probe layer of 0 would be wrong."""
    model, _ = load_model_and_tokenizer(local_config)
    layers = resolve_layers(model, local_config)
    assert min(layers) >= 1
    assert max(layers) <= model.config.num_hidden_layers


def test_sanity_generate_returns_a_string(local_config):
    """A random-weight model says nonsense, but it must say it without crashing."""
    model, tokenizer = load_model_and_tokenizer(local_config)
    out = sanity_generate(model, tokenizer, "who wrote the iliad ?", local_config)
    assert isinstance(out, str)
