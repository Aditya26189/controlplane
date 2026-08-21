"""Shared pytest fixtures.

Puts the repository root on ``sys.path`` so ``import src...`` works regardless
of where pytest is invoked from, and provides the fixtures that more than one
test file needs.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config, load_config  # noqa: E402  (path set up above)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def config() -> Config:
    """The committed ``config.yaml``, loaded and validated.

    Session-scoped: the config is frozen, so no test can mutate it for another.
    """
    return load_config(REPO_ROOT / "config.yaml")


# --------------------------------------------------------------------------- #
# Offline model fixtures.
#
# The test suite must run from a clean checkout with no network and no model
# cache, so nothing here downloads anything. The tokenizer is built from a
# word-level vocabulary and the model is a randomly initialised Qwen2 with four
# tiny layers -- a real transformer with real attention masking, which is what
# the left-padding equivalence check actually needs to exercise.
# --------------------------------------------------------------------------- #

TINY_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|> {{ message['role'] }} {{ message['content'] }} <|im_end|> "
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
)


@pytest.fixture(scope="session")
def tiny_tokenizer():
    """A left-paddable tokenizer with a chat template, built without a download."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = [
        "<unk>", "<eos>", "<pad>", "<|im_start|>", "<|im_end|>",
        "system", "user", "assistant", "you", "are", "a", "helpful",
        "answer", "the", "question", "concisely", "who", "what", "wrote",
        "is", "capital", "of", "france", "iliad", "paris", "homer", ".", "?",
    ]
    vocab = {word: i for i, word in enumerate(words)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    tokenizer.chat_template = TINY_CHAT_TEMPLATE
    return tokenizer


@pytest.fixture(scope="session")
def tiny_model():
    """A randomly initialised 4-layer Qwen2, on CPU, in float32.

    float32 rather than a half precision on purpose: the equivalence check
    compares batched against unbatched activations, and we want it to fail for
    real reasons, not for accumulated fp16 rounding.
    """
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        use_cache=False,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model
