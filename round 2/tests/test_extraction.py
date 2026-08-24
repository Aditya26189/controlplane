"""Extraction logic that can be tested without a GPU.

The parts needing a model are exercised on the GPU session by the notebook's
self-check. The parts here are the ones that fail *silently* — alias matching,
question-level splitting, padding assertions, and the transfer path — so they
are tested on CPU where they can be run on every commit.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.config import Config
from src.detectors.probe import LinearProbe
from src.extract.model import PaddingError, assert_left_padding
from src.extract.triviaqa import (
    SHORT_ALIAS_CHARS,
    TriviaItem,
    is_correct,
    normalise_answer,
    split_questions,
)
from src.model import WarrantStatus
from src.validation.evalsets import TEST, TRAIN, split_by_question
from src.validation.runner import validate, validate_transferred
from src.validation.synthetic import synthetic_cache, synthetic_evalset


# --------------------------------------------------------------------------- #
# Alias matching — the guard that keeps the base rate honest
# --------------------------------------------------------------------------- #


def test_short_aliases_require_an_exact_token_match() -> None:
    """``"US"`` appears inside "thus", "focus" and "must".

    Substring containment on a two-character alias marks unrelated generations
    correct, which lowers the measured base rate and inflates every downstream
    number without raising anything.
    """
    assert not is_correct("The answer is thus clear", ["US"])[0]
    assert not is_correct("I focus on music", ["US"])[0]
    assert not is_correct("you must go", ["US"])[0]
    # But a genuine mention still matches.
    assert is_correct("US", ["US"])[0]
    assert is_correct("It was the US", ["US"])[0]


def test_long_aliases_match_by_substring() -> None:
    correct, how = is_correct("I think the United States of America", ["United States"])
    assert correct
    assert "substring" in how


def test_the_guard_boundary_is_where_it_is_documented() -> None:
    """Three characters, so the rule and the constant cannot drift apart."""
    assert SHORT_ALIAS_CHARS == 3
    # Two characters: token match only.
    assert not is_correct("abcd", ["ab"])[0]
    assert is_correct("ab cd", ["ab"])[0]
    # Three: substring is allowed.
    assert is_correct("abcd", ["abc"])[0]


def test_empty_generation_is_incorrect_and_says_so() -> None:
    correct, how = is_correct("", ["Paris"])
    assert not correct
    assert "empty" in how


def test_normalisation_strips_articles_punctuation_and_case() -> None:
    assert normalise_answer("The Iliad!") == normalise_answer("iliad")
    assert normalise_answer("  A  Tale,  of Two Cities ") == "tale of two cities"


# --------------------------------------------------------------------------- #
# Splitting by question
# --------------------------------------------------------------------------- #


def _items(n: int) -> list[TriviaItem]:
    return [TriviaItem(f"q{i:04d}", f"question {i}?", [f"answer {i}"]) for i in range(n)]


def test_split_questions_never_overlaps() -> None:
    splits = split_questions(_items(400), fractions=(0.5, 0.25, 0.25), seed=1729)
    all_indices = [i for indices in splits.values() for i in indices]
    assert len(set(all_indices)) == len(all_indices) == 400
    assert len(splits["train"]) == 200
    assert len(splits["validation"]) == 100
    assert len(splits["test"]) == 100


def test_split_questions_is_reproducible() -> None:
    first = split_questions(_items(200), fractions=(0.5, 0.25, 0.25), seed=7)
    second = split_questions(_items(200), fractions=(0.5, 0.25, 0.25), seed=7)
    assert first == second


def test_split_questions_refuses_an_empty_split() -> None:
    with pytest.raises(RuntimeError, match="is empty"):
        split_questions(_items(10), fractions=(0.98, 0.01, 0.01), seed=1)


def test_split_fractions_must_sum_to_one() -> None:
    with pytest.raises(RuntimeError, match="must sum to 1"):
        split_questions(_items(100), fractions=(0.5, 0.25, 0.1), seed=1)


def test_2400_questions_yields_the_600_item_test_split() -> None:
    """``DECISIONS.md`` 030: an eval set is sized by its test split."""
    splits = split_questions(_items(2400), fractions=(0.5, 0.25, 0.25), seed=1729)
    assert len(splits["test"]) == 600
    assert len(splits["train"]) == 1200


# --------------------------------------------------------------------------- #
# Padding
# --------------------------------------------------------------------------- #


def test_assert_left_padding_refuses_right_padding() -> None:
    """The check that costs a string comparison and saves a day."""

    class Tokenizer:
        padding_side = "right"

    with pytest.raises(PaddingError, match="not 'left'"):
        assert_left_padding(Tokenizer(), where="test")


def test_assert_left_padding_refuses_an_absent_attribute() -> None:
    """Absence reads as refusal, not as permission (``DECISIONS.md`` 050)."""

    class Tokenizer:
        pass

    with pytest.raises(PaddingError, match="not 'left'"):
        assert_left_padding(Tokenizer(), where="test")


def test_assert_left_padding_accepts_left() -> None:
    class Tokenizer:
        padding_side = "left"

    assert_left_padding(Tokenizer(), where="test")  # does not raise


# --------------------------------------------------------------------------- #
# The attention backend context manager
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "exception",
    [RuntimeError, MemoryError, ValueError, KeyboardInterrupt],
)
def test_efficient_attention_propagates_body_exceptions(exception) -> None:
    """A failure inside the block must escape it unchanged.

    The first version wrapped ``yield`` in ``except (ImportError, AttributeError,
    RuntimeError)`` so it could fall back across torch versions. But
    ``torch.cuda.OutOfMemoryError`` subclasses ``RuntimeError``, so an OOM in the
    body was caught by the fallback handler, which then yielded a second time:

        RuntimeError: generator didn't stop after throw()

    That replaced the real error with a confusing one **and** defeated the retry
    logic watching for ``OutOfMemoryError``. ``RuntimeError`` is in the list here
    because it is the one that actually bit.
    """
    from src.extract.activations import efficient_attention

    with pytest.raises(exception, match="boom"):
        with efficient_attention():
            raise exception("boom")


def test_efficient_attention_exits_cleanly() -> None:
    """And the ordinary path still works, exactly once."""
    from src.extract.activations import efficient_attention

    entered = 0
    with efficient_attention():
        entered += 1
    assert entered == 1


def test_efficient_attention_yields_once_even_without_torch_support(
    monkeypatch,
) -> None:
    """On a torch with neither API, it degrades to a no-op rather than failing.

    A context manager that raises on an old torch would break extraction for a
    reason unrelated to memory. It warns and continues, paying the math
    backend's cost.
    """
    import torch

    from src.extract import activations

    monkeypatch.delattr(torch.backends.cuda, "sdp_kernel", raising=False)
    monkeypatch.setitem(
        __import__("sys").modules, "torch.nn.attention", None
    )
    entered = 0
    with activations.efficient_attention():
        entered += 1
    assert entered == 1


# --------------------------------------------------------------------------- #
# The forward pass must not build logits it throws away
# --------------------------------------------------------------------------- #


def _fake_causal_lm():
    """A three-block causal LM that counts vocabulary projections."""
    import torch

    class Block(torch.nn.Module):
        def forward(self, hidden, **kwargs):
            return (hidden + 1.0,)

    class Trunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block() for _ in range(3)])

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            hidden = input_ids.unsqueeze(-1).float().repeat(1, 1, 4)
            for block in self.layers:
                hidden = block(hidden)[0]
            return hidden

    class CausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Trunk()
            self.lm_head_calls = 0

        @property
        def device(self):
            return torch.device("cpu")

        def forward(self, **kwargs):
            hidden = self.model(**kwargs)
            # Stands in for lm_head: seq x vocab, the allocation being avoided.
            self.lm_head_calls += 1
            return hidden

    return CausalLM()


class _FakeBatch(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    padding_side = "left"

    def __call__(self, prompts, return_tensors=None, padding=None, truncation=None):
        import torch

        if isinstance(prompts, str):
            prompts = [prompts]
        width = max(len(p) for p in prompts)
        ids, mask = [], []
        for prompt in prompts:
            pad = width - len(prompt)
            ids.append([0] * pad + [ord(c) % 97 + 1 for c in prompt])
            mask.append([0] * pad + [1] * len(prompt))
        return _FakeBatch(
            input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask)
        )


def test_forward_runs_the_trunk_not_the_causal_lm() -> None:
    """The vocabulary projection must never run during extraction.

    ``*ForCausalLM`` applies ``lm_head`` at **every position** and transformers
    upcasts the result to float32 with both copies alive. At Qwen2.5-7B's
    152,064-token vocabulary that is ``seq x 152064 x 6`` bytes — 9.43 GiB at an
    11k-token sequence, larger than even the math backend's attention matrix at
    the same length (6.43 GiB). Every byte is discarded: the probe reads one
    hidden state from one layer.

    Two GPU sessions were lost to this. The first diagnosis computed the
    attention term, matched it against the reported OOM size, and stopped —
    the two terms are the same order of magnitude, so a plausible match is not
    evidence that the other term is small.
    """
    from src.extract.activations import _hidden_states_at
    from src.extract.model import LoadedModel

    model = _fake_causal_lm()
    loaded = LoadedModel(
        model=model,
        tokenizer=_FakeTokenizer(),
        name="fake",
        num_hidden_layers=3,
        hidden_size=4,
        quantization="none",
        device="cpu",
        dtype="torch.float32",
    )
    hidden, mask = _hidden_states_at(loaded, ["abc", "abcdef"], 2, padding="left")

    assert model.lm_head_calls == 0, (
        "the causal-LM wrapper ran, so vocabulary-sized logits were built and "
        "discarded -- this is what exhausts the card at long context"
    )
    assert hidden.shape == (2, 6, 4)
    assert mask.tolist() == [[0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1]]


def test_base_model_falls_back_rather_than_guessing() -> None:
    """An unrecognised architecture costs memory; it must not be silently wrong.

    Returning the wrapper is safe because nothing reads the return value — the
    activation arrives through a forward hook.
    """
    from src.extract.activations import _base_model

    class Opaque:
        def forward(self):
            pass

    opaque = Opaque()
    assert _base_model(opaque) is opaque


def test_base_model_finds_the_trunk_on_a_causal_lm() -> None:
    from src.extract.activations import _base_model

    model = _fake_causal_lm()
    assert _base_model(model) is model.model


# --------------------------------------------------------------------------- #
# What the real transformers code path actually does
#
# These run a genuinely small Qwen2 rather than a fake, because the three GPU
# sessions lost on this stage were all lost to reasoning about the code path
# instead of executing it.
# --------------------------------------------------------------------------- #


def _tiny_qwen(vocab: int = 512):
    transformers = pytest.importorskip("transformers")
    import torch

    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

    config = Qwen2Config(
        vocab_size=vocab, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    model = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    return model.eval()


def test_all_ones_mask_is_already_skipped(monkeypatch) -> None:
    """Transformers hands SDPA ``is_causal=True`` when nothing is padded.

    This is the measurement that showed a previous "fix" to be a no-op. The
    reasoning behind it was that passing an attention mask forces a 4D float
    mask, which SDPA's memory-efficient kernel declines, falling back to the
    math backend and its ``seq^2`` allocation. The first half is true of a
    *padded* batch. It is not true of an all-ones mask: ``masking_utils.
    _ignore_causal_mask_sdpa`` returns True when ``padding_mask.all()``, and
    SDPA receives no mask at all.

    A GPU session was spent on that no-op, and it was written up as the fix.
    The test exists so the claim is checked rather than argued.
    """
    import torch
    import torch.nn.functional as functional

    model = _tiny_qwen()
    assert model.config._attn_implementation == "sdpa"

    calls = []
    real = functional.scaled_dot_product_attention

    def spy(query, key, value, attn_mask=None, is_causal=False, **kwargs):
        calls.append((attn_mask is None, is_causal))
        return real(query, key, value, attn_mask=attn_mask, is_causal=is_causal, **kwargs)

    import transformers.integrations.sdpa_attention as sdpa_attention

    monkeypatch.setattr(functional, "scaled_dot_product_attention", spy)
    monkeypatch.setattr(
        sdpa_attention.torch.nn.functional, "scaled_dot_product_attention", spy
    )

    ids = torch.zeros(1, 32, dtype=torch.long)
    with torch.no_grad():
        model(input_ids=ids, attention_mask=torch.ones(1, 32, dtype=torch.long),
              use_cache=False)
    with_mask = calls[0]
    calls.clear()
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    without_mask = calls[0]

    assert with_mask == (True, True), (
        "an all-ones mask reached SDPA; the version-independence assumed by "
        "_hidden_states_at no longer holds"
    )
    assert with_mask == without_mask, "dropping an all-ones mask changed the call"

    # And the mask must still survive when it carries information, or padded
    # batches would silently attend to pad tokens.
    calls.clear()
    padded = torch.cat([torch.zeros(1, 32), torch.ones(1, 32)]).long()
    with torch.no_grad():
        model(input_ids=torch.zeros(2, 32, dtype=torch.long),
              attention_mask=padded, use_cache=False)
    assert calls[0] == (False, False), "a real padding mask was discarded"


def test_trunk_allocates_far_less_than_the_causal_lm() -> None:
    """Counted bytes, not estimated ones.

    The causal LM's largest tensor is ``(seq, vocab)``; the trunk's is
    ``(seq, intermediate)``. At Qwen2.5-7B's real vocabulary of 152,064 against
    an intermediate size of 18,944 that is an 8x ratio on the largest single
    allocation, and the logits are then upcast to float32 on top.
    """
    pytest.importorskip("transformers")
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    from src.extract.activations import _base_model

    class Counter(TorchDispatchMode):
        def __init__(self) -> None:
            self.total = 0
            self.largest = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for item in out if isinstance(out, (tuple, list)) else [out]:
                if isinstance(item, torch.Tensor) and item.numel() > 1:
                    size = item.numel() * item.element_size()
                    self.total += size
                    self.largest = max(self.largest, size)
            return out

    model = _tiny_qwen(vocab=15206)
    ids = torch.zeros(1, 1024, dtype=torch.long)
    measured = {}
    for label, target in (("full", model), ("trunk", _base_model(model))):
        counter = Counter()
        with torch.no_grad(), counter:
            target(input_ids=ids, use_cache=False)
        measured[label] = (counter.total, counter.largest)

    assert measured["trunk"][0] < measured["full"][0] / 3, (
        "the trunk should allocate far less; got %s" % (measured,)
    )
    assert measured["trunk"][1] < measured["full"][1] / 8, (
        "the largest tensor should no longer be vocabulary-sized; got %s" % (measured,)
    )


def test_no_seq_squared_tensor_is_materialised() -> None:
    """The attention score matrix must never appear as a real tensor.

    On the eager path it does: ``heads x seq x seq``, with the softmax upcast to
    float32, which is 28.7 GiB for one op at 16k tokens on this model. That is
    why :func:`load_model` asks for sdpa explicitly and refuses anything else.
    """
    pytest.importorskip("transformers")
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    seq = 256

    class Shapes(TorchDispatchMode):
        def __init__(self) -> None:
            self.square = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for item in out if isinstance(out, (tuple, list)) else [out]:
                if isinstance(item, torch.Tensor) and item.dim() >= 2:
                    if item.shape[-1] == seq and item.shape[-2] == seq:
                        self.square.append(tuple(item.shape))
            return out

    model = _tiny_qwen()
    shapes = Shapes()
    with torch.no_grad(), shapes:
        model(input_ids=torch.zeros(1, seq, dtype=torch.long), use_cache=False)
    assert not shapes.square, (
        "a seq x seq tensor was materialised: %s. Attention has fallen off the "
        "memory-efficient path and long context will not fit." % (shapes.square,)
    )


# --------------------------------------------------------------------------- #
# Transfer: what THIS probe is worth on new traffic
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def transfer_setup(config: Config):
    """A short-context source run and a long-context envelope to transfer onto."""
    short = synthetic_evalset(
        eval_set_id="triviaqa-600-synthetic", n_items=800, base_rate=0.152,
        seed=config.seed, items_per_question=2, declare_splits=True,
    )
    short_cache = synthetic_cache(
        short, seed=config.seed,
        window=config.probe.rolling_window, stride=config.probe.rolling_stride,
    )
    variant = "T1-max_rolling_means"
    source = validate(
        config, short, short_cache, variant=variant,
        detector_id=f"probe-{variant}", detector_version="0.1.0+fixture",
        target_flag_rate=0.05,
    )
    splits = split_by_question(short, seed=config.seed)
    probe = LinearProbe(
        source.probe_fit.C,
        class_weight=config.probe.class_weight,
        standardize=config.probe.standardize,
        seed=config.seed,
    ).fit(short_cache.matrix(variant), short_cache.labels, splits[TRAIN])

    long_items = tuple(
        dataclasses.replace(item, split=TEST)
        for item in short.items
        if item.split == TEST
    )
    long = dataclasses.replace(
        short, eval_set_id="triviaqa-longctx-600-synthetic", items=long_items,
        construction={**short.construction, "long_context": True},
    )
    long_cache = synthetic_cache(
        long, seed=config.seed,
        window=config.probe.rolling_window, stride=config.probe.rolling_stride,
    )
    return config, source, probe, long, long_cache, variant


def test_transfer_refits_nothing(transfer_setup) -> None:
    """The drift question is what *this* probe is worth, not what a new one would be.

    Refitting on the new envelope produces a better number and a weaker claim:
    nobody retrains between one request and the next, so a refitted number
    describes a system that does not exist.
    """
    config, source, probe, long, long_cache, variant = transfer_setup
    run = validate_transferred(
        config, long, long_cache, source=source, probe=probe, variant=variant
    )
    # Threshold and regularisation both come from the source run.
    assert run.operating_point == source.operating_point
    assert run.probe_fit.C == source.probe_fit.C
    assert run.probe_fit.selected_on == "validation"
    # Nothing was selected on the new envelope: the whole set is test.
    assert run.splits == {"train": 0, "validation": 0, "test": len(long)}


def test_transfer_keys_the_warrant_to_the_new_envelope(transfer_setup) -> None:
    """Same detector, new envelope, new cell. Invariant 1's whole point."""
    config, source, probe, long, long_cache, variant = transfer_setup
    run = validate_transferred(
        config, long, long_cache, source=source, probe=probe, variant=variant
    )
    assert run.warrant.detector_id == source.warrant.detector_id
    assert run.warrant.eval_set_id == long.eval_set_id
    assert run.warrant.key != source.warrant.key
    assert run.warrant.warrant_id != source.warrant.warrant_id
    assert run.warrant.envelope.envelope_id == long.envelope_id


def test_transfer_marks_carried_controls_as_carried(transfer_setup) -> None:
    """A control describing the probe travels; one describing the envelope does not.

    The carried controls say so in their detail, so a reader of the warrant can
    see that the padding evidence and the negative controls were established on
    the source extraction rather than re-run here.
    """
    config, source, probe, long, long_cache, variant = transfer_setup
    run = validate_transferred(
        config, long, long_cache, source=source, probe=probe, variant=variant
    )
    assert len(run.controls) == len(source.controls)
    for control in run.controls:
        assert "carried from the source validation" in control.detail
        assert source.eval_set_id in control.detail


def test_transfer_refuses_a_stale_cache(transfer_setup) -> None:
    config, source, probe, long, long_cache, variant = transfer_setup
    edited = dataclasses.replace(
        long, construction={**long.construction, "note": "edited after extraction"}
    )
    with pytest.raises(ValueError, match="re-extract"):
        validate_transferred(
            config, edited, long_cache, source=source, probe=probe, variant=variant
        )


def test_transfer_produces_a_usable_warrant_or_an_honest_refusal(
    transfer_setup,
) -> None:
    """Either outcome is fine; a crash or a silent pass is not."""
    config, source, probe, long, long_cache, variant = transfer_setup
    run = validate_transferred(
        config, long, long_cache, source=source, probe=probe, variant=variant
    )
    assert run.warrant.status in (WarrantStatus.VALID, WarrantStatus.REFUSED)
    if run.warrant.status is WarrantStatus.REFUSED:
        assert run.warrant.status_reason
    assert run.test_scored == 1
    assert run.metrics.recall is not None
