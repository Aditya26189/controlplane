"""Loading the model and tokenizer, with the padding assertion that has to hold.

**Left padding, asserted three times.** With right padding, position ``-1`` of a
batched sequence is a pad token, so every activation is read from nothing. The
probe still trains, still scores, and lands near 0.5 AUROC — which reads as
*"the idea doesn't work"* rather than as a bug, and costs a day. So the padding
side is asserted at load, again before every batched call, and a third time at
validation via the fault-injection control that compares batched against
unbatched scoring (``SPEC.md`` §2.1).

Everything here imports torch **lazily, inside functions**. The store, matrix,
sampling and policy layers must stay runnable on a laptop with no GPU stack, so
importing torch at module scope would make a CPU-only test suite depend on it.
"""

from __future__ import annotations

import dataclasses
import logging
import statistics
from typing import Any, Optional, Sequence

__all__ = [
    "LoadedModel",
    "assert_left_padding",
    "build_prompt",
    "load_model",
    "load_tokenizer",
    "token_length_summary",
]

_LOG = logging.getLogger(__name__)


class PaddingError(RuntimeError):
    """Raised when a tokenizer is not configured to pad on the left.

    Its own type because this is the failure mode that produces a *plausible*
    wrong answer rather than a crash, and a caller catching broad exceptions
    around model loading must not swallow it.
    """


@dataclasses.dataclass
class LoadedModel:
    """A model, its tokenizer, and the facts needed to record provenance.

    Args:
        model: The loaded causal LM, in eval mode.
        tokenizer: Its tokenizer, padding on the left.
        name: Model id, as recorded in the extraction cache and pinned into
            every activation-tier warrant (``SPEC.md`` §5.4).
        num_hidden_layers: Depth, used to resolve fractional layer depths.
        hidden_size: Residual stream width.
        quantization: What was actually applied, which may differ from what was
            requested if bitsandbytes is unavailable — recorded rather than
            assumed, because a warrant is pinned to the model that produced it.
        device: Where it ran.
        dtype: Compute dtype, as a string.
    """

    model: Any
    tokenizer: Any
    name: str
    num_hidden_layers: int
    hidden_size: int
    quantization: str
    device: str
    dtype: str

    def provenance(self) -> dict[str, Any]:
        """What goes into the cache beside the activations."""
        return {
            "model_name": self.name,
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "quantization": self.quantization,
            "device": self.device,
            "dtype": self.dtype,
        }


def assert_left_padding(tokenizer: Any, *, where: str) -> None:
    """Refuse to proceed unless the tokenizer pads on the left.

    Called at load and again immediately before every batched forward pass.
    Twice, because the failure is silent and the cost of the check is a string
    comparison — and because a tokenizer's ``padding_side`` is a mutable
    attribute that any library touching it can change between load and use.

    Args:
        tokenizer: The tokenizer to check.
        where: What is about to happen, named in the error so the trace points
            at the call site rather than at this function.

    Raises:
        PaddingError: If padding is not on the left.
    """
    side = getattr(tokenizer, "padding_side", None)
    if side != "left":
        raise PaddingError(
            f"{where}: tokenizer.padding_side is {side!r}, not 'left'. With right "
            "padding the last position of a batched sequence is a pad token, so "
            "every activation is read from nothing. This does not raise on its "
            "own — the probe trains, scores, and lands near 0.5 AUROC, which "
            "reads as 'the idea does not work'. Set padding_side='left'."
        )


def load_model(
    name: str,
    *,
    quantization: str = "nf4",
    device_map: str = "auto",
    trust_remote_code: bool = False,
    reserve_gib_per_gpu: float = 4.0,
) -> LoadedModel:
    """Load a causal LM for activation extraction.

    Args:
        name: HuggingFace model id, from ``config.model.name``.
        quantization: ``"nf4"`` for 4-bit via bitsandbytes, or ``"none"``.
            **Falls back to unquantised with a loud warning** if bitsandbytes is
            missing, and records what was actually used — a warrant pinned to
            "nf4" that was measured unquantised would be describing a different
            model.
        device_map: Passed to ``from_pretrained``.
        trust_remote_code: Left False by default. Qwen2.5 does not need it, and
            enabling it by default would execute arbitrary code from a hub
            repository on a machine that is about to handle evaluation data.
        reserve_gib_per_gpu: Memory left free on each GPU for activations and
            the attention workspace. With more than one GPU this forces the
            weights to spread rather than piling onto card 0 — a 7B model in NF4
            is only ~5 GB and accelerate will happily fit it all on one device,
            leaving the second idle and the first without headroom for a
            long-context forward pass.

    Returns:
        A :class:`LoadedModel`.

    Raises:
        PaddingError: If the tokenizer cannot be made to pad on the left.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _LOG.info("loading tokenizer %s", name)
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        # Qwen and most Llama-family tokenizers ship without a pad token. Reusing
        # EOS is standard; what matters is that it exists and is masked out, and
        # the attention mask handles that.
        tokenizer.pad_token = tokenizer.eos_token
        _LOG.info("tokenizer had no pad token; using eos_token %r", tokenizer.eos_token)
    assert_left_padding(tokenizer, where="load_model")

    quantization_config = None
    applied = "none"
    if quantization == "nf4":
        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            applied = "nf4"
        except Exception as exc:  # pragma: no cover - depends on the environment
            _LOG.warning(
                "NF4 requested but unavailable (%s); loading unquantised. The "
                "cache will record quantization='none', because a warrant "
                "pinned to nf4 that was measured unquantised describes a "
                "different model.",
                exc,
            )

    max_memory = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        max_memory = {}
        for index in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(index).total_memory / 2**30
            max_memory[index] = f"{max(1.0, total - reserve_gib_per_gpu):.1f}GiB"
        _LOG.info(
            "%d GPUs available; capping weights at %s so each card keeps ~%.0f "
            "GiB free for activations and the attention workspace",
            torch.cuda.device_count(),
            max_memory,
            reserve_gib_per_gpu,
        )

    _LOG.info("loading model %s (quantization=%s)", name, applied)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=quantization_config,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=torch.float16 if quantization_config is None else None,
        trust_remote_code=trust_remote_code,
        # Asked for explicitly rather than left to the default. The eager path
        # materialises the full heads x seq x seq score matrix and upcasts the
        # softmax to float32: 28 x 16000^2 x 4 bytes is 28.7 GiB for one op,
        # against O(seq) for SDPA's memory-efficient kernel. That difference is
        # invisible at the 100-token sequences of the short pass and fatal at
        # 16k, so the default being right on one machine proves nothing.
        attn_implementation="sdpa",
    )
    model.eval()

    implementation = getattr(model.config, "_attn_implementation", None)
    if implementation != "sdpa":
        raise RuntimeError(
            f"attention implementation is {implementation!r}, not 'sdpa'. Long "
            "context cannot run on the eager path: it builds a heads x seq x "
            "seq score matrix and upcasts the softmax to float32, which is "
            "28.7 GiB for a single op at 16k tokens on this model. Refusing "
            "rather than discovering it 40 minutes into a GPU session."
        )

    config = model.config
    device = str(next(model.parameters()).device)
    dtype = str(next(model.parameters()).dtype)
    loaded = LoadedModel(
        model=model,
        tokenizer=tokenizer,
        name=name,
        num_hidden_layers=int(config.num_hidden_layers),
        hidden_size=int(config.hidden_size),
        quantization=applied,
        device=device,
        dtype=dtype,
    )
    devices = sorted({str(p.device) for p in model.parameters()})
    _LOG.info(
        "loaded %s: %d layers, hidden %d, %s across %s",
        name,
        loaded.num_hidden_layers,
        loaded.hidden_size,
        applied,
        devices,
    )
    return loaded


def build_prompt(tokenizer: Any, question: str, system: Optional[str] = None) -> str:
    """Render a question through the model's chat template.

    Uses the tokenizer's own template rather than a hand-written format, because
    the activations we care about are the ones the model actually produces in
    deployment, and deployment goes through the template. A hand-rolled prompt
    measures a distribution the model is never asked to handle.

    ``add_generation_prompt=True`` so the final position is the one the model is
    about to generate from — which is the question-time position the whole
    method rests on: after reading the question, before producing a token.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_tokenizer(name: str, *, trust_remote_code: bool = False) -> Any:
    """Load a tokenizer on its own, without the weights it usually arrives with.

    The envelope stage needs to count model tokens and nothing else. Going
    through :func:`load_model` to get there would pull a 7B checkpoint onto a
    CPU box to answer a question the vocabulary already answers, so this is the
    tokenizer-only door.

    No left-padding assertion here, deliberately. That assertion protects
    *activation extraction*, where a right-padded batch makes position ``-1`` a
    pad token and the measurement meaningless. Counting the tokens in a single
    prompt reads no activations and pads nothing, and asserting a property this
    caller does not depend on would only teach the next reader that the
    assertion is ceremonial.

    Args:
        name: HuggingFace model id, from ``config.model.name``.
        trust_remote_code: Left False for the reason given in
            :func:`load_model` -- a tokenizer load is still a code path a hub
            repository can reach.

    Returns:
        The tokenizer.
    """
    from transformers import AutoTokenizer

    _LOG.info("loading tokenizer %s (no weights)", name)
    return AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)


def token_length_summary(prompts: Sequence[str], tokenizer: Any) -> dict:
    """Model-token length of each prompt: mean, median, IQR, and the extremes.

    **Why this is not a nicety.** The probe reads activations, and activations
    are indexed by model tokens. A whitespace count is a proxy for that, and it
    is a proxy that fails in exactly the direction this project cares about:
    Qwen2.5 fragments romanised Hindi far more aggressively than English, and a
    digit run like ``9999 0687 2026`` can split per-digit. Three whitespace
    tokens can be twenty model tokens. Sequence length is also one of the few
    things a question-time probe is most plausibly sensitive to, so measuring
    the wrong length is measuring the wrong envelope.

    ``add_special_tokens=False`` because the chat template's wrapper is constant
    across every item in both sets. Including it would add the same offset to
    both means and pull the ratio toward 1.0 -- flattering the comparison by
    diluting the difference with boilerplate.

    Args:
        prompts: The raw prompts, before chat templating.
        tokenizer: A tokenizer from :func:`load_tokenizer` or
            :func:`load_model`.

    Returns:
        A dict with ``n``, ``mean``, ``median``, ``p25``, ``p75``, ``iqr``,
        ``min`` and ``max``. Quantiles are linear-interpolated, matching numpy's
        default, so they are comparable against the score quantiles used
        elsewhere.

    Raises:
        ValueError: If ``prompts`` is empty. An envelope over nothing is not an
            envelope, and returning zeros would look like a measurement.
    """
    if not prompts:
        raise ValueError("token_length_summary needs at least one prompt")

    lengths = [
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
    ]
    ordered = sorted(lengths)

    def _quantile(q: float) -> float:
        """Linear-interpolated quantile, numpy's default convention."""
        if len(ordered) == 1:
            return float(ordered[0])
        position = q * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

    p25 = _quantile(0.25)
    p75 = _quantile(0.75)
    return {
        "n": len(lengths),
        "mean": statistics.fmean(lengths),
        "median": statistics.median(lengths),
        "p25": p25,
        "p75": p75,
        "iqr": p75 - p25,
        "min": min(lengths),
        "max": max(lengths),
        "stdev": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
    }
