"""Model and tokenizer loading, NF4 quantisation, chat templating.

The one thing in this module that will ruin the experiment if it is wrong is
the padding side. With right padding, position ``-1`` of a padded sequence is a
pad token, so every activation extracted at "the last prompt token" is actually
read off padding. Nothing raises, the pipeline completes, and AUROC lands near
0.5 -- which reads as "the idea does not work" rather than as a bug. So left
padding is forced at load time and asserted here, and asserted again before
every batched call in ``extract.py`` (CLAUDE.md invariant 4).
"""

import logging
from typing import Any, Optional, Sequence

from src.config import Config

LOGGER = logging.getLogger(__name__)


class PaddingSideError(ValueError):
    """Raised when a tokenizer is not left-padded.

    Its own type because this is invariant 4, the highest-value check in the
    repo, and a caller should never be able to catch it by accident while
    meaning to catch something else.
    """


def resolve_dtype(name: str) -> Any:
    """Map a config dtype name to a torch dtype."""
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype {name!r}; expected one of {sorted(mapping)}")
    return mapping[name]


def _dtype_kwarg(dtype: Any) -> dict[str, Any]:
    """Return the version-correct dtype keyword for ``from_pretrained``.

    transformers renamed ``torch_dtype`` to ``dtype`` in v5. v5 still accepts
    the old name for backwards compatibility, but v4 does not know the new one
    and would swallow it into ``**kwargs`` -- loading the model in fp32 and
    exhausting a 16GB GPU with no error message pointing at the cause. The
    reference environment is a Colab/Kaggle image whose transformers version we
    do not control, so the keyword is chosen at runtime.
    """
    import transformers

    major = int(transformers.__version__.split(".")[0])
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


def build_quantization_config(config: Config) -> Optional[Any]:
    """Build the NF4 ``BitsAndBytesConfig``, or None when quantisation is off.

    NF4 is what makes a 7B model fit alongside its activations on a 16GB T4
    (CLAUDE.md, Environment notes). It requires CUDA, so a CPU run must set
    ``model.quantization: none`` rather than silently falling back -- a silent
    fallback would change the experimental condition without changing the
    config hash recorded next to the results.

    Args:
        config: Resolved experiment config.

    Returns:
        A ``BitsAndBytesConfig`` for NF4, or None.

    Raises:
        RuntimeError: if NF4 is requested without a CUDA device.
    """
    if config.model.quantization == "none":
        return None

    import torch
    from transformers import BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError(
            "model.quantization is 'nf4' but no CUDA device is visible. "
            "bitsandbytes NF4 is GPU-only. Either run on a GPU, or set "
            "model.quantization: none in the config -- do not let it fall back "
            "silently, because that would change the experimental condition "
            "without changing the config hash."
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=resolve_dtype(config.model.dtype),
        bnb_4bit_use_double_quant=True,
    )


def assert_left_padding(tokenizer: Any) -> None:
    """Assert the tokenizer pads on the left (CLAUDE.md invariant 4).

    Called at load time and again before every batched forward pass. Cheap, and
    it is the only thing standing between a right-padded batch and a set of
    activations read entirely off pad tokens.

    Args:
        tokenizer: Any object with a ``padding_side`` attribute.

    Raises:
        PaddingSideError: if the padding side is anything but ``"left"``.
    """
    side = getattr(tokenizer, "padding_side", None)
    if side != "left":
        raise PaddingSideError(
            f"tokenizer.padding_side is {side!r}, must be 'left' "
            "(CLAUDE.md invariant 4). With right padding, position -1 is a pad "
            "token and every extracted activation is meaningless, with nothing "
            "raised to tell you."
        )


def configure_tokenizer(tokenizer: Any) -> Any:
    """Force left padding and ensure a pad token exists.

    Qwen2.5 ships without a distinct pad token; batching needs one, and reusing
    EOS is the standard choice. Padded positions are masked out of attention, so
    reusing EOS cannot contaminate the activation we read.

    Args:
        tokenizer: A loaded tokenizer.

    Returns:
        The same tokenizer, mutated in place.
    """
    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        LOGGER.info("pad_token was unset; using eos_token %r", tokenizer.eos_token)
    assert_left_padding(tokenizer)
    return tokenizer


def load_tokenizer(config: Config) -> Any:
    """Load the tokenizer for the configured model, left-padded.

    Args:
        config: Resolved experiment config.

    Returns:
        The configured tokenizer.
    """
    from transformers import AutoTokenizer

    LOGGER.info("loading tokenizer %s", config.model.name)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    return configure_tokenizer(tokenizer)


def load_model_and_tokenizer(config: Config) -> tuple[Any, Any]:
    """Load the quantised model and its left-padded tokenizer.

    The model is put in eval mode: there is no training anywhere in this repo
    beyond the logistic regression, and dropout left active would make greedy
    decoding non-deterministic.

    Args:
        config: Resolved experiment config.

    Returns:
        ``(model, tokenizer)``.
    """
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(config)
    quantization_config = build_quantization_config(config)

    kwargs: dict[str, Any] = {
        "device_map": config.model.device_map,
        **_dtype_kwarg(resolve_dtype(config.model.dtype)),
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config

    LOGGER.info(
        "loading model %s (quantization=%s, dtype=%s, device_map=%s)",
        config.model.name,
        config.model.quantization,
        config.model.dtype,
        config.model.device_map,
    )
    model = AutoModelForCausalLM.from_pretrained(config.model.name, **kwargs)
    model.eval()
    if torch.cuda.is_available():
        LOGGER.info("peak GPU memory after load: %.2f GB", peak_memory_gb())
    return model, tokenizer


def peak_memory_gb() -> Optional[float]:
    """Peak CUDA memory allocated so far, in GB, or None on CPU."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024**3)


def build_prompt(tokenizer: Any, question: str, config: Config) -> str:
    """Render one question through the chat template, ready for prefill.

    ``add_generation_prompt=True`` appends the assistant turn header, which is
    what makes the final prompt token the last token before the model would
    start answering -- exactly the position this experiment probes (SPEC.md §3,
    CLAUDE.md invariant 1).

    The system prompt comes from config and is fixed for the whole run: it is
    part of the experimental condition, so varying it mid-run would mean two
    different experiments sharing one set of labels.

    Args:
        tokenizer: A tokenizer exposing ``apply_chat_template``.
        question: The raw question text.
        config: Resolved experiment config.

    Returns:
        The rendered prompt string.
    """
    messages = [
        {"role": "system", "content": config.prompt.system},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=config.prompt.add_generation_prompt,
    )


def build_prompts(
    tokenizer: Any, questions: Sequence[str], config: Config
) -> list[str]:
    """Render many questions through the chat template."""
    return [build_prompt(tokenizer, q, config) for q in questions]


def resolve_layers(model: Any, config: Config) -> tuple[int, ...]:
    """Resolve the configured fractional depths against this model's depth.

    Args:
        model: A loaded causal LM.
        config: Resolved experiment config.

    Returns:
        Absolute ``hidden_states`` indices, all within ``[1, num_hidden_layers]``.
    """
    n_layers = int(model.config.num_hidden_layers)
    layers = config.resolve_layers(n_layers)
    LOGGER.info(
        "model has %d hidden layers; probing %s (fractions %s)",
        n_layers,
        list(layers),
        list(config.model.layer_fractions),
    )
    return layers


def describe_model(model: Any, tokenizer: Any, config: Config) -> dict[str, Any]:
    """Summarise the loaded model for the Stage 2 gate and for artifacts.

    Args:
        model: A loaded causal LM.
        tokenizer: Its tokenizer.
        config: Resolved experiment config.

    Returns:
        A JSON-serialisable description including the resolved layer indices
        and the exact rendered prompt template (SPEC.md §3).
    """
    return {
        "name": config.model.name,
        "quantization": config.model.quantization,
        "dtype": config.model.dtype,
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "probe_layers": list(resolve_layers(model, config)),
        "layer_fractions": list(config.model.layer_fractions),
        "padding_side": tokenizer.padding_side,
        "pad_token": tokenizer.pad_token,
        "eos_token": tokenizer.eos_token,
        "peak_memory_gb_after_load": peak_memory_gb(),
        "example_prompt": build_prompt(tokenizer, "<QUESTION>", config),
        "system_prompt": config.prompt.system,
    }


def sanity_generate(
    model: Any, tokenizer: Any, question: str, config: Config
) -> str:
    """Generate one greedy answer, for the Stage 2 gate check.

    Lives here rather than in a notebook because notebooks in this repo hold no
    logic (CLAUDE.md, Coding standards). Its only job is to let a human look at
    one completion and confirm it is a short answer rather than an empty string
    or a repetition loop.

    Args:
        model: A loaded causal LM.
        tokenizer: Its left-padded tokenizer.
        question: The question to answer.
        config: Resolved experiment config.

    Returns:
        The decoded completion, prompt stripped.
    """
    import torch

    assert_left_padding(tokenizer)
    prompt = build_prompt(tokenizer, question, config)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=config.generation.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_len = enc["input_ids"].shape[1]
    return tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)[0]
