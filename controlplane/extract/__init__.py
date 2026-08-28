"""The extraction stage: the only part that needs a GPU.

Question-time activations, taken from a **separate prefill forward pass** before
any token has been generated. Never from inside ``generate()`` -- retaining
hidden states across generated steps exhausts a 16GB card, and the position that
matters is the last prompt token, which exists before generation starts.

Everything here imports torch lazily inside functions, so the rest of the
project stays runnable on a laptop with no GPU stack.
"""

from .activations import capture_padding_evidence, extract_activations, generate_answers
from .model import LoadedModel, PaddingError, assert_left_padding, build_prompt, load_model
from .pipeline import ExtractionResult, extract_triviaqa
from .triviaqa import TriviaItem, is_correct, load_triviaqa, normalise_answer, split_questions

__all__ = [
    "ExtractionResult",
    "LoadedModel",
    "PaddingError",
    "TriviaItem",
    "assert_left_padding",
    "build_prompt",
    "capture_padding_evidence",
    "extract_activations",
    "extract_triviaqa",
    "generate_answers",
    "is_correct",
    "load_model",
    "load_triviaqa",
    "normalise_answer",
    "split_questions",
]
