"""Evaluation sets: hand-written corpora, builders, and the frozen registry.

``CLAUDE.md`` invariant 9 lives here. A set's content hash IS its envelope id
and therefore the third element of every warrant key measured on it, so a
changed set is a different set and inherits nothing.

``DECISIONS.md`` 007: the Hinglish and hard-negative corpora are hand-written.
Generating an evaluation set with a model makes that model's judgment the
ground truth for measuring models, which is circular.
"""

from .builders import (
    build_canary_pii,
    build_hard_negatives,
    build_hinglish_pii,
    build_longctx,
)
from .identifiers import (
    DISCLOSURE_FORMS,
    Identifier,
    verhoeff_check_digit,
    verhoeff_is_valid,
)
from .registry import (
    MANIFEST_NAME,
    load_evalset,
    read_manifest,
    save_evalset,
    verify_manifest,
    write_manifest,
)

__all__ = [
    "DISCLOSURE_FORMS",
    "MANIFEST_NAME",
    "Identifier",
    "build_canary_pii",
    "build_hard_negatives",
    "build_hinglish_pii",
    "build_longctx",
    "load_evalset",
    "read_manifest",
    "save_evalset",
    "verhoeff_check_digit",
    "verhoeff_is_valid",
    "verify_manifest",
    "write_manifest",
]
