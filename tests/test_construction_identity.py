"""`construction` records the inputs to generation, never the generator's code.

``DECISIONS.md`` 098, the mirror of 092.

An eval set's ``construction`` dict is inside its content hash, the content hash
**is** the envelope id, and the envelope id is the third element of every
warrant key. So anything written into ``construction`` becomes part of the
identity that warrants are filed under.

A seed belongs there: change it and the data genuinely differs. A module path
does not: change it and nothing about the data differs, but every warrant keyed
on that envelope is orphaned — silently, because a changed hash does not raise.
The numbers simply stop belonging to anything.

That nearly happened. ``controlplane/validation/synthetic.py`` writes
``"generator": "src.validation.synthetic.synthetic_evalset"`` into
``construction``, and the ``src/`` -> ``controlplane/`` rename in 095 would have
re-issued every synthetic fixture under a new envelope. It was caught by
checking, and the two literals are frozen in place rather than corrected.

Freezing two sites is a patch. This is the rule: no ``construction`` dict may
carry a string that resolves to a module in this package.
"""

from __future__ import annotations

import importlib.util
import re

import pytest

from controlplane.config import Config
from controlplane.validation.synthetic import synthetic_cache, synthetic_evalset

#: The two literals frozen by DECISIONS 098, allowlisted **by exact value** so
#: the allowlist cannot silently absorb a third. Correcting them would change
#: the envelope id of every synthetic fixture and orphan the warrants in
#: results/fixtures/, which is the damage this rule exists to prevent.
FROZEN = {
    "src.validation.synthetic.synthetic_evalset",
    "src.validation.synthetic.synthetic_cache",
}

#: Something.that.looks.like.this -- at least two dotted segments, identifier
#: characters only, no spaces. Deliberately loose: a false positive costs one
#: allowlist entry with a reason, a false negative costs an orphaned envelope.
_DOTTED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$")

#: Package roots whose names appearing in a hashed dict is the actual hazard.
#: `src` is included because that is what the frozen literals say and what a
#: copy-paste of them would say.
_OUR_ROOTS = ("controlplane", "src", "scripts", "tests", "demo")


def _looks_like_our_code(value: str) -> bool:
    """Whether a string is a dotted path into this project's code.

    Checks the root against our package names and, where the root is importable,
    that the module actually exists -- so an ordinary dotted string like a
    filename or a version does not trip it.
    """
    if not _DOTTED.match(value):
        return False
    root = value.split(".", 1)[0]
    if root not in _OUR_ROOTS:
        return False
    if root == "src":
        # The historical name. Not importable any more, which is precisely why
        # the frozen literals are stale -- and precisely why they must still be
        # caught rather than passing because the module has gone.
        return True
    try:
        return importlib.util.find_spec(root) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def _offending(mapping: dict, where: str) -> list[str]:
    """Every key in a hashed dict whose value names this project's code."""
    found = []
    for key, value in (mapping or {}).items():
        if not isinstance(value, str):
            continue
        if value in FROZEN:
            continue
        if _looks_like_our_code(value):
            found.append(f"{where}[{key!r}] = {value!r}")
    return found


def _message(offenders: list[str]) -> str:
    return (
        "a hashed construction record names this project's code:\n  "
        + "\n  ".join(offenders)
        + "\n\nDECISIONS 098: `construction` records the INPUTS to generation "
        "-- seed, sizes, requested rates -- never the code identity of the "
        "generator. It is inside the content hash, the content hash is the "
        "envelope id, and the envelope id is the third element of every "
        "warrant key. A module path there couples every warrant to the package "
        "layout, so renaming a package silently orphans them.\n\n"
        "Put it in `provenance` instead, which every artifact already carries "
        "and which is deliberately outside the hash."
    )


def test_construction_records_inputs_not_code_identity(config: Config) -> None:
    """The rule, on the synthetic eval set builder."""
    evalset = synthetic_evalset(
        eval_set_id="triviaqa-600-synthetic", n_items=120, base_rate=0.4, seed=1729
    )
    offenders = _offending(evalset.construction, "construction")
    assert not offenders, _message(offenders)


def test_the_extraction_cache_extra_carries_no_code_identity(config: Config) -> None:
    """Caches too: ``extra`` travels with the cache and into what reads it."""
    evalset = synthetic_evalset(
        eval_set_id="triviaqa-600-synthetic", n_items=120, base_rate=0.4, seed=1729
    )
    cache = synthetic_cache(
        evalset, seed=1729,
        window=config.probe.rolling_window, stride=config.probe.rolling_stride,
    )
    offenders = _offending(getattr(cache, "extra", {}) or {}, "cache.extra")
    assert not offenders, _message(offenders)


def test_the_frozen_literals_are_still_exactly_the_two_that_were_frozen(
    config: Config,
) -> None:
    """The allowlist must not grow by accident.

    Allowlisting by exact value rather than by key means a third generator
    string cannot slip in under an existing exemption -- it would have to be
    added here deliberately, which is a review someone has to pass.
    """
    evalset = synthetic_evalset(
        eval_set_id="triviaqa-600-synthetic", n_items=40, base_rate=0.4, seed=1729
    )
    cache = synthetic_cache(
        evalset, seed=1729,
        window=config.probe.rolling_window, stride=config.probe.rolling_stride,
    )
    seen = {
        value
        for mapping in (evalset.construction, getattr(cache, "extra", {}) or {})
        for value in (mapping or {}).values()
        if isinstance(value, str) and value in FROZEN
    }
    assert seen == FROZEN, (
        f"the frozen generator literals changed: expected {sorted(FROZEN)}, "
        f"found {sorted(seen)}. If one was corrected, every synthetic fixture "
        "now has a different envelope id and the warrants in results/fixtures/ "
        "are orphaned. See DECISIONS 098."
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("controlplane.validation.synthetic.synthetic_evalset", True),
        ("src.validation.synthetic.synthetic_cache", True),
        ("scripts.10_freeze_scores", False),  # not an identifier: digits lead
        ("tests.factories", True),
        ("hinglish-pii-200.json", False),
        ("2.2.364", False),
        ("hand-written scenarios crossed with systematic disclosure forms", False),
        ("sha256:e77c64f908491ad1", False),
        ("bootstrap-percentile-1000 over questions, seed=1729", False),
    ],
)
def test_the_detector_distinguishes_code_paths_from_ordinary_strings(
    value: str, expected: bool
) -> None:
    """A rule that fired on every dotted string would be turned off within a week."""
    assert _looks_like_our_code(value) is expected


def test_a_new_generator_string_is_caught() -> None:
    """The rule must fire on the next occurrence, not only describe the last two.

    Freezing the two known literals is a patch; this is the part that makes the
    class non-recurring. A third generator string -- under any key, in any
    builder -- fails here rather than surviving to a rename.
    """
    offenders = _offending(
        {
            "seed": 1729,
            "n_items": 600,
            "generator": "controlplane.evalsets.builders.build_something",
        },
        "construction",
    )
    assert len(offenders) == 1
    assert "generator" in offenders[0]
    assert "DECISIONS 098" in _message(offenders)


def test_the_legitimate_inputs_are_left_alone() -> None:
    """Everything that genuinely belongs inside the hash must pass untouched."""
    assert not _offending(
        {
            "seed": 1729,
            "n_items": 600,
            "requested_base_rate": 0.4,
            "items_per_question": 1,
            "long_context": False,
            "method": "hand-written scenarios crossed with systematic forms",
            "label_meaning": "1 = message contains a personal identifier",
            "warning": "Synthetic fixture. Exercises the harness.",
            "derived_from": "triviaqa-2400",
        },
        "construction",
    )
