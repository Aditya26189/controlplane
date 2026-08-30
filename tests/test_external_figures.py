"""The external-figure register gates the proposal, mechanically.

``DECISIONS.md`` 113, 115. A number about the world is not measured here,
derived here, or a declared estimate, so ``PROPOSAL.md``'s three rules did not
cover it and it entered as ordinary prose. That is how a hypothetical
``DEFF 1.60`` from a planning document reached a sentence about this project's
certification cost, and how ``prEN 18284`` was cited for Article 15 when it is
the dataset-governance standard.

The register exists to stop that. These tests exist because a register nobody
checks is a register nobody consults -- the same argument as ``050``: reading
is not a control.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTER = PROJECT_ROOT / "docs" / "EXTERNAL_FIGURES.md"
PROPOSAL = PROJECT_ROOT / "docs" / "PROPOSAL.md"


def _flat(text: str) -> str:
    """Collapse whitespace. These are prose assertions, and prose wraps --
    a phrase split across two lines is the same phrase, and a test that
    disagrees is testing the line width."""
    return re.sub(r"\s+", " ", text)


def _register() -> str:
    return REGISTER.read_text(encoding="utf-8")


def _proposal() -> str:
    return PROPOSAL.read_text(encoding="utf-8")


#: Figures the register marks `dropped` or `unverified`, each with a token that
#: would appear if it were used. The register may DISCUSS them -- that is its
#: job -- but the proposal must not carry them.
FORBIDDEN_IN_PROPOSAL = {
    "prEN 18284": "the dataset-governance standard, cited for Article 15 in an earlier draft",
    "350+": "no official ISO 42001 certificate count exists",
    "Ponemon": "vendor-sponsored, and the headline is a mean not a median",
    "1.000 false-positive": "traces to PromptGuard, which is not a shipped commercial product",
    "Platform Work Directive": "single non-primary source, unverified",
}


def test_the_register_claims_no_primary_source() -> None:
    """Nothing here was checked against an issuing body's own text.

    The first version of this register asserted `verified+primary`. What was
    actually consulted was law-firm briefings, LexisNexis summaries and news
    reporting -- not the AP decision, the OJ text, or the BCCRT judgment. The
    register built to stop a number arriving with its provenance stripped was
    doing exactly that in its own provenance column.

    If a `primary` tier ever appears, it means someone read the source, and
    this test is the thing that makes that a deliberate act.
    """
    text = _register()
    # It appears exactly once, in the paragraph that retracts it. Any
    # second occurrence is an entry claiming the tier.
    occurrences = text.count("verified+primary")
    assert occurrences == 1, (
        f"`verified+primary` appears {occurrences} times; it belongs only in "
        "the retraction, never as an entry's tier"
    )
    assert "That was an overclaim" in _flat(text)
    assert "Nothing in this register reaches `primary`" in _flat(text)


def test_the_two_secondary_tiers_are_distinguished() -> None:
    """Cross-checked and relayed are different, and collapsing them is the
    defect the register exists to prevent."""
    text = _register()
    assert "`secondary+crosschecked`" in text
    assert "`secondary+relayed`" in text
    assert "never be load-bearing" in _flat(text)


@pytest.mark.parametrize(
    "figure",
    ["Moffatt", "Foodinho", "Armilla", "Verisk", "40% of announced GDPR fines"],
)
def test_relayed_figures_sit_under_the_relayed_heading(figure: str) -> None:
    """A relayed figure promoted into the cross-checked list would be the
    DEFF 1.60 shape a second time."""
    text = _register()
    relayed_start = text.index("## `secondary+relayed`")
    relayed_end = text.index("## The Digital Omnibus")
    assert figure in text[relayed_start:relayed_end], (
        f"{figure!r} is not under the relayed heading; if it was promoted, "
        "someone must have checked it"
    )


def test_the_anchor_does_not_lean_on_a_relayed_percentage() -> None:
    """The ~40% figure is relayed, so the appeal is stated without it."""
    text = _register()
    uber = text[text.index("### Uber") : text.index("### Regulation (EU)")]
    assert "appealed" in uber
    assert "`secondary+relayed`" in uber, (
        "the Uber entry no longer flags that the 40% figure it would otherwise "
        "lean on is relayed"
    )


@pytest.mark.parametrize("token,why", sorted(FORBIDDEN_IN_PROPOSAL.items()))
def test_a_dropped_or_unverified_figure_is_not_in_the_proposal(
    token: str, why: str
) -> None:
    """Dropped, not softened. Softening is how DEFF 1.60 survived."""
    assert token not in _proposal(), (
        f"{token!r} appears in PROPOSAL.md; the register marks it dropped or "
        f"unverified because {why}. Remove it rather than hedging it."
    )


@pytest.mark.parametrize("token", sorted(FORBIDDEN_IN_PROPOSAL))
def test_every_forbidden_token_is_actually_discussed_in_the_register(
    token: str,
) -> None:
    """Guards against the list rotting into tokens nobody recorded a reason for."""
    assert token in _register(), (
        f"{token!r} is checked against the proposal but is not in the register, "
        "so nothing states why it is forbidden"
    )


def test_the_correct_article_15_standard_is_the_one_named() -> None:
    """The corrected citation, pinned so the wrong one cannot come back."""
    text = _register()
    assert "prEN 18229-2" in text
    assert "prEN ISO/IEC 23282" in text
    # And the register must say plainly what 18284 actually is.
    assert re.search(r"prEN 18284`?\*{0,2} is \*{0,2}dataset", text), (
        "the register no longer states what prEN 18284 actually covers, which "
        "is what stops it being cited for Article 15 again"
    )


def test_the_uber_anchor_carries_both_caveats() -> None:
    """An anchor quoted without its instability invites a reader to supply it."""
    text = _register()
    assert "824,990,000" in text
    assert "appeal" in text.lower(), "the appeal caveat is missing"
    assert "40%" in text, "the annulled/challenged proportion is missing"
    assert "Digital Omnibus" in text, "the Article 22 instability is missing"


def test_the_omnibus_is_framed_as_an_argument_not_only_a_caveat() -> None:
    """A permission list generates a standing evidence obligation, which is
    what a warrant serves. Losing that framing loses the stronger half."""
    text = _register()
    assert "standing evidence obligation" in _flat(text)
    assert "revocable condition" in _flat(text)


def test_the_omnibus_is_marked_a_proposal_not_law() -> None:
    """It sits beside Regulation (EU) 2026/1744, which IS in force. Unmarked,
    a reader takes both as enacted and finds out otherwise by checking."""
    text = _register()
    omnibus = text[text.index("## The Digital Omnibus") :]
    assert "a proposal, not law" in _flat(omnibus)
    assert "ordinary legislative procedure" in _flat(omnibus)


def test_the_omnibus_argument_survives_both_readings() -> None:
    """Staking it on the contested reading makes the pitch depend on an
    outcome the EDPB and EDPS are actively pushing against."""
    text = _register()
    omnibus = text[text.index("## The Digital Omnibus") :]
    assert "broad" in omnibus and "narrow" in omnibus
    assert "22(3)" in omnibus, "the narrow reading's safeguards are unstated"
    assert "either way" in _flat(omnibus)
    assert "Do not stake the argument on the broad reading" in _flat(omnibus)


def test_the_proposal_declares_the_fourth_class() -> None:
    """Rule 4 is the gate. Without it the register is advice."""
    text = _proposal()
    assert "EXTERNAL_FIGURES.md" in text
    assert "dropped, not softened" in text


def test_secondary_sources_are_marked_as_claims() -> None:
    """Vendor and trade press are cited as claims by that party, never as fact."""
    text = _register()
    assert "`secondary+relayed`" in text
    assert "vendor and trade press" in _flat(text).lower()
    assert "relayed" in text[text.index("### Armilla") : text.index("### \"~40%")]
