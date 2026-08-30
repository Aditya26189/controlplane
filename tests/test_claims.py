"""Every number in README.md resolves to a field in results/. Block E, E.6.

A README number edited by hand and drifting from its artifact is the same
failure as an uncertified response that looks certified: the reader cannot
tell, and nothing objects. This suite makes it impossible rather than a matter
of discipline.

The negative tests matter more than the positive one. A checker that passes on
the real README proves very little on its own -- it would also pass if it
silently found no claims, or compared nothing. So the tampering cases are here
too: a wrong value, a wrong interval, a missing artifact, a dead field.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from controlplane.report.claims import (
    check_claims,
    parse_claim_table,
    render,
    resolve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = PROJECT_ROOT / "README.md"


# --------------------------------------------------------------------------- #
# The real table
# --------------------------------------------------------------------------- #


def test_the_readme_has_a_claim_table() -> None:
    claims = parse_claim_table(README)
    assert len(claims) >= 20, (
        f"README.md's claim table parsed to {len(claims)} rows. E.3 requires "
        "every quantitative claim in the README, the proposal and the deck to "
        "appear here."
    )


def test_every_readme_claim_resolves() -> None:
    """The headline check: every committed number matches its artifact.

    On failure the report names the artifact, the field and both values, so the
    fix is obvious and is never "edit the number until it matches".
    """
    results = check_claims(PROJECT_ROOT, README)
    failed = [r for r in results if not r.ok]
    assert not failed, "\n" + render(results)


def test_every_claim_names_an_artifact_that_exists() -> None:
    for claim in parse_claim_table(README):
        assert (PROJECT_ROOT / claim.artifact).is_file(), (
            f"claim {claim.label!r} names {claim.artifact}, which does not exist"
        )


def test_every_claim_names_a_regeneration_command() -> None:
    """A number with no command behind it cannot be checked by a reader."""
    for claim in parse_claim_table(README):
        assert claim.command.strip(), (
            f"claim {claim.label!r} names no command that regenerates it"
        )


# --------------------------------------------------------------------------- #
# Tampering -- the checker must actually catch drift
# --------------------------------------------------------------------------- #


def _tampered(tmp_path: Path, old: str, new: str) -> Path:
    text = README.read_text(encoding="utf-8")
    assert old in text, f"fixture text {old!r} is no longer in README.md"
    path = tmp_path / "README.md"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def test_a_hand_edited_value_is_caught(tmp_path: Path) -> None:
    """The failure this whole mechanism exists for."""
    readme = _tampered(tmp_path, "| 0.8256 | [0.7934, 0.8567]", "| 0.9256 | [0.7934, 0.8567]")
    results = check_claims(PROJECT_ROOT, readme)
    failed = [r for r in results if not r.ok]
    assert len(failed) == 1, render(results)
    assert "0.9256" in failed[0].detail and "0.8256" in failed[0].detail


def test_a_hand_edited_interval_is_caught(tmp_path: Path) -> None:
    """A point estimate that matches with a widened interval that does not.

    Easier to do by accident than editing the value, and harder to spot.
    """
    readme = _tampered(tmp_path, "[0.7934, 0.8567]", "[0.7134, 0.8567]")
    failed = [r for r in check_claims(PROJECT_ROOT, readme) if not r.ok]
    assert len(failed) == 1
    assert "ci_low" in failed[0].detail


def test_a_claim_pointing_at_a_missing_artifact_is_caught(tmp_path: Path) -> None:
    readme = _tampered(
        tmp_path,
        "results/validation-T1-last_token.json | metrics.auroc.value",
        "results/does-not-exist.json | metrics.auroc.value",
    )
    failed = [r for r in check_claims(PROJECT_ROOT, readme) if not r.ok]
    assert failed and "artifact not found" in failed[0].detail


def test_a_claim_pointing_at_a_dead_field_is_caught(tmp_path: Path) -> None:
    """The silent one: the artifact is there, the field inside it is not."""
    readme = _tampered(
        tmp_path,
        "results/validation-T1-last_token.json | metrics.auroc.value",
        "results/validation-T1-last_token.json | metrics.auroc.mean",
    )
    failed = [r for r in check_claims(PROJECT_ROOT, readme) if not r.ok]
    assert failed and "not found" in failed[0].detail


def test_an_empty_table_is_an_error_not_a_pass(tmp_path: Path) -> None:
    """Zero claims must never read as zero failures."""
    (tmp_path / "README.md").write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="no claim table"):
        check_claims(PROJECT_ROOT, tmp_path / "README.md")


# --------------------------------------------------------------------------- #
# The resolver
# --------------------------------------------------------------------------- #


def test_a_selector_matching_nothing_raises() -> None:
    doc = {"runs": [{"id": "a"}, {"id": "b"}]}
    with pytest.raises(KeyError, match="matched 0 elements"):
        resolve(doc, "runs[id=c]")


def test_a_selector_matching_two_rows_raises() -> None:
    """Taking the first match would claim whichever row was written first."""
    doc = {"runs": [{"id": "a", "v": 1}, {"id": "a", "v": 2}]}
    with pytest.raises(KeyError, match="matched 2 elements"):
        resolve(doc, "runs[id=a].v")


def test_a_selector_key_may_itself_be_dotted() -> None:
    """The policy artifact needs this, and the first splitter got it wrong."""
    doc = {"ops": [{"op": {"id": "P-x"}, "v": 7}, {"op": {"id": "P-y"}, "v": 9}]}
    assert resolve(doc, "ops[op.id=P-y].v") == 9


def test_the_readme_test_count_is_the_real_one() -> None:
    """A hardcoded count in the README drifts the moment a test is added.

    It drifted twice while Block E was being written, which is why this is a
    check rather than a note asking someone to remember. The count is a claim
    like any other; it just happens to live in the suite instead of in an
    artifact, so it is checked against the suite.
    """
    import re
    import subprocess
    import sys

    stated = {int(m) for m in re.findall(r"(\d{3,4}) tests", README.read_text(encoding="utf-8"))}
    assert stated, "README.md no longer states a test count"

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "tests/"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=900,
    )
    match = re.search(r"(\d+) tests? collected", result.stdout)
    assert match, (
        "could not read a collection count from pytest:\n" + result.stdout[-1500:]
    )
    collected = int(match.group(1))
    assert stated == {collected}, (
        f"README.md states {sorted(stated)} tests; pytest collects {collected}. "
        "Update the README -- and if two different counts are stated, make them "
        "agree."
    )
