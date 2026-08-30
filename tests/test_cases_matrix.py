"""The case matrix cannot drift from the suite. Block E, E.5.

``docs/CASES.md`` is the answer to "have you covered all the cases?", and a
table of cases is only worth reading if it describes the code. A row naming a
test that was renamed, deleted, or never written is worse than no table: it
reads as coverage and is not.

So the table is checked rather than trusted. Every test name it cites must
exist, every artifact path it cites must exist, and every row must actually
name something in both columns.

This is the same argument as everything else here. An absent test reading as a
passing one is the failure mode this project exists to make impossible.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES = PROJECT_ROOT / "docs" / "CASES.md"

#: A markdown table row: leading pipe, at least three cells.
_ROW = re.compile(r"^\|(?!\s*[-: ]+\|)(.+)\|\s*$")
#: Anything in backticks.
_TICKED = re.compile(r"`([^`]+)`")
#: A test function definition in any test module.
_DEF = re.compile(r"^\s*def (test_[A-Za-z0-9_]+)")


def _defined_tests() -> set[str]:
    """Every test function name defined under tests/, by source inspection.

    Deliberately not pytest collection: collection skips modules whose optional
    dependencies are absent -- presidio is the live example -- and a test that
    exists but is skipped here is still a test the table may legitimately cite.
    A test that does not exist at all is the thing being caught.
    """
    names: set[str] = set()
    for path in (PROJECT_ROOT / "tests").glob("test_*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = _DEF.match(line)
            if match:
                names.add(match.group(1))
    return names


def _rows() -> list[list[str]]:
    """Every data row of every table in CASES.md, as lists of cells."""
    rows: list[list[str]] = []
    for line in CASES.read_text(encoding="utf-8").splitlines():
        match = _ROW.match(line.strip())
        if not match:
            continue
        cells = [c.strip() for c in match.group(1).split("|")]
        # Skip header rows: they are the ones whose last cell is a column name.
        if cells and cells[-1] in {"artifact", "test", "on drift"}:
            continue
        rows.append(cells)
    return rows


def _cited_tests(cells: list[str]) -> list[str]:
    """Test names cited anywhere in a row."""
    cited: list[str] = []
    for cell in cells:
        cited += [t for t in _TICKED.findall(cell) if t.startswith("test_")]
    return cited


def test_cases_md_exists_and_has_rows() -> None:
    assert CASES.is_file(), "docs/CASES.md is missing"
    rows = _rows()
    assert len(rows) >= 60, (
        f"docs/CASES.md parsed to only {len(rows)} rows. Either the table "
        "shrank or the parser stopped matching it; both need looking at."
    )


def test_every_case_names_a_real_test() -> None:
    """Every test cited in the case matrix exists in the suite.

    The failure a reader would never catch: a row citing a test that was
    renamed in a refactor. The table still reads as coverage.
    """
    defined = _defined_tests()
    missing: list[tuple[str, str]] = []
    for cells in _rows():
        for name in _cited_tests(cells):
            if name not in defined:
                missing.append((name, cells[0][:70]))
    assert not missing, (
        "docs/CASES.md cites tests that do not exist:\n"
        + "\n".join(f"  {name}  (row: {case})" for name, case in missing)
        + "\n\nEither the test was renamed and the table needs the new name, or "
        "the case is not actually covered and the row is a claim with nothing "
        "behind it."
    )


def test_every_case_row_names_a_test() -> None:
    """No row may be silent about its coverage.

    A row with an empty test column is a case someone meant to cover. Leaving
    it blank is how it stops being noticed.
    """
    uncovered = [
        cells[0][:70]
        for cells in _rows()
        if not _cited_tests(cells)
    ]
    assert not uncovered, (
        "rows in docs/CASES.md name no covering test:\n"
        + "\n".join(f"  {case}" for case in uncovered)
    )


def test_every_artifact_the_matrix_cites_exists() -> None:
    """A row pointing at a results file that is not there is a dead citation.

    Only paths that look like repository paths are checked; prose in backticks
    (`status_reason`, detector ids, config keys) is left alone.
    """
    prefixes = ("results/", "evalsets/", "policies/", "notebooks/", "docs/")
    missing: list[tuple[str, str]] = []
    for cells in _rows():
        for cell in cells:
            for ticked in _TICKED.findall(cell):
                if not ticked.startswith(prefixes):
                    continue
                if not (PROJECT_ROOT / ticked).exists():
                    missing.append((ticked, cells[0][:70]))
    assert not missing, (
        "docs/CASES.md cites artifacts that do not exist:\n"
        + "\n".join(f"  {path}  (row: {case})" for path, case in missing)
    )


@pytest.mark.parametrize(
    "heading",
    [
        "## 1. Warrant states",
        "## 2. Composition",
        "## 3. The three policy profiles",
        "## 4. The tier curve",
        "## 5. Detector refusals",
        "## 6. Drift, revocation, downgrade, refusal",
        "## 7. Controls",
        "## 8. Guard rejections",
    ],
)
def test_the_matrix_still_covers_every_required_area(heading: str) -> None:
    """E.5 enumerates the areas the matrix must cover at minimum.

    Parametrised so a deletion names the section that went missing rather than
    failing on a single opaque assertion.
    """
    text = CASES.read_text(encoding="utf-8")
    assert heading in text, (
        f"docs/CASES.md no longer has a section starting {heading!r}. E.5 "
        "requires this area to be enumerated."
    )
