"""Library guards must survive ``python -O``.

``assert`` statements are removed by the optimiser. Every one of them in
library code is therefore a check that exists in development and silently does
not exist in production -- and this repository's guards are the product, not a
convenience. A verifier whose "the claim table is empty" check vanishes under a
flag reports success on an empty table.

This is the third instance of the same class here. ``DECISIONS.md`` records an
earlier one: a finiteness ``assert`` in the monitor, which under ``-O`` would
have let a ``NaN`` wealth value pass as "not above the revocation threshold".
Fixing instances one at a time is how a class of defect survives, so this file
holds the scan that fails on the next one rather than a list of the ones fixed.

Tests, not a convention: reading is not a control.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT_ROOT / "controlplane"


def _package_modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


@pytest.mark.parametrize(
    "module",
    _package_modules(),
    ids=lambda p: str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
)
def test_no_assert_statements_in_library_code(module: Path) -> None:
    """No ``assert`` in ``controlplane/`` -- raise the exception instead.

    Tests may assert; they are never run under ``-O``. Library code may not,
    because the guard disappears exactly where it matters. Keeping the
    ``AssertionError`` type is fine and often correct -- what must change is
    that the raise is a statement the optimiser cannot remove.
    """
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    offenders = [
        f"{module.relative_to(PROJECT_ROOT)}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
    ]
    assert not offenders, (
        "assert statements in library code are stripped by `python -O`, so "
        "these guards do not exist in an optimised run:\n  "
        + "\n  ".join(offenders)
        + "\nRaise the exception explicitly instead: `if not cond: raise "
        "AssertionError(...)`."
    )


def test_the_claim_table_guard_still_fires_under_dash_o(tmp_path: Path) -> None:
    """The end-to-end version of the check above, on the guard that matters most.

    ``check_claims`` refuses a missing or empty claim table, because an empty
    table passes every check it contains. Run under ``-O``, that refusal has to
    be a real raise -- this executes it in a subprocess with the flag actually
    set, rather than trusting that the source reads correctly.
    """
    program = (
        "from pathlib import Path\n"
        "from controlplane.report.claims import check_claims\n"
        "try:\n"
        f"    check_claims(Path(r'{tmp_path}'))\n"
        "except AssertionError as exc:\n"
        "    print('GUARD FIRED:', exc)\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit('GUARD DID NOT FIRE')\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", program],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        "the claim-table guard did not fire under `python -O`:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "GUARD FIRED" in result.stdout
