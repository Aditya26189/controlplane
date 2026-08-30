"""The documentation index cannot drift from the directory it indexes.

Same argument as ``test_cases_matrix``: a document that lists other documents
is a document that goes stale silently. A page added to ``docs/`` and never
indexed is a page nobody finds, and a link that stops resolving reads as a
missing document rather than as a typo.

Reading is not a control, so these are tests rather than a convention.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS = PROJECT_ROOT / "docs"
INDEX = DOCS / "README.md"

#: Markdown inline links, capturing the target. Reference-style links are not
#: used in this repository; if they ever are, this pattern needs widening.
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")

#: Every markdown file that carries prose a reader is meant to follow. The
#: index itself is excluded -- it does not need to list itself.
INDEXED_ROOTS = (PROJECT_ROOT / "README.md", PROJECT_ROOT / "CLAUDE.md")


def _docs_pages() -> list[Path]:
    return sorted(p for p in DOCS.glob("*.md") if p.name != "README.md")


def _link_targets(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [
        target
        for target in LINK.findall(text)
        if not target.startswith(("http://", "https://", "mailto:", "#"))
    ]


def test_the_index_exists() -> None:
    assert INDEX.is_file(), "docs/README.md is the index and is missing"


@pytest.mark.parametrize("page", _docs_pages(), ids=lambda p: p.name)
def test_every_docs_page_is_indexed(page: Path) -> None:
    """A page nobody links to is a page nobody reads.

    The index is how a reader finds the document that answers their question;
    an unindexed page is discoverable only by listing the directory, which is
    exactly the failure the index exists to prevent.
    """
    index_text = INDEX.read_text(encoding="utf-8")
    assert page.name in index_text, (
        f"docs/{page.name} exists but docs/README.md does not mention it. "
        "Add a row naming what it answers, or delete the page."
    )


@pytest.mark.parametrize(
    "source",
    [*INDEXED_ROOTS, INDEX, *_docs_pages()],
    ids=lambda p: str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
)
def test_every_relative_link_resolves(source: Path) -> None:
    """A dead link in a document is a missing document, to the reader.

    This catches the failure the 2026-08-29 reorganisation could have caused
    and did not: a file moves, nothing errors, and every document pointing at
    its old path silently becomes wrong (``docs/PATHS.md``).
    """
    broken = []
    for target in _link_targets(source):
        resolved = (source.parent / target.split("#")[0]).resolve()
        if not resolved.exists():
            broken.append(target)
    assert not broken, (
        f"{source.relative_to(PROJECT_ROOT)} links to paths that do not exist:\n  "
        + "\n  ".join(broken)
    )


def test_the_index_routes_by_need() -> None:
    """The index is a router, not a directory listing.

    Its value is the first table -- "you are X, read these in this order".
    A plain alphabetical list of filenames would pass every other test here
    while being useless to the reader it exists for.
    """
    index_text = INDEX.read_text(encoding="utf-8")
    assert "## By why you are here" in index_text, (
        "docs/README.md lost its routing table. The index exists to answer "
        "'what do I read first', not to list the directory."
    )
