"""Parse the README claim table and check every number against its artifact.

Block E, E.6. A number hand-edited into the README and drifting from the file
that produced it is the same failure as an uncertified response that looks
certified: the reader cannot tell, and nothing in the repository objects. This
module makes it impossible rather than a matter of discipline.

The claim table in ``README.md`` is the source. Each row names the artifact, a
resolver path into it, and the value to the precision it is quoted at. This
resolves the path, compares at that precision, and reports every row.

**The precision is the tolerance.** A claim written ``0.8256`` is checked to
four decimals; one written ``0.83`` is checked to two. Quoting a number less
precisely is a weaker claim and is checked as one -- which is correct, and is
also why the table quotes what the artifacts actually say.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "Claim",
    "ClaimResult",
    "check_claims",
    "parse_claim_table",
    "resolve",
]


# --------------------------------------------------------------------------- #
# The resolver
# --------------------------------------------------------------------------- #

#: ``runs[detector_id=presidio-stock,eval_set_id=hinglish-pii-200]`` --
#: a list segment plus the key=value pairs that select one element of it.
_SELECTOR = re.compile(r"^([A-Za-z0-9_-]+)\[(.+)\]$")


def _select(items: list, spec: str, path: str) -> Any:
    """Pick the one element of ``items`` matching every ``key=value`` in spec.

    Refuses ambiguity: zero matches and two matches are both errors. A selector
    that silently took the first match would make a claim about whichever row
    happened to be written first.
    """
    wanted = []
    for pair in spec.split(","):
        key, _, value = pair.partition("=")
        wanted.append((key.strip(), value.strip()))

    def matches(item: Any) -> bool:
        for key, value in wanted:
            found = item
            for part in key.split("."):
                if not isinstance(found, dict) or part not in found:
                    return False
                found = found[part]
            if str(found) != value:
                return False
        return True

    hits = [i for i in items if matches(i)]
    if len(hits) != 1:
        raise KeyError(
            f"selector [{spec}] matched {len(hits)} elements in {path!r}; "
            "exactly one is required"
        )
    return hits[0]


def _segments(path: str) -> list[str]:
    """Split a path on dots that are outside ``[...]``.

    A selector key can itself be dotted -- ``[operating_point.operating_point_id=P-x]``
    -- so a plain ``path.split(".")`` tears the selector in half. It did, on the
    first path written against this resolver.
    """
    parts: list[str] = []
    depth = 0
    current = ""
    for char in path:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth < 0:
                raise KeyError(f"unbalanced ] in path {path!r}")
        if char == "." and depth == 0:
            parts.append(current)
            current = ""
            continue
        current += char
    if depth != 0:
        raise KeyError(f"unbalanced [ in path {path!r}")
    parts.append(current)
    return [p for p in parts if p]


def resolve(document: Any, path: str) -> Any:
    """Resolve a dotted path with optional list selectors.

    Supports ``a.b.c``, ``a[k=v].b`` and ``a[k=v,j=w].b``. Raises rather than
    returning None for a missing key: a claim pointing at a field that is not
    there must fail loudly, since that is exactly the drift being caught.
    """
    current = document
    for segment in _segments(path):
        match = _SELECTOR.match(segment)
        if match:
            name, spec = match.group(1), match.group(2)
            if not isinstance(current, dict) or name not in current:
                raise KeyError(f"{name!r} not found while resolving {path!r}")
            listing = current[name]
            if not isinstance(listing, list):
                raise KeyError(f"{name!r} is not a list, resolving {path!r}")
            current = _select(listing, spec, path)
            continue
        if isinstance(current, list):
            raise KeyError(
                f"{segment!r} indexes a list without a selector, in {path!r}"
            )
        if not isinstance(current, dict) or segment not in current:
            available = sorted(current)[:8] if isinstance(current, dict) else "n/a"
            raise KeyError(
                f"{segment!r} not found while resolving {path!r}; "
                f"available keys: {available}"
            )
        current = current[segment]
    return current


# --------------------------------------------------------------------------- #
# Claims
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Claim:
    """One row of the README claim table."""

    label: str
    value: str
    interval: str
    artifact: str
    field: str
    command: str

    @property
    def decimals(self) -> int:
        """Digits after the point in the quoted value -- the checking tolerance."""
        _, _, frac = self.value.partition(".")
        return len(frac)


@dataclass(frozen=True)
class ClaimResult:
    """The outcome of checking one claim."""

    claim: Claim
    ok: bool
    measured: Optional[str]
    detail: str

    @property
    def status(self) -> str:
        return "OK" if self.ok else "DRIFT"


#: A README table row. Six cells, pipe-delimited.
_ROW = re.compile(r"^\|(?!\s*[-: ]+\|)(.+)\|\s*$")
#: Strip markdown emphasis and backticks from a cell.
_CLEAN = re.compile(r"[`*]")
#: An interval, as ``[lo, hi]``.
_INTERVAL = re.compile(r"^\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]$")


def parse_claim_table(readme: Path) -> list[Claim]:
    """Extract the claim table from README.md.

    The table is found by its header rather than by position, so adding prose
    above it does not silently stop the check from finding anything -- an empty
    result would otherwise read as "every claim passes".
    """
    lines = readme.read_text(encoding="utf-8").splitlines()
    claims: list[Claim] = []
    inside = False
    for line in lines:
        stripped = line.strip()
        match = _ROW.match(stripped)
        if not match:
            if inside and stripped and not stripped.startswith("|"):
                inside = False
            continue
        cells = [_CLEAN.sub("", c).strip() for c in match.group(1).split("|")]
        if len(cells) < 6:
            continue
        header = [c.lower() for c in cells[:6]]
        if header[0] == "claim" and "artifact" in header:
            inside = True
            continue
        if not inside:
            continue
        claims.append(
            Claim(
                label=cells[0],
                value=cells[1],
                interval=cells[2],
                artifact=cells[3],
                field=cells[4],
                command=cells[5],
            )
        )
    return claims


def _quantise(value: float, decimals: int) -> str:
    """Round half-even to ``decimals`` places, as a string for exact comparison."""
    return str(Decimal(repr(round(value, decimals))).quantize(
        Decimal(1).scaleb(-decimals) if decimals else Decimal(1)
    ))


def _check_one(claim: Claim, root: Path) -> ClaimResult:
    artifact = root / claim.artifact
    if not artifact.is_file():
        return ClaimResult(claim, False, None, f"artifact not found: {claim.artifact}")
    try:
        document = json.loads(artifact.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return ClaimResult(claim, False, None, f"artifact is not valid JSON: {exc}")

    try:
        found = resolve(document, claim.field)
    except KeyError as exc:
        return ClaimResult(claim, False, None, str(exc))

    # Non-numeric claims -- a status, a branch letter, a count as text.
    try:
        stated = float(claim.value)
    except ValueError:
        ok = str(found) == claim.value
        return ClaimResult(
            claim, ok, str(found),
            "matches" if ok else f"README says {claim.value!r}, artifact says {found!r}",
        )

    if not isinstance(found, (int, float)):
        return ClaimResult(
            claim, False, str(found),
            f"README quotes a number but the field holds {type(found).__name__}",
        )

    decimals = claim.decimals
    measured = _quantise(float(found), decimals)
    expected = _quantise(stated, decimals)
    if measured != expected:
        return ClaimResult(
            claim, False, measured,
            f"README says {expected} at {decimals}dp, artifact gives {measured} "
            f"(raw {found!r})",
        )

    # The interval, when one is quoted, comes from the sibling bounds.
    if claim.interval and claim.interval not in {"-", "—", "n/a"}:
        bounds = _INTERVAL.match(claim.interval)
        if not bounds:
            return ClaimResult(
                claim, False, measured,
                f"interval {claim.interval!r} is not of the form [lo, hi]",
            )
        parent_path = ".".join(_segments(claim.field)[:-1])
        try:
            parent = resolve(document, parent_path)
        except KeyError as exc:
            return ClaimResult(claim, False, measured, f"interval: {exc}")
        for name, quoted in (("ci_low", bounds.group(1)), ("ci_high", bounds.group(2))):
            if not isinstance(parent, dict) or parent.get(name) is None:
                return ClaimResult(
                    claim, False, measured,
                    f"README quotes an interval but {parent_path}.{name} is absent",
                )
            places = len(quoted.partition(".")[2])
            got = _quantise(float(parent[name]), places)
            if got != _quantise(float(quoted), places):
                return ClaimResult(
                    claim, False, measured,
                    f"interval {name}: README says {quoted}, artifact gives {got}",
                )

    return ClaimResult(claim, True, measured, "matches")


def check_claims(root: Path, readme: Optional[Path] = None) -> list[ClaimResult]:
    """Check every claim in the README table against its artifact.

    Args:
        root: Project root; artifact paths resolve against it.
        readme: The README to parse. Defaults to ``root / "README.md"``.

    Returns:
        One result per claim, in table order.

    Raises:
        AssertionError: If the table is missing or empty. An empty table would
            otherwise pass trivially, which is the failure this exists to stop.
    """
    readme = readme or (root / "README.md")
    assert readme.is_file(), f"{readme} does not exist"
    claims = parse_claim_table(readme)
    assert claims, (
        f"no claim table found in {readme.name}. Expected a markdown table whose "
        "first column header is 'Claim' and which has an 'Artifact' column. An "
        "empty table passes every check, so this is an error rather than a pass."
    )
    return [_check_one(c, root) for c in claims]


def render(results: list[ClaimResult]) -> str:
    """A fixed-width report: committed value beside measured value, per row."""
    width = max((len(r.claim.label) for r in results), default=10)
    width = min(width, 52)
    lines = [
        f"{'claim'.ljust(width)}  {'committed':>11}  {'measured':>11}  status",
        f"{'-' * width}  {'-' * 11}  {'-' * 11}  ------",
    ]
    for r in results:
        lines.append(
            f"{r.claim.label[:width].ljust(width)}  {r.claim.value:>11}  "
            f"{(r.measured or '-'):>11}  {r.status}"
        )
    failed = [r for r in results if not r.ok]
    lines.append("")
    lines.append(f"{len(results) - len(failed)}/{len(results)} claims reproduce.")
    if failed:
        lines.append("")
        lines.append("DRIFT:")
        for r in failed:
            lines.append(f"  {r.claim.label}")
            lines.append(f"    artifact: {r.claim.artifact}")
            lines.append(f"    field:    {r.claim.field}")
            lines.append(f"    {r.detail}")
    return "\n".join(lines)
