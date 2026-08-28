"""The "prove it" button. Block E, E.4 and E.6.

Two checks, in order of strength:

1. **Every number in the README claim table matches its artifact.** Fast, no
   dependencies beyond the repository, always runs.
2. **The artifacts re-derive from cached activations.** Proves the committed
   evidence still describes what the code does, which check 1 cannot: both
   could be stale together and check 1 would still pass. Needs the extraction
   cache, which is gitignored, so it reports SKIPPED rather than passing when
   the cache is absent.

Thin wrapper: parses arguments, calls ``controlplane/``, prints. No logic
(``CLAUDE.md``). Exits non-zero if any number drifts.

Usage:
    python scripts/verify.py
    python scripts/verify.py --claims-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from controlplane.report.claims import check_claims  # noqa: E402
from controlplane.report.claims import render as render_claims  # noqa: E402
from controlplane.report.reproduce import reproduce_frozen_set  # noqa: E402
from controlplane.report.reproduce import render as render_reproduction  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", default=str(PROJECT_ROOT / "README.md"))
    parser.add_argument("--eval-set", default="triviaqa-600")
    parser.add_argument(
        "--claims-only",
        action="store_true",
        help="skip the re-derivation even when the cache is present",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("1. README claim table vs committed artifacts")
    print("=" * 78)
    claims = check_claims(PROJECT_ROOT, Path(args.readme))
    print(render_claims(claims))
    claims_ok = all(c.ok for c in claims)

    print()
    print("=" * 78)
    print("2. Committed artifacts vs a re-run from cached activations")
    print("=" * 78)
    if args.claims_only:
        print("re-derivation SKIPPED\n  --claims-only was passed")
        reproduction_ok = True
        skipped = True
    else:
        report = reproduce_frozen_set(PROJECT_ROOT, eval_set=args.eval_set)
        print(render_reproduction(report))
        reproduction_ok = report.ok
        skipped = not report.ran

    print()
    print("=" * 78)
    if claims_ok and reproduction_ok:
        note = " (re-derivation skipped)" if skipped else ""
        print(f"VERIFIED{note}")
        return 0
    print("FAILED", file=sys.stderr)
    print(
        "A number does not reproduce. Either the pipeline changed and the "
        "artifacts need regenerating, or a number was edited by hand. Never "
        "edit the number to match.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
