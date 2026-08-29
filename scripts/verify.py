"""The "prove it" button. Block E, E.4, E.6 and E.8.

Three checks, weakest first, each proving something the one before it cannot.

1. **Every number in the README claim table matches its artifact.** Resolves
   each field and compares at the quoted precision. Needs nothing but the
   repository. Cannot detect a README and a set of artifacts that are stale
   *together*.

2. **Every metrics block recomputes from the frozen per-item scores.** Same
   builder, same bootstrap count, same seed. Catches exactly what (1) cannot:
   an artifact whose numbers no longer follow from the data behind them. The
   scores are ~200 KB and committed, so **this runs on a fresh clone** -- which
   is the whole reason they are committed. Cannot prove the scores came from
   the model and probe the artifact names.

3. **The frozen scores re-derive from cached activations.** The deepest tier,
   and the only one that closes the loop back to the model. The caches are
   ~100 MB and gitignored, so on a fresh clone this reports SKIPPED rather
   than passing.

The tiering is the honest form of "verified". Each check states what it
covers, a skipped check is never rendered as a pass, and the final line names
any tier that did not run.

Thin wrapper: parses arguments, calls ``controlplane/``, prints. No logic
(``CLAUDE.md``). Exits non-zero if any number drifts.

Usage:
    python scripts/verify.py                  # all three where possible
    python scripts/verify.py --no-activations # what a fresh clone runs
    python scripts/verify.py --claims-only    # the table alone
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from controlplane.report.claims import check_claims  # noqa: E402
from controlplane.report.claims import render as render_claims  # noqa: E402
from controlplane.report.reproduce import (  # noqa: E402
    reproduce_frozen_set,
    reproduce_from_scores,
)
from controlplane.report.reproduce import render as render_reproduction  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", default=str(PROJECT_ROOT / "README.md"))
    parser.add_argument("--eval-set", default="triviaqa-600")
    parser.add_argument(
        "--claims-only",
        action="store_true",
        help="check the claim table only; skip both re-derivation tiers",
    )
    parser.add_argument(
        "--no-activations",
        action="store_true",
        help=(
            "skip the activation tier even where the cache is present. The "
            "score tier still runs, which is what a fresh clone does anyway."
        ),
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
    print("2. Committed metrics vs a recomputation from frozen scores")
    print("=" * 78)
    print(
        "Recomputes every metrics block from results/scores/ -- the per-item\n"
        "labels, scores and question ids -- with the same builder, bootstrap\n"
        "count and seed. Committed and ~200 KB, so this runs on a fresh clone.\n"
    )
    if args.claims_only:
        print("re-derivation SKIPPED\n  --claims-only was passed")
        scores_ok, scores_skipped = True, True
    else:
        score_report = reproduce_from_scores(PROJECT_ROOT)
        print(render_reproduction(score_report))
        scores_ok, scores_skipped = score_report.ok, not score_report.ran

    print()
    print("=" * 78)
    print("3. Frozen scores vs a re-run from cached activations")
    print("=" * 78)
    print(
        "The deeper tier. Check 2 proves the metrics follow from the recorded\n"
        "scores; it cannot prove those scores came from the model and probe the\n"
        "artifact names. This re-runs the extraction-tier validation and\n"
        "compares. The caches are ~100 MB and gitignored, so on a fresh clone\n"
        "this reports SKIPPED -- never a pass it did not earn.\n"
    )
    if args.claims_only or args.no_activations:
        flag = "--claims-only" if args.claims_only else "--no-activations"
        print(f"re-derivation SKIPPED\n  {flag} was passed")
        activations_ok, activations_skipped = True, True
    else:
        act_report = reproduce_frozen_set(PROJECT_ROOT, eval_set=args.eval_set)
        print(render_reproduction(act_report))
        activations_ok, activations_skipped = act_report.ok, not act_report.ran

    print()
    print("=" * 78)
    if claims_ok and scores_ok and activations_ok:
        skipped = [
            name
            for name, was in (("scores", scores_skipped), ("activations", activations_skipped))
            if was
        ]
        note = f" ({' and '.join(skipped)} tier skipped)" if skipped else ""
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
