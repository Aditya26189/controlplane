"""Classify the last_token result against the branches pre-registered in 065.

The decision rule is executed rather than composed after the number is visible.
That is the whole value of the pre-registration: a rule written down and then
applied by hand is a rule with a thumb available to it, and this is the case
that most invites one — a public handover asserts 0.1416 and Round 2 measured
0.0794.

The bounds below are Round 1's published bootstrap CI, copied from its
``test_scoring_log.json``. They are constants here, not arguments, so a run
cannot widen its own acceptance region.

Usage:
    python scripts/06_reconcile.py --config config.yaml \
        --cache results/cache-triviaqa-600.npz --eval-set triviaqa-600
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, provenance, set_seeds, setup_logging, write_json_artifact
from src.detectors.probe import LinearProbe
from src.evalsets.registry import load_evalset
from src.validation.evalsets import (
    TEST,
    TRAIN,
    VALIDATION,
    ExtractionCache,
    split_by_question,
)
from src.validation.stats import auroc

_LOG = logging.getLogger("scripts.06_reconcile")

#: Round 1's published test AUROC and its 95% bootstrap CI, from the Round 1
#: results bundle. Fixed constants: branch A is "inside Round 1's interval",
#: and an interval a later run could adjust is not a pre-registration.
ROUND1_AUROC = 0.8551414437908573
ROUND1_CI = (0.8216804377990431, 0.8878182998424411)
ROUND1_RECALL = 0.14163090128755365
ROUND1_FLAG_RATE = 0.061666666666666667
ROUND1_BASE_RATE = 0.3883333333333333

#: The pooled band, for branch B. Round 2's measured pooled AUROC was 0.7853 and
#: 0.7855; "near 0.785" is read as within 0.02 of it.
POOLED_AUROC = 0.7854
POOLED_TOLERANCE = 0.02

#: Declared in 065, before any of this was measured. Not chosen from the results.
DECLARED_VARIANT = "T1-last_token"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--eval-set", default="triviaqa-600")
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def _fit_and_score(config, cache, matrix, train, validation):
    """Select C on validation, exactly as validate() does, then score."""
    best_c, best = None, -1.0
    for candidate in config.probe.C_grid:
        trial = LinearProbe(
            candidate,
            class_weight=config.probe.class_weight,
            standardize=config.probe.standardize,
            seed=config.seed,
        ).fit(matrix, cache.labels, train)
        value = auroc(cache.labels[validation], trial.score(matrix)[validation])
        if value > best:
            best_c, best = candidate, value
    probe = LinearProbe(
        best_c,
        class_weight=config.probe.class_weight,
        standardize=config.probe.standardize,
        seed=config.seed,
    ).fit(matrix, cache.labels, train)
    return probe, probe.score(matrix), best_c, best


def classify(last_token_auroc: float) -> tuple[str, str]:
    """Which pre-registered branch the number falls into. No judgement here."""
    if ROUND1_CI[0] <= last_token_auroc <= ROUND1_CI[1]:
        return "A", (
            "inside Round 1's CI [%.4f, %.4f]: the aggregation attribution in "
            "062 is CONFIRMED, Round 1 reproduces on Round 2's pipeline, and "
            "the suspended ceiling framing becomes sayable."
            % ROUND1_CI
        )
    if abs(last_token_auroc - POOLED_AUROC) <= POOLED_TOLERANCE:
        return "B", (
            "within %.2f of the pooled variants (%.4f): the aggregation "
            "attribution is REFUTED. The gap is something else and becomes the "
            "next investigation ahead of Phase 5 -- labelling rules, split "
            "derivation, sample draw, or model revision/quantisation, and "
            "nothing else until those are eliminated."
            % (POOLED_TOLERANCE, POOLED_AUROC)
        )
    return "C", (
        "neither branch: outside Round 1's CI and not with the pooled "
        "variants. Gets its own DECISIONS entry before anything is claimed."
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)

    evalset = load_evalset(PROJECT_ROOT / "evalsets" / f"{args.eval_set}.json")
    cache = ExtractionCache.load(args.cache)
    if cache.eval_set_hash != evalset.content_hash:
        raise SystemExit(
            f"cache/eval-set mismatch: {cache.eval_set_hash} vs "
            f"{evalset.content_hash}"
        )
    if DECLARED_VARIANT not in cache.variants:
        raise SystemExit(
            "%s is not in this cache (%s). 065 pre-registered all three "
            "aggregations in one session; a cache without last_token cannot "
            "settle the comparison." % (DECLARED_VARIANT, sorted(cache.variants))
        )

    splits = split_by_question(evalset, seed=config.seed)
    train, validation, test = (np.asarray(splits[k]) for k in (TRAIN, VALIDATION, TEST))
    base_rate = float(cache.labels[test].mean())

    rows = {}
    for variant in sorted(cache.variants):
        matrix = cache.matrix(variant)
        _, scores, chosen_c, val_auroc = _fit_and_score(
            config, cache, matrix, train, validation
        )
        test_auroc = auroc(cache.labels[test], scores[test])
        # Re-thresholded to Round 1's measured flag rate, so recall is compared
        # at a matched budget rather than across two different operating points.
        threshold = float(np.quantile(scores[validation], 1 - ROUND1_FLAG_RATE))
        flagged = scores[test] >= threshold
        measured_f = float(flagged.mean())
        positives = int((cache.labels[test] == 1).sum())
        tp = int((flagged & (cache.labels[test] == 1)).sum())
        rows[variant] = {
            "C": chosen_c,
            "val_auroc": val_auroc,
            "test_auroc": test_auroc,
            "flag_rate_at_round1_budget": measured_f,
            "recall_at_round1_budget": tp / positives if positives else None,
            "precision_at_round1_budget": tp / flagged.sum() if flagged.sum() else None,
            "lift_at_round1_budget": (
                (tp / positives) / measured_f if positives and measured_f else None
            ),
        }

    branch, meaning = classify(rows[DECLARED_VARIANT]["test_auroc"])

    print()
    print("Reconciliation against Round 1 (DECISIONS 065, pre-registered)")
    print("  Round 1: AUROC %.4f %s, recall %.4f at f %.4f, base %.4f"
          % (ROUND1_AUROC, list(ROUND1_CI), ROUND1_RECALL, ROUND1_FLAG_RATE,
             ROUND1_BASE_RATE))
    print("  Round 2 base rate on test: %.4f" % base_rate)
    print()
    header = "%-24s %-6s %-11s %-11s %-9s %s" % (
        "variant", "C", "test AUROC", "recall@R1f", "lift", "declared")
    print(header)
    print("-" * len(header))
    for variant, row in rows.items():
        print("%-24s %-6g %-11.4f %-11.4f %-9.3f %s" % (
            variant, row["C"], row["test_auroc"],
            row["recall_at_round1_budget"] or float("nan"),
            row["lift_at_round1_budget"] or float("nan"),
            "<- Beat 4" if variant == DECLARED_VARIANT else ""))
    print()
    print("BRANCH %s - %s" % (branch, meaning))
    print()

    out = Path(args.out) if args.out else PROJECT_ROOT / config.paths.results_dir
    out.mkdir(parents=True, exist_ok=True)
    write_json_artifact(
        out / "reconciliation.json",
        {
            "round1": {
                "auroc": ROUND1_AUROC, "ci": list(ROUND1_CI),
                "recall": ROUND1_RECALL, "flag_rate": ROUND1_FLAG_RATE,
                "base_rate": ROUND1_BASE_RATE,
            },
            "round2": rows,
            "round2_base_rate": base_rate,
            "declared_variant": DECLARED_VARIANT,
            "branch": branch,
            "branch_meaning": meaning,
            "preregistered_in": "DECISIONS.md 065",
        },
        config,
    )
    _LOG.info("branch %s written to %s", branch, out / "reconciliation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
