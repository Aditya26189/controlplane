"""Freeze a canary set for the activation probe, from the TRAIN split only.

The canary is a **regression tripwire, not a measurement**. It is a small set of
known-incorrect items the probe must catch at its frozen threshold; anything
less than perfect recall means something changed between runs — a moved
threshold, a reordered dataset, a silently swapped model.

**Why one is needed.** ``canary_control`` fails *closed* when no canary is
available: absent is refused rather than skipped. Until now the only canary in
the repo was ``canary-20-pii``, which belongs to Presidio, so every measured
activation-tier warrant was refused on that control alone — a refusal that said
nothing about the probe and drowned the one that did (``DECISIONS.md`` 059).

**Train split only.** Selecting canary items on validation or test would be
selection on the data those splits exist to protect, and every downstream number
would inherit it.

**What this does and does not prove.** Items are chosen by the current probe's
own score, so the current probe catches them by construction. That is
circular as a measurement and exactly right as a tripwire: it detects *change*,
not quality. A canary that a future probe misses means the pipeline moved, which
is the only thing this control claims to detect. It is not independent evidence
that the probe works.

No GPU: the activations already exist in the extraction cache, and the canary
cache is a row slice of it.

Usage:
    python scripts/05_canary.py --config config.yaml \
        --cache results/measured/cache-triviaqa-600.npz \
        --eval-set triviaqa-600 --variant T1-mean_pool
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

from controlplane.config import load_config, set_seeds, setup_logging
from controlplane.detectors.probe import LinearProbe
from controlplane.evalsets.registry import load_evalset, save_evalset
from controlplane.validation.evalsets import (
    TRAIN,
    VALIDATION,
    EvalSet,
    ExtractionCache,
    split_by_question,
)
from controlplane.validation.stats import auroc as roc_auc

_LOG = logging.getLogger("scripts.05_canary")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--eval-set", default="triviaqa-600")
    parser.add_argument("--variant", default="T1-mean_pool")
    parser.add_argument("--n-items", type=int, default=20)
    parser.add_argument("--target-flag-rate", type=float, default=0.05)
    parser.add_argument("--canary-id", default="canary-20-triviaqa")
    parser.add_argument("--evalsets-out", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)

    evalset = load_evalset(PROJECT_ROOT / "evalsets" / f"{args.eval_set}.json")
    cache = ExtractionCache.load(args.cache)
    if cache.eval_set_hash != evalset.content_hash:
        raise SystemExit(
            f"cache/eval-set mismatch: cache records {cache.eval_set_hash}, "
            f"{args.eval_set} hashes to {evalset.content_hash}"
        )

    splits = split_by_question(evalset, seed=config.seed)
    train = np.asarray(splits[TRAIN])
    validation_idx = np.asarray(splits[VALIDATION])

    # A canary must be catchable by EVERY variant in the ladder, not just the
    # declared one. Items chosen by one aggregation's score are not necessarily
    # in another's top band: a canary built on last_token alone was caught 20/20
    # by last_token and 15/20 by mean_pool, refusing two of three ladder rungs
    # for a reason that had nothing to do with their quality. "Unambiguous"
    # means unambiguous to all of them.
    scores_by_variant, thresholds, chosen_c, val_auroc = {}, {}, {}, {}
    for variant in sorted(cache.variants):
        matrix = cache.matrix(variant)
        best_c, best_auroc = None, -1.0
        for candidate in config.probe.C_grid:
            trial = LinearProbe(
                candidate,
                class_weight=config.probe.class_weight,
                standardize=config.probe.standardize,
                seed=config.seed,
            ).fit(matrix, cache.labels, train)
            value = roc_auc(
                cache.labels[validation_idx], trial.score(matrix)[validation_idx]
            )
            if value > best_auroc:
                best_c, best_auroc = candidate, value
        probe = LinearProbe(
            best_c,
            class_weight=config.probe.class_weight,
            standardize=config.probe.standardize,
            seed=config.seed,
        ).fit(matrix, cache.labels, train)
        scores = probe.score(matrix)
        scores_by_variant[variant] = scores
        thresholds[variant] = float(
            np.quantile(scores[validation_idx], 1.0 - args.target_flag_rate)
        )
        chosen_c[variant] = best_c
        val_auroc[variant] = best_auroc
        _LOG.info(
            "%s: C=%g on validation (AUROC %.4f), threshold %.4f",
            variant, best_c, best_auroc, thresholds[variant],
        )

    # A variant that cannot clear the AUROC floor is refused on merit whatever
    # the canary says, so letting it constrain the canary lets a near-chance
    # detector break the tripwire for the ones that work. Excluded from the
    # intersection, loudly. This is a stated rule, not a relaxation chosen to
    # make a case pass: the excluded variant's warrant outcome is unchanged.
    floor = config.validation.min_auroc_lower_ci
    binding = [v for v in sorted(cache.variants) if val_auroc[v] > floor]
    excluded = [v for v in sorted(cache.variants) if v not in binding]
    for variant in excluded:
        _LOG.warning(
            "%s: validation AUROC %.4f does not clear the %.2f floor, so it "
            "cannot hold a warrant regardless of the canary. Excluded from the "
            "canary constraint; it would otherwise veto items every working "
            "variant catches.",
            variant, val_auroc[variant], floor,
        )
    if not binding:
        raise SystemExit(
            "no variant clears the %.2f AUROC floor on validation, so no "
            "canary can be meaningful: every warrant is refused on merit."
            % floor
        )

    incorrect = train[cache.labels[train] == 1]
    if len(incorrect) < args.n_items:
        raise SystemExit(
            f"only {len(incorrect)} incorrect items in train; need {args.n_items}"
        )

    # Eligible = clears every variant's threshold. Ranked by the WORST margin
    # across variants, so the ones kept are the ones no variant finds marginal.
    margins = np.stack([
        scores_by_variant[v][incorrect] - thresholds[v] for v in binding
    ])
    worst = margins.min(axis=0)
    eligible = incorrect[worst > 0]
    if len(eligible) < args.n_items:
        raise SystemExit(
            "only %d of %d incorrect train items clear every binding "
            "variant's threshold (%s); need %d. A canary that one rung of the "
            "ladder cannot catch refuses that rung on every run for a reason "
            "unrelated to it."
            % (len(eligible), len(incorrect), ", ".join(binding), args.n_items)
        )
    order = np.argsort(-worst[worst > 0])[: args.n_items]
    chosen = np.sort(eligible[order])

    for variant in binding:
        caught = int(
            (scores_by_variant[variant][chosen] >= thresholds[variant]).sum()
        )
        if caught != len(chosen):
            raise SystemExit(
                "%s catches only %d/%d chosen items. Refusing to freeze a "
                "tripwire that is already tripped for one rung."
                % (variant, caught, len(chosen))
            )
    _LOG.info(
        "all %d items clear all %d binding variants (%s); worst margin %.4f",
        len(chosen), len(binding), ", ".join(binding),
        float(worst[worst > 0][order].min()),
    )

    items = tuple(evalset.items[int(i)] for i in chosen)
    canary = EvalSet(
        eval_set_id=args.canary_id,
        items=items,
        data_source=cache.data_source,
        construction={
            "derived_from": evalset.eval_set_id,
            "derived_from_hash": evalset.content_hash,
            "split_used": "train",
            "selected_by": "worst-margin-across-all-variants, all positive",
            "variants_required_to_catch": binding,
            "variants_excluded_below_auroc_floor": excluded,
            "C_per_variant": chosen_c,
            "thresholds_cleared": thresholds,
            "n_items": int(args.n_items),
            "purpose": (
                "regression tripwire: recall must be 1.0 at the frozen "
                "threshold. Chosen by the probe's own score, so it detects "
                "change rather than quality, and is not independent evidence "
                "that the probe works."
            ),
        },
    )

    canary_cache = ExtractionCache(
        eval_set_id=canary.eval_set_id,
        eval_set_hash=canary.content_hash,
        model_name=cache.model_name,
        layer=cache.layer,
        data_source=cache.data_source,
        features={name: values[chosen] for name, values in cache.features.items()},
        labels=cache.labels[chosen],
        question_ids=cache.question_ids[chosen],
        token_lengths=cache.token_lengths[chosen],
        # Padding evidence belongs to the extraction that produced it and
        # describes 2400 rows, not these 20. Carrying it here would attach a
        # control's evidence to a set it was not captured on.
        padding_evidence=None,
        extra={**cache.extra, "derived_from": evalset.eval_set_id},
    )

    registry = Path(args.evalsets_out) if args.evalsets_out else PROJECT_ROOT / "evalsets"
    out = Path(args.out) if args.out else Path(args.cache).parent
    evalset_path = save_evalset(canary, registry)
    cache_path = canary_cache.save(out / f"cache-{canary.eval_set_id}.npz")

    _LOG.info(
        "canary %s: %d items from %s train, all label=1, hash %s",
        canary.eval_set_id,
        len(canary),
        evalset.eval_set_id,
        canary.content_hash,
    )
    _LOG.info("score range of chosen items: %.4f to %.4f",
              float(scores[chosen].min()), float(scores[chosen].max()))
    print(evalset_path)
    print(cache_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
