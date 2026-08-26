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

from src.config import load_config, set_seeds, setup_logging
from src.detectors.probe import LinearProbe
from src.evalsets.registry import load_evalset, save_evalset
from src.validation.evalsets import (
    TRAIN,
    VALIDATION,
    EvalSet,
    ExtractionCache,
    split_by_question,
)
from src.validation.stats import auroc as roc_auc

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
    matrix = cache.matrix(args.variant)

    # C is selected on validation, exactly as validate() does. Taking
    # C_grid[0] instead would fit a different probe from the validated one, and
    # a canary chosen by a probe nobody runs is a tripwire for the wrong thing.
    validation = np.asarray(splits[VALIDATION])
    best_c, best_auroc = None, -1.0
    for candidate in config.probe.C_grid:
        trial = LinearProbe(
            candidate,
            class_weight=config.probe.class_weight,
            standardize=config.probe.standardize,
            seed=config.seed,
        ).fit(matrix, cache.labels, train)
        auroc = roc_auc(cache.labels[validation], trial.score(matrix)[validation])
        if auroc > best_auroc:
            best_c, best_auroc = candidate, auroc
    _LOG.info("selected C=%g on validation (AUROC %.4f)", best_c, best_auroc)

    probe = LinearProbe(
        best_c,
        class_weight=config.probe.class_weight,
        standardize=config.probe.standardize,
        seed=config.seed,
    ).fit(matrix, cache.labels, train)
    scores = probe.score(matrix)

    # The operating point the canary must clear, derived the same way the
    # validated run derives it: the flag-rate budget applied to validation.
    threshold = float(
        np.quantile(scores[validation], 1.0 - args.target_flag_rate)
    )

    # Incorrect items only, from train only, ranked by the probe's own score.
    incorrect = train[cache.labels[train] == 1]
    if len(incorrect) < args.n_items:
        raise SystemExit(
            f"only {len(incorrect)} incorrect items in train; need {args.n_items}"
        )
    chosen = incorrect[np.argsort(-scores[incorrect])[: args.n_items]]
    chosen = np.sort(chosen)

    # Refuse to freeze a tripwire that is already tripped. canary_control
    # requires recall == 1.0, so an item below the threshold makes the control
    # fail on every future run for a reason that has nothing to do with the
    # run -- which is the failure this canary exists to end, reintroduced.
    below = chosen[scores[chosen] < threshold]
    if len(below):
        raise SystemExit(
            "%d of %d chosen items score below the operating threshold %.4f "
            "(lowest %.4f). A canary that cannot pass its own control is worse "
            "than none: it refuses every warrant and says nothing."
            % (len(below), len(chosen), threshold, float(scores[chosen].min()))
        )
    _LOG.info(
        "all %d items clear the threshold %.4f (margin %.4f at the closest)",
        len(chosen), threshold, float(scores[chosen].min() - threshold),
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
            "selected_by": f"highest probe score on {args.variant}",
            "C": best_c,
            "threshold_cleared": threshold,
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
