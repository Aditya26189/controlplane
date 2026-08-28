"""Paired comparison of the 1200-trained and 960-trained probes.

Answers the claim ``DECISIONS.md`` 079 committed to and 080 did not actually
test. Thin wrapper: parses arguments, calls ``src/``, writes files. No logic
(``CLAUDE.md``).

Thresholds are read from the **run artifacts** rather than recomputed, because
recomputing them here would answer a question about this script rather than
about the runs that issued the warrants.

Usage:
    python scripts/08_paired.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, provenance, set_seeds, setup_logging, write_json_artifact
from src.evalsets.registry import load_evalset
from src.evalsets.resplit import cache_source_id
from src.validation.evalsets import TEST, ExtractionCache, split_by_question
from src.report.plots import plot_roc_operating_points
from src.validation.paired import (
    compare_models,
    fit_on,
    fixture_thresholds,
    split_relationship,
)
from src.validation.roc import roc_curve
from src.validation.synthetic import synthetic_cache, synthetic_evalset

_LOG = logging.getLogger("scripts.08_paired")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--baseline-set", default="triviaqa-600")
    parser.add_argument("--variant-set", default="triviaqa-2400-t960")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--fixture",
        action="store_true",
        help=(
            "run on a synthetic cache. Exercises the wiring; produces no result."
        ),
    )
    parser.add_argument("--out", default=None, help="results directory override")
    return parser.parse_args(argv)


def _thresholds(results: Path, baseline_set: str, variant_set: str) -> dict:
    """Read both runs' operating points from their artifacts, at full precision.

    Not from ``config.yaml`` and not by re-deriving them: the question is what
    the runs that issued the warrants actually used, and a value recomputed here
    would agree with itself whether or not it agreed with them.
    """
    def points(path: Path) -> dict:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            run["warrant"]["operating_point"]["operating_point_id"]: float(
                run["warrant"]["operating_point"]["threshold"]
            )
            for run in payload["operating_points"]
        }

    base = points(results / f"policy-{baseline_set}.json")
    var = points(results / f"policy-{variant_set}.json")
    missing = sorted(set(base) ^ set(var))
    if missing:
        raise SystemExit(
            f"the two runs do not share operating points; symmetric difference "
            f"{missing}. A paired comparison needs the same points on both sides."
        )
    return {point: (base[point], var[point]) for point in base}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)

    results = Path(args.out) if args.out else PROJECT_ROOT / config.paths.results_dir
    results.mkdir(parents=True, exist_ok=True)
    caches = PROJECT_ROOT / config.paths.results_dir
    evalsets_dir = PROJECT_ROOT / config.paths.evalsets_dir

    if args.fixture:
        from src.evalsets.resplit import resplit_by_question

        base = synthetic_evalset(
            eval_set_id="paired-fixture-base", n_items=1200, base_rate=0.46,
            seed=config.seed, items_per_question=1, declare_splits=True,
        )
        # Both splits are cut from the SAME permutation at the same seed, which
        # is what makes them nest and therefore what makes a paired comparison
        # possible (DECISIONS 081). Deriving the baseline the same way as the
        # variant is deliberate: a fixture that nested by accident would not
        # exercise the path the measured run takes.
        baseline = resplit_by_question(
            base, eval_set_id="paired-fixture", fractions=(0.5, 0.25, 0.25),
            seed=config.seed, rationale="fixture baseline",
        )
        variant = resplit_by_question(
            base, eval_set_id="paired-fixture-t40", fractions=(0.4, 0.2, 0.4),
            seed=config.seed, rationale="fixture variant",
        )
        cache = synthetic_cache(
            base, seed=config.seed,
            window=config.probe.rolling_window, stride=config.probe.rolling_stride,
        )
        variant_name = args.variant or "T1-mean_pool"
        thresholds = fixture_thresholds(
            config, cache, baseline, variant,
            variant_name=variant_name,
            budgets={"P-fixture-low": 0.10, "P-fixture-high": 0.50},
        )
    else:
        baseline = load_evalset(str(evalsets_dir / f"{args.baseline_set}.json"))
        variant = load_evalset(str(evalsets_dir / f"{args.variant_set}.json"))
        source = cache_source_id(variant)
        # No re-extraction is permitted here (DECISIONS 081). If the cache is
        # missing we stop, because regenerating activations would silently make
        # this a comparison of two extractions rather than two training sizes.
        cache_path = caches / f"cache-{source}.npz"
        if not cache_path.is_file():
            raise SystemExit(
                f"no extraction cache at {cache_path}. Stopping rather than "
                "re-extracting: this comparison is only meaningful if both "
                "models read the same activations."
            )
        cache = ExtractionCache.load(cache_path, expected_items=baseline)
        variant_name = args.variant or "T1-last_token"
        thresholds = _thresholds(results, args.baseline_set, args.variant_set)

    relationship = split_relationship(baseline, variant)
    _LOG.info("split relationship: %s", relationship.to_payload())
    if not relationship.usable:
        write_json_artifact(
            results / "paired_comparison.json",
            {
                "provenance": provenance(config),
                "split_relationship": relationship.to_payload(),
                "comparison": None,
                "withdrawn": (
                    "The paired set is too small or contaminated to support a "
                    "comparison. The 079 claim is withdrawn to: AUROC on the "
                    "larger test set is consistent with the smaller one; the two "
                    "are measured on different samples and the effect of the "
                    "training-set reduction was not isolated."
                ),
            },
        )
        _LOG.error("paired set unusable; claim withdrawn, see the artifact")
        return 1

    payload = compare_models(
        config,
        cache,
        baseline_set=baseline,
        variant_set=variant,
        variant_name=variant_name,
        thresholds=thresholds,
        n_bootstrap=args.bootstrap,
        seed=config.seed,
    )
    # Block C.2: the geometry of the curve the three profiles sit on. Measured
    # on the variant model's own test split, since that is the envelope the
    # warrants now describe.
    variant_probe, variant_splits = fit_on(config, cache, variant, variant=variant_name)
    features = cache.matrix(variant_name)
    rows = variant_splits[TEST]
    curve = roc_curve(
        variant_probe.score(features[rows]),
        cache.labels[rows],
        operating_points={k: v[1] for k, v in thresholds.items()},
    )
    plot_path = plot_roc_operating_points(
        curve,
        results / "roc_operating_points.png",
        title=(
            f"{variant.eval_set_id} · {variant_name} · n={len(rows)} "
            "— three profiles on one measured curve"
        ),
        config_hash=config.config_hash,
    )

    write_json_artifact(
        results / "paired_comparison.json",
        {"provenance": provenance(config), **payload, "roc": curve.to_payload()},
    )
    _LOG.info("wrote %s and %s", results / "paired_comparison.json", plot_path)

    for regime in ("pinned_to_baseline_threshold", "each_at_its_own_threshold"):
        _LOG.info("-- %s", regime)
        for row in payload[regime]:
            _LOG.info(
                "   %-34s %.4f -> %.4f  delta %+.4f [%+.4f, %+.4f]  MDD %.4f",
                row["quantity"], row["baseline"], row["variant"], row["difference"],
                row["ci_low"], row["ci_high"], row["minimum_detectable"],
            )
    _LOG.info("-- ROC geometry on the variant test split")
    for point in curve.points:
        _LOG.info(
            "   %-28s fpr %.4f  recall %.4f  f %.4f  slope %.3f (window %.3f)",
            point.operating_point_id, point.fpr, point.tpr, point.flag_rate,
            point.slope, point.window_fpr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
