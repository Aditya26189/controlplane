"""Issue the three operating points, load the three bundles, and decide one input.

Thin wrapper: parses arguments, calls ``src/``, writes files. No logic
(``CLAUDE.md``). Anything this script decides is a decision nobody can review in
a diff of the pipeline.

Usage:
    python scripts/07_policy.py --config config.yaml
    python scripts/07_policy.py --config config.yaml --fixture
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, provenance, set_seeds, setup_logging, write_json_artifact
from src.evalsets.registry import load_evalset
from src.policy.runner import issue_operating_points, run_profile_comparison
from src.store import Ledger
from src.validation.evalsets import ExtractionCache
from src.validation.synthetic import synthetic_cache, synthetic_evalset

_LOG = logging.getLogger("scripts.07_policy")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--fixture",
        action="store_true",
        help=(
            "run against a synthetic cache instead of the measured extraction. "
            "Every artifact is stamped data_source=synthetic and the matrix "
            "masks its numbers; this exercises the pipeline, it does not "
            "produce results."
        ),
    )
    parser.add_argument("--detector", default="probe-qwen2.5-7b-instruct-T1-last_token")
    parser.add_argument(
        "--variant",
        default=None,
        help=(
            "tier variant. Defaults to T1-last_token on the measured cache and "
            "T1-mean_pool under --fixture, which is the variant a synthetic "
            "cache carries. An explicit variant the cache does not hold is an "
            "error rather than a silent substitution."
        ),
    )
    parser.add_argument("--detector-version", default="0.1.0")
    parser.add_argument("--eval-set", default="triviaqa-600")
    parser.add_argument("--out", default=None, help="results directory override")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)

    results = Path(args.out) if args.out else PROJECT_ROOT / config.paths.results_dir
    results.mkdir(parents=True, exist_ok=True)
    # The caches live beside the committed results, never in an override
    # directory, because a smoke run must not depend on a GPU extraction being
    # copied into a temporary path.
    caches = PROJECT_ROOT / config.paths.results_dir

    if args.fixture:
        evalset = synthetic_evalset(
            eval_set_id=f"{args.eval_set}-synthetic", n_items=1800, base_rate=0.46,
            seed=config.seed, items_per_question=2, declare_splits=True,
        )
        cache = synthetic_cache(
            evalset,
            seed=config.seed,
            window=config.probe.rolling_window,
            stride=config.probe.rolling_stride,
        )
        canary_evalset = synthetic_evalset(
            eval_set_id="canary-20-synthetic", n_items=20, base_rate=0.95, seed=7
        )
        canary = synthetic_cache(
            canary_evalset,
            seed=7,
            window=config.probe.rolling_window,
            stride=config.probe.rolling_stride,
            signal_by_tier={v: 8.0 for v in cache.variants},
            amplitude_spread=0.05,
        )
    else:
        evalset = load_evalset(
            str(PROJECT_ROOT / config.paths.evalsets_dir / f"{args.eval_set}.json")
        )
        cache = ExtractionCache.load(
            caches / f"cache-{args.eval_set}.npz", expected_hash=evalset.content_hash
        )
        canary_path = caches / "cache-canary-20-triviaqa.npz"
        canary = ExtractionCache.load(canary_path) if canary_path.is_file() else None

    variant = args.variant or ("T1-mean_pool" if args.fixture else "T1-last_token")
    if variant not in cache.variants:
        raise SystemExit(
            f"cache for {evalset.eval_set_id} holds {sorted(cache.variants)}, "
            f"not {variant!r}"
        )

    ledger = Ledger(results / "policy.db", retention_days=config.store.retention_days)
    try:
        runs = issue_operating_points(
            config,
            evalset,
            cache,
            variant=variant,
            detector_id=args.detector,
            detector_version=args.detector_version,
            canary_cache=canary,
            ledger=ledger,
            progress=_LOG.info,
        )
        comparison = run_profile_comparison(
            config, ledger, policies_dir=PROJECT_ROOT / config.paths.policies_dir
        )
    finally:
        ledger.close()

    write_json_artifact(
        results / "policy.json",
        {
            "provenance": provenance(config),
            "operating_points": [run.to_payload() for run in runs],
            "comparison": comparison.to_payload(),
        },
    )
    _LOG.info("wrote %s", results / "policy.json")

    for row in comparison.rows:
        _LOG.info(
            "%-20s tau=%.6f  ->  %-8s  %s",
            row.profile, row.threshold, row.action.value, row.rule_id,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
