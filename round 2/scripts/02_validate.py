"""Run /validate and the tier ablation on one eval set, and write the artifacts.

Thin wrapper: parses arguments, calls ``src/``, writes files. No logic
(``CLAUDE.md``). Anything this script decides is a decision nobody can review in
a diff of the pipeline.

Usage:
    python scripts/02_validate.py --config config.yaml
    python scripts/02_validate.py --config config.yaml --fixture --smoke
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
from src.report.plots import plot_tier_ladder
from src.store import Ledger
from src.validation.ablation import run_ablation
from src.validation.evalsets import ExtractionCache
from src.validation.synthetic import synthetic_cache, synthetic_evalset

_LOG = logging.getLogger("scripts.02_validate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--fixture",
        action="store_true",
        help=(
            "run against a synthetic fixture instead of a real extraction. The "
            "fixture's eval set hashes differently, so its numbers occupy "
            "different matrix cells and can never be read as measured ones."
        ),
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="path to an extraction cache (.npz) produced by 01_extract.py",
    )
    parser.add_argument(
        "--eval-set", default="triviaqa-600", help="eval set id to validate on"
    )
    parser.add_argument(
        "--target-flag-rate",
        type=float,
        default=0.05,
        help="flag-rate budget for threshold selection, chosen on validation",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="tiny sizes, for the end-to-end check"
    )
    parser.add_argument("--out", default=None, help="results directory override")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)

    results_dir = Path(args.out) if args.out else PROJECT_ROOT / config.paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.fixture:
        n_items = 400 if args.smoke else 2400
        _LOG.warning(
            "running against a SYNTHETIC FIXTURE — these numbers are not "
            "measurements and must not reach RESULTS.md (DECISIONS.md 027)"
        )
        evalset = synthetic_evalset(
            eval_set_id=f"{args.eval_set}-synthetic",
            n_items=n_items,
            base_rate=0.152,
            seed=config.seed,
            items_per_question=2,
            declare_splits=True,
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
        canary_cache = synthetic_cache(
            canary_evalset,
            seed=7,
            window=config.probe.rolling_window,
            stride=config.probe.rolling_stride,
            signal_by_tier={v: 8.0 for v in cache.variants},
            amplitude_spread=0.05,
        )
    else:
        if not args.cache:
            _LOG.error(
                "no --cache given and --fixture not set. A real validation needs an "
                "extraction from 01_extract.py; there is deliberately no default "
                "that would silently fall back to synthetic data."
            )
            return 2
        cache = ExtractionCache.load(args.cache)
        _LOG.error(
            "loading a measured eval set is Phase 3 work; %s carries %d items but "
            "the registry that turns it back into an EvalSet does not exist yet",
            args.cache,
            cache.n_items,
        )
        return 2

    ladder = run_ablation(
        config,
        evalset,
        cache,
        detector_prefix=f"probe-{config.model.name.split('/')[-1].lower()}",
        detector_version="0.1.0+fixture" if args.fixture else "0.1.0",
        target_flag_rate=args.target_flag_rate,
        canary_cache=canary_cache,
    )

    print()
    print(ladder.render())
    print()

    suffix = "-fixture" if args.fixture else ""
    write_json_artifact(
        results_dir / f"tier_ladder{suffix}.json", ladder.to_payload(), config
    )
    for run in ladder.runs:
        write_json_artifact(
            results_dir / f"validation-{run.variant}{suffix}.json",
            run.to_payload(),
            config,
        )

    prov = provenance(config)
    plot_tier_ladder(
        ladder,
        results_dir / f"tier_ladder{suffix}.png",
        config_hash=prov.get("config_hash"),
        git_commit=prov.get("git_commit"),
    )

    ledger = Ledger(
        PROJECT_ROOT / config.store.path, retention_days=config.store.retention_days
    )
    try:
        for run in ladder.runs:
            if ledger.contains(RecordKind.WARRANT, run.warrant.warrant_id):
                _LOG.info(
                    "warrant %s already recorded; identical validation, not "
                    "appending a duplicate",
                    run.warrant.warrant_id,
                )
                continue
            ledger.append_warrant(run.warrant)
            ledger.append_validation_run(
                run.run_id + "-" + run.variant,
                run.to_payload(),
                eval_set_id=run.eval_set_id,
                detector_id=run.detector_id,
            )
        verification = ledger.verify_chain()
        _LOG.info(
            "ledger: %d records, chain %s",
            verification.n_records,
            "intact" if verification.ok else f"BROKEN at seq {verification.first_break_seq}",
        )
    finally:
        ledger.close()

    slowest = max(run.duration_seconds for run in ladder.runs)
    _LOG.info("slowest single validation: %.2fs", slowest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
