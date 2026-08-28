"""Build, freeze, register and validate the evaluation sets.

Thin wrapper: parses arguments, calls ``src/``, writes files. No logic
(``CLAUDE.md``).

The TriviaQA sets are **not** built here. They need model generations to label —
whether an answer was incorrect is not a property of the question — so they come
from the extraction stage, and this script builds the sets that can be
constructed without a model.

Usage:
    python scripts/01_build_evalsets.py --config config.yaml
    python scripts/01_build_evalsets.py --verify-only
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
from src.detectors.pii_reference import ReferencePiiDetector
from src.evalsets import (
    build_canary_pii,
    build_hard_negatives,
    build_hinglish_pii,
    build_longctx,
    save_evalset,
    verify_manifest,
    write_manifest,
)
from src.store import Ledger, RecordKind
from src.validation.text_runner import validate_text_detector

_LOG = logging.getLogger("scripts.01_build_evalsets")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="re-hash the registered sets and report drift, building nothing",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="build and register without scoring the reference detector",
    )
    parser.add_argument("--out", default=None, help="results directory override")
    parser.add_argument(
        "--evalsets-out",
        default=None,
        help="evalsets directory override, so a smoke run cannot clobber the "
        "registered sets",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)

    evalsets_dir = (
        Path(args.evalsets_out)
        if args.evalsets_out
        else PROJECT_ROOT / config.paths.evalsets_dir
    )
    results_dir = Path(args.out) if args.out else PROJECT_ROOT / config.paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        problems = verify_manifest(evalsets_dir)
        for problem in problems:
            _LOG.error("%s", problem)
        if problems:
            _LOG.error("%d registered set(s) no longer match their hash", len(problems))
            return 1
        _LOG.info("every registered eval set still matches its recorded hash")
        return 0

    hinglish = build_hinglish_pii(seed=config.seed)
    hard_negatives = build_hard_negatives(seed=config.seed)
    canary = build_canary_pii(seed=config.seed)
    longctx_spec = next(
        (s for s in config.evalsets if s.pad_tokens is not None), None
    )
    pad_tokens = longctx_spec.pad_tokens if longctx_spec else (4000, 16000)
    hinglish_longctx = build_longctx(
        hinglish,
        seed=config.seed,
        pad_tokens=pad_tokens,
        eval_set_id="hinglish-pii-200-longctx",
    )

    built = [hinglish, hinglish_longctx, hard_negatives, canary]
    for evalset in built:
        save_evalset(evalset, evalsets_dir)
    write_manifest(
        built,
        evalsets_dir,
        extra={
            "not_built_here": {
                "triviaqa-600": (
                    "needs model generations to label; built by the extraction "
                    "stage, since whether an answer is incorrect is not a "
                    "property of the question"
                ),
                "triviaqa-longctx-600": (
                    "derived from triviaqa-600 by build_longctx once that set "
                    "exists"
                ),
                "triviaqa-2400-t960": (
                    "derived from triviaqa-600 by resplit_by_question -- same "
                    "items, 40/20/40 instead of 50/25/25, so the test split can "
                    "support a calibration claim at a 0.10 budget "
                    "(DECISIONS.md 079)"
                ),
            },
        },
    )

    problems = verify_manifest(evalsets_dir)
    if problems:
        for problem in problems:
            _LOG.error("%s", problem)
        return 1
    _LOG.info("registered %d sets, all hashes verified", len(built))

    # Build provenance goes to results/, not into the manifest. The manifest has
    # to be byte-identical across runs or the tree is never clean.
    write_json_artifact(
        results_dir / "evalset_build.json",
        {
            "sets": [
                {
                    "eval_set_id": e.eval_set_id,
                    "content_hash": e.content_hash,
                    "envelope_id": e.envelope_id,
                    "n_items": len(e),
                    "base_rate": e.base_rate,
                }
                for e in built
            ]
        },
        config,
    )

    if args.skip_validation:
        return 0

    detector = ReferencePiiDetector()
    threshold = detector.min_confidence
    ledger = Ledger(
        results_dir / Path(config.store.path).name,
        retention_days=config.store.retention_days,
    )
    summaries = []
    try:
        for evalset in (hinglish, hinglish_longctx, hard_negatives):
            is_hard_negatives = evalset is hard_negatives
            run = validate_text_detector(
                config,
                evalset,
                detector,
                threshold=threshold,
                canary_evalset=canary,
                # The declared hard-negative maximum applies only to the set it
                # is about. Applying it to the Hinglish set's own near-miss
                # negatives would judge the detector against a bar nobody set
                # for that envelope.
                max_fpr_hard_negatives=0.02 if is_hard_negatives else None,
                is_hard_negative_set=is_hard_negatives,
            )
            # A re-run at the same seed on the same code produces the same
            # content-derived warrant id. That is determinism working, not a
            # new fact, so it is not appended twice.
            if ledger.contains(RecordKind.WARRANT, run.warrant.warrant_id):
                _LOG.info(
                    "warrant %s already recorded for %s; not appending a "
                    "duplicate of an identical validation",
                    run.warrant.warrant_id,
                    evalset.eval_set_id,
                )
            else:
                ledger.append_warrant(run.warrant)
            write_json_artifact(
                results_dir / f"validation-{detector.detector_id}-{evalset.eval_set_id}.json",
                run.to_payload(),
                config,
            )
            summaries.append(
                {
                    "eval_set_id": evalset.eval_set_id,
                    "envelope_id": evalset.envelope_id,
                    "n_items": len(evalset),
                    "base_rate": evalset.base_rate,
                    "warrant_status": run.warrant.status.value,
                    "status_reason": run.warrant.status_reason,
                    "metrics": {
                        m.name: {
                            "value": m.value,
                            "ci_low": m.ci_low,
                            "ci_high": m.ci_high,
                            "kind": m.kind.value,
                            "n": m.n,
                        }
                        for m in run.metrics.all_metrics()
                    },
                    "controls_run": sum(1 for c in run.controls if c.applicable),
                    "controls_inapplicable": [
                        c.control for c in run.controls if not c.applicable
                    ],
                }
            )
        verification = ledger.verify_chain()
        _LOG.info(
            "ledger: %d records, chain %s",
            verification.n_records,
            "intact" if verification.ok else "BROKEN",
        )
    finally:
        ledger.close()

    write_json_artifact(
        results_dir / "evalset_validation.json",
        {
            "detector_id": detector.detector_id,
            "detector_version": detector.detector_version,
            "threshold": threshold,
            "note": (
                "The reference PII detector is ours, not Presidio. It exists so "
                "these sets are measured rather than merely built, and so the "
                "Phase 8 Presidio comparison has an honest floor to sit against."
            ),
            "runs": summaries,
        },
        config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
