"""Extract TriviaQA activations for both envelopes, and self-check the output.

The only stage that needs a GPU. Everything downstream runs from the caches this
writes, which is what makes ``/validate`` fast enough to be a button.

Runs **both** envelopes in one session: ``triviaqa-600`` with declared splits,
and ``triviaqa-longctx-600`` over the test split. Beat 4 needs the second, and a
two-session plan is a plan where the second session does not happen.

Ends by checking its own output against the fixture path with
``assert_metric_shape_compatible``, so a normalisation or polarity divergence is
caught here rather than after the artifacts have been downloaded.

Thin wrapper: parses arguments, calls ``controlplane/``, writes files. No logic.

Usage:
    python scripts/00_extract.py --config config.yaml
    python scripts/00_extract.py --config config.yaml --smoke     # 120 questions
    python scripts/00_extract.py --config config.yaml --no-long-context
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlplane.config import load_config, set_seeds, setup_logging, write_json_artifact
from controlplane.detectors.probe import LinearProbe
from controlplane.evalsets import save_evalset
from controlplane.extract.model import load_model
from controlplane.extract.pipeline import extract_triviaqa
from controlplane.validation.evalsets import TRAIN, split_by_question
from controlplane.validation.metrics_builder import (
    assert_metric_shape_compatible,
    build_warrant_metrics,
)
from controlplane.validation.runner import validate, validate_transferred
from controlplane.validation.synthetic import synthetic_cache, synthetic_evalset

_LOG = logging.getLogger("scripts.00_extract")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--questions",
        type=int,
        default=2400,
        help="distinct questions after dedup; 2400 yields a 600-item test split",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--long-batch-size",
        type=int,
        default=1,
        help="1 by default: a 16k-token sequence at 7B NF4 does not batch on 16GB",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--no-long-context",
        action="store_true",
        help="skip the long-context pass. Only for debugging the short one; "
        "shipping without it leaves Beat 4 with no measured basis.",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="120 questions, for an end-to-end check"
    )
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    parser.add_argument("--out", default=None, help="results directory override")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)

    results_dir = Path(args.out) if args.out else PROJECT_ROOT / config.paths.results_dir
    evalsets_dir = PROJECT_ROOT / config.paths.evalsets_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    n_questions = 120 if args.smoke else args.questions

    loaded = load_model(config.model.name, quantization=config.model.quantization)
    result = extract_triviaqa(
        config,
        loaded,
        n_questions=n_questions,
        batch_size=args.batch_size,
        long_batch_size=args.long_batch_size,
        max_new_tokens=args.max_new_tokens,
        cache_dir=args.cache_dir,
        long_context=not args.no_long_context,
    )

    save_evalset(result.short_evalset, evalsets_dir)
    short_cache_path = result.short_cache.save(results_dir / "cache-triviaqa-600.npz")
    written = [str(short_cache_path)]
    if result.long_evalset is not None:
        save_evalset(result.long_evalset, evalsets_dir)
        written.append(
            str(result.long_cache.save(results_dir / "cache-triviaqa-longctx-600.npz"))
        )

    # -- self-check, on the GPU, before anything is downloaded ---------------- #
    _LOG.info("self-check: validating the extraction and comparing shape to the fixture")
    variant = f"T1-{config.probe.aggregations[0]}"
    source = validate(
        config,
        result.short_evalset,
        result.short_cache,
        variant=variant,
        detector_id=f"probe-{variant}",
        detector_version=f"0.1.0+{loaded.name.split('/')[-1]}",
        target_flag_rate=0.05,
    )
    _LOG.info("short-context run: %s", source.warrant.status.value)

    fixture_evalset = synthetic_evalset(
        eval_set_id="shape-check-synthetic",
        n_items=len(result.short_evalset),
        base_rate=result.short_evalset.base_rate,
        seed=config.seed,
        items_per_question=1,
        declare_splits=True,
    )
    fixture = synthetic_cache(
        fixture_evalset,
        seed=config.seed,
        window=config.probe.rolling_window,
        stride=config.probe.rolling_stride,
    )
    fixture_metrics = build_warrant_metrics(
        config,
        fixture.labels,
        fixture.matrix(variant)[:, 0],
        0.5,
        groups=fixture.question_ids,
    )
    assert_metric_shape_compatible(
        fixture_metrics,
        source.metrics,
        first_name="fixture path",
        second_name="measured extraction",
    )
    _LOG.info("shape check passed: the two paths produce the same metric structure")

    transferred = None
    if result.long_evalset is not None:
        splits = split_by_question(result.short_evalset, seed=config.seed)
        probe = LinearProbe(
            source.probe_fit.C,
            class_weight=config.probe.class_weight,
            standardize=config.probe.standardize,
            seed=config.seed,
        ).fit(result.short_cache.matrix(variant), result.short_cache.labels, splits[TRAIN])
        transferred = validate_transferred(
            config,
            result.long_evalset,
            result.long_cache,
            source=source,
            probe=probe,
            variant=variant,
        )
        _LOG.info(
            "long-context transfer: %s (recall %s)",
            transferred.warrant.status.value,
            transferred.metrics.recall.render(3) if transferred.metrics.recall else "n/a",
        )

    write_json_artifact(
        results_dir / "extraction.json",
        {
            "report": result.report,
            "caches": written,
            "eval_sets": {
                "triviaqa-600": {
                    "content_hash": result.short_evalset.content_hash,
                    "envelope_id": result.short_evalset.envelope_id,
                    "n_items": len(result.short_evalset),
                    "base_rate": result.short_evalset.base_rate,
                },
                **(
                    {
                        "triviaqa-longctx-600": {
                            "content_hash": result.long_evalset.content_hash,
                            "envelope_id": result.long_evalset.envelope_id,
                            "n_items": len(result.long_evalset),
                            "base_rate": result.long_evalset.base_rate,
                        }
                    }
                    if result.long_evalset is not None
                    else {}
                ),
            },
            "self_check": {
                "short_context_status": source.warrant.status.value,
                "short_context_reason": source.warrant.status_reason,
                "shape_compatible_with_fixture": True,
                "long_context_status": (
                    transferred.warrant.status.value if transferred else None
                ),
                "long_context_reason": (
                    transferred.warrant.status_reason if transferred else None
                ),
            },
        },
        config,
    )
    _LOG.info("extraction complete; caches are gitignored and must be downloaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
