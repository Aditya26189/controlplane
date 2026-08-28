"""Transfer each fitted probe to the long-context envelope, and report both.

This is the Beat 4 measurement. The probe is fitted on ``triviaqa-600`` and
scored, unchanged, on ``triviaqa-longctx-600`` — the same questions padded into
a 4-16k token context. Nothing is refitted and nothing is reselected: refitting
would produce a better number describing a system nobody runs, since no
production deployment retrains between one request and the next.

Both aggregations are carried through, because the comparison is the point.
``mean_pool`` is documented to collapse under long-context shift and
``max_rolling_means`` was built to survive it; reporting only one would leave
the claim untested in exactly the place it matters.

Thin wrapper: parses arguments, calls ``controlplane/``, writes files (``CLAUDE.md``).

Usage:
    python scripts/04_transfer.py --config config.yaml \
        --source-cache results/measured/cache-triviaqa-600.npz \
        --target-cache results/measured/cache-triviaqa-longctx-600.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlplane.config import load_config, provenance, set_seeds, setup_logging, write_json_artifact
from controlplane.evalsets.registry import load_evalset
from controlplane.store import Ledger, RecordKind
from controlplane.detectors.probe import LinearProbe
from controlplane.validation.evalsets import TRAIN, ExtractionCache, split_by_question
from controlplane.validation.runner import validate, validate_transferred

_LOG = logging.getLogger("scripts.04_transfer")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--target-cache", required=True)
    parser.add_argument("--source-eval-set", default="triviaqa-600")
    parser.add_argument("--target-eval-set", default="triviaqa-longctx-600")
    parser.add_argument("--target-flag-rate", type=float, default=0.05)
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def _load(eval_set_id: str, cache_path: str) -> tuple:
    """Load an eval set and its cache, refusing a mismatched pair.

    The content hash is the envelope id and the third element of every warrant
    key. Pairing a cache with the wrong eval set would attach real activations
    to someone else's labels and report a number for a detector-envelope pair
    that was never measured.
    """
    evalset = load_evalset(PROJECT_ROOT / "evalsets" / f"{eval_set_id}.json")
    cache = ExtractionCache.load(cache_path)
    if cache.eval_set_hash != evalset.content_hash:
        raise SystemExit(
            f"cache/eval-set mismatch for {eval_set_id}: cache records "
            f"{cache.eval_set_hash}, eval set hashes to {evalset.content_hash}"
        )
    return evalset, cache


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)

    out = Path(args.out) if args.out else PROJECT_ROOT / config.paths.results_dir
    out.mkdir(parents=True, exist_ok=True)

    source_evalset, source_cache = _load(args.source_eval_set, args.source_cache)
    target_evalset, target_cache = _load(args.target_eval_set, args.target_cache)

    # Both warrants go to the ledger. A transfer result that lives only in a
    # JSON file never reaches the matrix, and the matrix is where the envelope
    # comparison is supposed to be legible.
    ledger = Ledger(
        out / Path(config.store.path).name,
        retention_days=config.store.retention_days,
    )

    # Without a canary, canary_control fails closed and refuses every warrant
    # for a reason unrelated to the envelope -- which would bury the one real
    # refusal this script exists to surface.
    canary_cache = None
    canary_path = Path(args.source_cache).parent / "cache-canary-20-triviaqa.npz"
    if canary_path.exists():
        canary_cache = ExtractionCache.load(canary_path)
        _LOG.info("canary %s: %d items", canary_cache.eval_set_id, canary_cache.n_items)
    else:
        _LOG.warning(
            "no canary at %s; every warrant below will be refused on that "
            "control alone. Build one with scripts/05_canary.py.", canary_path,
        )

    rows = []
    for variant in sorted(source_cache.variants):
        _LOG.info("fitting %s on %s", variant, source_evalset.eval_set_id)
        source_run = validate(
            config,
            source_evalset,
            source_cache,
            # Model-qualified, matching 02_validate. A probe reads one
            # specific model's residual stream, so a model change invalidates
            # every activation-tier warrant; the id has to say which model.
            # Using a bare "probe-{variant}" here put the same measurement in
            # the matrix twice under two names.
            detector_id=(
                f"probe-{config.model.name.split('/')[-1].lower()}-{variant}"
            ),
            detector_version="0.1.0",
            variant=variant,
            target_flag_rate=args.target_flag_rate,
            canary_cache=canary_cache,
        )
        # Refit with the C the source run selected, on the source TRAIN split
        # only. ValidationRun records how the probe was fitted, not the fitted
        # object, so the probe is reconstructed rather than carried. The seed
        # and C both come from the source run, so this is the same probe --
        # nothing here is selected on the target envelope.
        splits = split_by_question(source_evalset, seed=config.seed)
        probe = LinearProbe(
            source_run.probe_fit.C,
            class_weight=config.probe.class_weight,
            standardize=config.probe.standardize,
            seed=config.seed,
        ).fit(
            source_cache.matrix(variant), source_cache.labels, splits[TRAIN]
        )
        _LOG.info("transferring %s to %s", variant, target_evalset.eval_set_id)
        target_run = validate_transferred(
            config,
            target_evalset,
            target_cache,
            source=source_run,
            probe=probe,
            variant=variant,
        )
        rows.append((variant, source_run, target_run))
        for run in (source_run, target_run):
            if ledger.contains(RecordKind.WARRANT, run.warrant.warrant_id):
                _LOG.info(
                    "warrant %s already recorded for %s on %s",
                    run.warrant.warrant_id, run.detector_id, run.eval_set_id,
                )
                continue
            ledger.append_warrant(run.warrant)
        write_json_artifact(
            out / f"transfer-{variant}.json",
            {"source": source_run.to_payload(), "target": target_run.to_payload()},
            config,
        )

    def _auroc(run):
        metric = run.metrics.auroc
        return (metric.value, metric.ci_low, metric.ci_high)

    print()
    print("Beat 4 — the same probe on two envelopes, nothing refitted")
    print("source %s [%s]" % (source_evalset.eval_set_id, source_evalset.content_hash))
    print("target %s [%s]" % (target_evalset.eval_set_id, target_evalset.content_hash))
    print()
    header = "%-24s %-28s %-28s %s" % ("variant", "AUROC on source", "AUROC on target", "target warrant")
    print(header)
    print("-" * len(header))
    for variant, source_run, target_run in rows:
        sv, sl, sh = _auroc(source_run)
        tv, tl, th = _auroc(target_run)
        print(
            "%-24s %.3f [%.3f, %.3f]        %.3f [%.3f, %.3f]        %s"
            % (variant, sv, sl, sh, tv, tl, th, target_run.warrant.status.value)
        )
    print()
    verification = ledger.verify_chain()
    _LOG.info(
        "ledger: %d records, chain %s",
        verification.n_records,
        "intact" if verification.ok else "BROKEN",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
