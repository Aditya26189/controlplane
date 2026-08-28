"""Validate the Presidio configurations and issue or refuse their warrants.

Thin wrapper: parses arguments, calls ``src/``, writes files. No logic
(``CLAUDE.md``).

The point of this stage is the **refusal**. Stock Presidio is measured on
Hinglish traffic and, where it does not clear a profile's floor, it is refused a
warrant with the measured recall in the refusal reason. A component that is
correctly refused is a stronger artifact than one tuned until it passed.

Usage:
    python scripts/09_detectors.py --config config.yaml
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
from src.detectors.presidio_adapter import CONFIGURATIONS, PresidioDetector, presidio_available
from src.detectors.pii_reference import ReferencePiiDetector
from src.evalsets.registry import load_evalset
from src.store import Ledger
from src.validation.text_runner import validate_text_detector

_LOG = logging.getLogger("scripts.09_detectors")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--eval-set", default="hinglish-pii-200")
    parser.add_argument("--hard-negatives", default="hard-negatives-200")
    parser.add_argument(
        "--configs",
        default=None,
        help=(
            "comma-separated Presidio configurations to measure. Defaults to "
            "all three, which DECISIONS 008 requires for a reported result. "
            "Narrow it only for a smoke run, never for a published one."
        ),
    )
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--out", default=None, help="results directory override")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)

    results = Path(args.out) if args.out else PROJECT_ROOT / config.paths.results_dir
    results.mkdir(parents=True, exist_ok=True)
    evalsets_dir = PROJECT_ROOT / config.paths.evalsets_dir

    if not presidio_available():
        # Reported, never silent. An adapter that does not run is how a
        # detector's absence gets mistaken for a detector's clean sheet.
        _LOG.error(
            "presidio-analyzer is not installed; this stage measures it and "
            "cannot substitute anything else. `pip install presidio-analyzer` "
            "and `python -m spacy download en_core_web_lg`."
        )
        return 1

    evalset = load_evalset(str(evalsets_dir / f"{args.eval_set}.json"))
    hard_negatives = load_evalset(str(evalsets_dir / f"{args.hard_negatives}.json"))
    canary = load_evalset(str(evalsets_dir / "canary-20-pii.json"))

    selected = (
        tuple(c.strip() for c in args.configs.split(",")) if args.configs
        else CONFIGURATIONS
    )
    unknown = sorted(set(selected) - set(CONFIGURATIONS))
    if unknown:
        raise SystemExit(f"unknown Presidio configuration(s) {unknown}")
    if set(selected) != set(CONFIGURATIONS):
        _LOG.warning(
            "measuring %s of %s. DECISIONS 008 requires all three in a reported "
            "result; a partial run is a smoke run.",
            list(selected), list(CONFIGURATIONS),
        )
    detectors = [PresidioDetector(name) for name in selected]
    if not args.skip_reference:
        detectors.append(ReferencePiiDetector())

    # A stage ledger is regenerated per run and is gitignored: its hash chain
    # is only meaningful for the run that built it. Removed explicitly with a
    # log line rather than appended to, because the ledger is append-only and a
    # second run of a deterministic stage produces the same warrant ids.
    ledger_path = results / "detectors.db"
    if ledger_path.exists():
        _LOG.info("removing the previous stage ledger at %s", ledger_path)
        ledger_path.unlink()
    ledger = Ledger(ledger_path, retention_days=config.store.retention_days)
    runs = []
    try:
        for detector in detectors:
            for target, is_hard_negative in ((evalset, False), (hard_negatives, True)):
                _LOG.info("validating %s on %s", detector.detector_id, target.eval_set_id)
                run = validate_text_detector(
                    config,
                    target,
                    detector,
                    # The detector's own declared confidence floor, not a value
                    # fitted to this set. Fitting it here would make every
                    # measured number a statement about the fitting rather than
                    # about the detector.
                    threshold=detector.min_confidence,
                    # The canary runs on BOTH sets. Withholding it from the
                    # hard-negative run made the control fail rather than be
                    # inapplicable, which refused every detector there for a
                    # reason about this script rather than about the detector.
                    canary_evalset=canary,
                    # A single-class envelope can only support an FPR claim, so
                    # issuance requires a declared ceiling. The loosest profile
                    # ceiling is used: the warrant records what was measured,
                    # and each profile applies its own tighter bar at policy
                    # load time (SPEC.md 7.2), which is where that decision
                    # belongs.
                    max_fpr_hard_negatives=(
                        max(p.max_fpr for p in config.profiles.values())
                        if is_hard_negative
                        else None
                    ),
                    is_hard_negative_set=is_hard_negative,
                    progress=_LOG.info,
                )
                ledger.append_warrant(run.warrant)
                runs.append(run)
    finally:
        ledger.close()

    write_json_artifact(
        results / "detectors.json",
        {
            "provenance": provenance(config),
            "runs": [run.to_payload() for run in runs],
        },
    )
    _LOG.info("wrote %s", results / "detectors.json")

    for run in runs:
        recall = run.metrics.recall
        _LOG.info(
            "%-30s %-24s %-8s recall %s",
            run.detector_id,
            run.eval_set_id,
            run.warrant.status.value,
            "n/a" if recall is None else f"{recall.value:.4f} [{recall.ci_low:.4f}, {recall.ci_high:.4f}]",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
