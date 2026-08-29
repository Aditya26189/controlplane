"""Freeze the per-item scores behind every claim, so a clean clone can check them.

Block E, E.8 follow-up. Thin wrapper: parses arguments, calls
``controlplane/``, writes files. No logic (``CLAUDE.md``).

``make verify``'s second check re-derives the committed metrics instead of
trusting them. Until this existed it could only do that from the extraction
caches, which are ~100 MB and gitignored — so on the clone a judge actually
has, the substantive half of verification reported SKIPPED.

Every metric in this repository is a function of four arrays: labels, scores,
question ids and a threshold. They are kilobytes. This writes them to
``results/scores/`` beside the artifacts they reproduce, and the verifier
recomputes each metrics block from them with the same estimator and seed.

**Runs the real runners.** It does not re-implement scoring — it calls
``validate``, ``validate_transferred`` and ``validate_text_detector`` and takes
the scoring off the run. Reproducing the pipeline in a freezer script would be
the dual-path failure ``DECISIONS.md`` 048 eliminated: two ways to compute one
number, agreeing until they do not.

Needs the extraction caches, so it runs where they exist and the output is
committed. Usage:

    python scripts/10_freeze_scores.py --config config.yaml
    python scripts/10_freeze_scores.py --config config.yaml --only triviaqa-600
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from controlplane.config import (  # noqa: E402
    load_config,
    provenance,
    set_seeds,
    setup_logging,
)
from controlplane.detectors.pii_reference import ReferencePiiDetector  # noqa: E402
from controlplane.detectors.presidio_adapter import (  # noqa: E402
    PresidioDetector,
    presidio_available,
)
from controlplane.evalsets.registry import load_evalset  # noqa: E402
from controlplane.detectors.probe import LinearProbe  # noqa: E402
from controlplane.validation.evalsets import (  # noqa: E402
    TRAIN,
    ExtractionCache,
    split_by_question,
)
from controlplane.validation.runner import validate, validate_transferred  # noqa: E402
from controlplane.validation.scores import (  # noqa: E402
    ScoreTarget,
    from_scoring,
    write_score_set,
)
from controlplane.validation.text_runner import validate_text_detector  # noqa: E402

_LOG = logging.getLogger("scripts.10_freeze_scores")

#: Aggregations validated on the frozen TriviaQA envelope.
VARIANTS = ("T1-last_token", "T1-max_rolling_means", "T1-mean_pool")


def _write(config, score_set, out: Path) -> None:
    path = out / f"{score_set.score_set_id}.json"
    write_score_set(score_set, path, provenance(config))
    _LOG.info(
        "wrote %s  n=%d  base_rate=%.4f  targets=%d",
        path.name, score_set.n, score_set.base_rate, len(score_set.targets),
    )


def freeze_probe_scores(config, out: Path, results: Path) -> int:
    """The activation-tier scorings: validation on triviaqa-600, then transfer."""
    cache_path = results / "cache-triviaqa-600.npz"
    if not cache_path.is_file():
        _LOG.warning("no %s; skipping the probe score sets", cache_path.name)
        return 0

    evalsets_dir = PROJECT_ROOT / config.paths.evalsets_dir
    evalset = load_evalset(evalsets_dir / "triviaqa-600.json")
    cache = ExtractionCache.load(cache_path)
    canary_path = results / "cache-canary-20-triviaqa.npz"
    canary = ExtractionCache.load(canary_path) if canary_path.is_file() else None

    long_path = results / "cache-triviaqa-longctx-600.npz"
    long_evalset = None
    long_cache = None
    if long_path.is_file():
        long_evalset = load_evalset(evalsets_dir / "triviaqa-longctx-600.json")
        long_cache = ExtractionCache.load(long_path)
    else:
        _LOG.warning("no %s; the transfer score sets will be skipped", long_path.name)

    written = 0
    for variant in VARIANTS:
        # Detector id and version match scripts/02_validate.py and
        # 04_transfer.py exactly. They do not affect a metric, but a score
        # set filed under a different detector than the artifact it claims
        # to reproduce would be a mislabelled piece of evidence.
        detector_id = (
            f"probe-{config.model.name.split(chr(47))[-1].lower()}-{variant}"
        )
        run = validate(
            config,
            evalset,
            cache,
            variant=variant,
            detector_id=detector_id,
            detector_version="0.1.0",
            operating_point_id="P-conservative",
            canary_cache=canary,
        )
        _write(
            config,
            from_scoring(
                f"validation-{variant}",
                run.scoring,
                detector_id=run.detector_id,
                variant=variant,
                eval_set_id=run.eval_set_id,
                envelope_id=run.envelope_id,
                targets=[
                    ScoreTarget(
                        artifact=f"results/validation-{variant}.json",
                        metrics_path="metrics",
                        threshold=run.operating_point.threshold,
                        operating_point_id=run.operating_point.operating_point_id,
                    )
                ],
                note=(
                    "Test-split scores from the probe fitted on triviaqa-600's "
                    "train split. Reproduces the metrics block of the artifact "
                    "named in targets."
                ),
            ),
            out,
        )
        written += 1

        if long_cache is None:
            continue
        # Refit with the C the source run selected, on the source TRAIN split
        # only -- exactly as scripts/04_transfer.py does. ValidationRun records
        # how the probe was fitted, not the fitted object, so the probe is
        # reconstructed rather than carried. Same seed, same C, so this is the
        # same probe; nothing is selected on the target envelope.
        splits = split_by_question(evalset, seed=config.seed)
        probe = LinearProbe(
            run.probe_fit.C,
            class_weight=config.probe.class_weight,
            standardize=config.probe.standardize,
            seed=config.seed,
        ).fit(cache.matrix(variant), cache.labels, splits[TRAIN])
        transferred = validate_transferred(
            config,
            long_evalset,
            long_cache,
            source=run,
            probe=probe,
            variant=variant,
        )
        _write(
            config,
            from_scoring(
                f"transfer-{variant}",
                transferred.scoring,
                detector_id=transferred.detector_id,
                variant=variant,
                eval_set_id=transferred.eval_set_id,
                envelope_id=transferred.envelope_id,
                targets=[
                    ScoreTarget(
                        artifact=f"results/transfer-{variant}.json",
                        metrics_path="target.metrics",
                        threshold=transferred.operating_point.threshold,
                        operating_point_id=(
                            transferred.operating_point.operating_point_id
                        ),
                    )
                ],
                note=(
                    "The same probe, unrefitted, scored on the long-context "
                    "envelope. The threshold is the source run's; nothing is "
                    "reselected, which is what makes the drop a property of the "
                    "shift rather than of a new fit."
                ),
            ),
            out,
        )
        written += 1
    return written


#: Committed artifacts that may hold a text-detector metrics block, and how a
#: (detector, eval set) pair is addressed inside each. A pair can legitimately
#: appear in more than one -- hard-negatives-200 is measured in both the main
#: and the holdout run -- and a score set then backs every block it reproduces
#: rather than an arbitrary first match.
_TEXT_ARTIFACTS = (
    ("results/detectors.json", "runs[detector_id={d},eval_set_id={e}].metrics"),
    ("results/holdout/detectors.json", "runs[detector_id={d},eval_set_id={e}].metrics"),
    ("results/validation-pii-reference-{e}.json", "metrics"),
)


def _owner_of(document: dict, metrics_path: str):
    """The (detector, eval set) a metrics block belongs to, if it declares one.

    Returns None when the artifact does not say, in which case the caller has
    no identity evidence either way and the target is allowed through.
    """
    if metrics_path == "metrics":
        if "detector_id" in document and "eval_set_id" in document:
            return (document["detector_id"], document["eval_set_id"])
        return None
    # A selector path already pins the identity by construction.
    return None


def _targets_for(detector_id: str, eval_set_id: str, operating_point) -> list:
    """Every committed metrics block this scoring should reproduce.

    Resolved by looking inside the artifacts rather than by guessing from the
    detector's name. A target naming a block that is not there would fail at
    verification time with a confusing error; one that is silently skipped
    would leave a claim unchecked.
    """
    from controlplane.report.claims import resolve

    found = []
    for template, path_template in _TEXT_ARTIFACTS:
        artifact = template.format(d=detector_id, e=eval_set_id)
        full = PROJECT_ROOT / artifact
        if not full.is_file():
            continue
        metrics_path = path_template.format(d=detector_id, e=eval_set_id)
        document = json.loads(full.read_text(encoding="utf-8"))
        try:
            resolve(document, metrics_path)
        except (KeyError, ValueError):
            continue
        # A resolving path is not proof of a matching block. "metrics" resolves
        # in EVERY single-run artifact, so without this check every Presidio
        # score set acquired a target pointing at the pii-reference artifact,
        # whose metrics are a different detector's. Caught by the verifier on
        # its first run, which is what the verifier is for -- but a target has
        # to be identified by identity, not by a path that happens to exist.
        owner = _owner_of(document, metrics_path)
        if owner is not None and owner != (detector_id, eval_set_id):
            continue
        found.append(
            ScoreTarget(
                artifact=artifact,
                metrics_path=metrics_path,
                threshold=operating_point.threshold,
                operating_point_id=operating_point.operating_point_id,
            )
        )
    return found


def freeze_text_scores(config, out: Path) -> int:
    """The text tier: the PII detectors on their two envelopes.

    Frozen even though they need no GPU, because they need Presidio and a spaCy
    model. A judge without those could not check the refusal that the demo's
    third beat rests on.
    """
    detectors = [("pii-reference", ReferencePiiDetector())]
    if presidio_available():
        detectors += [
            (f"presidio-{c}", PresidioDetector(c))
            for c in config.detectors.presidio_configs
        ]
    else:
        _LOG.warning("presidio unavailable; freezing the reference detector only")

    evalsets_dir = PROJECT_ROOT / config.paths.evalsets_dir
    canary = load_evalset(evalsets_dir / "canary-20-pii.json")
    written = 0
    for eval_set_id, artifact_dir, hard_negatives in (
        ("hinglish-pii-200", "results", False),
        ("hard-negatives-200", "results", True),
        ("hinglish-pii-200b", "results/holdout", False),
    ):
        path = evalsets_dir / f"{eval_set_id}.json"
        if not path.is_file():
            _LOG.warning("no eval set at %s; skipping", path)
            continue
        evalset = load_evalset(path)
        for detector_id, detector in detectors:
            run = validate_text_detector(
                config,
                evalset,
                detector,
                # The detector's own declared confidence floor, exactly as
                # 09_detectors.py uses it. A threshold fitted here would make
                # the frozen scores describe the fitting rather than the
                # detector, and they would no longer reproduce that artifact.
                threshold=detector.min_confidence,
                canary_evalset=canary,
                max_fpr_hard_negatives=(
                    max(p.max_fpr for p in config.profiles.values())
                    if hard_negatives
                    else None
                ),
                is_hard_negative_set=hard_negatives,
            )
            targets = _targets_for(
                detector_id, eval_set_id, run.operating_point
            )
            if not targets:
                _LOG.info(
                    "no committed artifact holds %s on %s; not freezing it",
                    detector_id, eval_set_id,
                )
                continue
            _write(
                config,
                from_scoring(
                    f"text-{detector_id}-{eval_set_id}",
                    run.scoring,
                    detector_id=detector_id,
                    variant="text",
                    eval_set_id=eval_set_id,
                    envelope_id=run.envelope_id,
                    targets=targets,
                    note=(
                        f"{detector_id} scored on every item of {eval_set_id}. "
                        "Frozen so the refusal is checkable without Presidio "
                        "and its spaCy model installed."
                    ),
                ),
                out,
            )
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "results" / "scores"))
    parser.add_argument(
        "--only",
        choices=("probe", "text", "all"),
        default="all",
        help="freeze one tier only",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = PROJECT_ROOT / "results"

    written = 0
    if args.only in ("probe", "all"):
        written += freeze_probe_scores(config, out, results)
    if args.only in ("text", "all"):
        written += freeze_text_scores(config, out)

    _LOG.info("froze %d score sets into %s", written, out)
    if written == 0:
        _LOG.error(
            "nothing was frozen. Without the extraction caches this script has "
            "nothing to read; run it where they exist."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
