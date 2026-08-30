"""Price the pilot's AUROC margin: how often does the gate clear, across seeds?

``DECISIONS.md`` 114. Thin wrapper: parses arguments, calls ``controlplane/``,
writes files. No logic (``CLAUDE.md``).

CPU only, seconds, and it needs no GPU because ``13_pilot_run.py`` now persists
the 24 pilot scores into ``results/pilot_run.json``.

**Why this exists.** The pilot cleared its 0.55 AUROC bar at 0.5554 -- by
0.0054. A percentile bootstrap's lower bound is itself a random variable, and
when a gate clears by less than the seed-to-seed spread of its own bound,
"it cleared" is a statement about the seed. The honest verdict is the fraction
of seeds that clear, not the one draw that was reported.

Usage:
    python scripts/15_pilot_seed_stability.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from controlplane.config import (  # noqa: E402
    load_config,
    set_seeds,
    setup_logging,
    write_json_artifact,
)
from controlplane.validation.stats import auroc, bootstrap_seed_stability  # noqa: E402

_LOG = logging.getLogger("scripts.15_pilot_seed_stability")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--pilot", default=str(PROJECT_ROOT / "results" / "pilot_run.json"),
        help="the pilot artifact, which must carry its scores",
    )
    parser.add_argument("--seeds", type=int, default=400)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "results" / "pilot_seed_stability.json"),
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)

    payload = json.loads(Path(args.pilot).read_text(encoding="utf-8"))
    if "scores" not in payload:
        _LOG.error(
            "%s carries no scores block. It predates the persistence added in "
            "DECISIONS 111, and the seed distribution cannot be computed from a "
            "summary. Re-run the pilot rather than approximating.",
            args.pilot,
        )
        return 1

    scores = payload["scores"]
    stability = bootstrap_seed_stability(
        np.asarray(scores["labels"], dtype=int),
        np.asarray(scores["pilot"], dtype=float),
        np.asarray(scores["question_ids"]),
        statistic=auroc,
        bar=payload["auroc"]["min_lower_ci_for_issuance"],
        ci=payload["auroc"]["ci_level"],
        n_seeds=args.seeds,
        n_resamples=args.resamples,
        published=payload["auroc"]["ci_low"],
    )

    _LOG.info(
        "AUROC lower bound across %d seeds: mean %.4f, sd %.4f, range "
        "[%.4f, %.4f]",
        stability.n_seeds, stability.mean, stability.sd,
        stability.minimum, stability.maximum,
    )
    _LOG.info(
        "CLEARS THE %.2f BAR IN %.1f%% OF SEEDS; the published %.4f sits at "
        "percentile %.0f",
        stability.bar, 100 * stability.clears_fraction,
        stability.published, stability.published_percentile,
    )
    if stability.clears_fraction < 0.5:
        _LOG.warning(
            "the reported verdict is a MINORITY outcome: the mean lower bound "
            "%.4f is below the %.2f bar, so 'clears_the_pilot' describes the "
            "seed as much as the detector",
            stability.mean, stability.bar,
        )

    write_json_artifact(
        Path(args.out),
        {
            "quantity": "bootstrap-seed distribution of the pilot's AUROC lower bound",
            "pilot_artifact": Path(args.pilot).name,
            "draft_content_hash": payload.get("draft_content_hash"),
            "point_auroc": payload["auroc"]["value"],
            "stability": stability.to_payload(),
            "verdict": (
                "the reported branch is a minority outcome across seeds"
                if stability.clears_fraction < 0.5
                else "the reported branch holds across a majority of seeds"
            ),
            "why": (
                "A percentile bootstrap's bound is a random variable. A gate "
                "cleared by less than the seed-to-seed sd of its own bound "
                "describes the seed as much as the detector. Reported as a "
                "fraction because that is the verdict; one draw is a sample "
                "from it."
            ),
            "preregistered_in": "DECISIONS.md 114",
        },
        config,
    )
    _LOG.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
