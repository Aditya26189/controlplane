"""Regenerate the IQR-ratio null band and its power, the numbers `101` routes on.

``DECISIONS.md`` 090 (corrected), 101, 103. Thin wrapper: parses arguments,
calls ``controlplane/``, writes files. No logic (``CLAUDE.md``).

CPU only, seconds. One output:

**``results/pilot_null_band.json``** -- the null band on the pilot/reference IQR
ratio across effective sample sizes and score shapes, plus the power of the
saturation rule against genuine collapses.

**Why this script exists.** ``SATURATION_IQR_RATIO = 0.439`` decides whether a
GPU run gets re-authored and whether the one retry in ``090`` is spent. It was
derived once, by hand, and lived as a constant with no code behind it --
failing ``CLAUDE.md``'s rule that every number in a document is computed by
code. This regenerates it, and reports the two things the constant never
carried: its false-alarm rate, and its power against a collapse it is supposed
to catch.

Usage:
    python scripts/14_pilot_null_band.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from controlplane.config import (  # noqa: E402
    load_config,
    set_seeds,
    setup_logging,
    write_json_artifact,
)
from controlplane.drift.null_band import (  # noqa: E402
    iqr_ratio_power,
    simulate_null_iqr_ratio,
)
from controlplane.evalsets.banking import SATURATION_IQR_RATIO  # noqa: E402
from controlplane.evalsets.registry import load_evalset  # noqa: E402
from controlplane.validation.evalsets import TEST  # noqa: E402

_LOG = logging.getLogger("scripts.14_pilot_null_band")

#: Effective sizes reported. 12 is the pilot as designed (12 questions, two
#: items each); 24 is what an item-level reading would wrongly assume; the rest
#: price what a larger pilot would buy.
_SIZES = (12, 24, 30, 60, 120)

#: True multiplicative shrinkages of the pilot's score spread. 1.0 is the null,
#: reported in the same table so the distance between a false alarm and a
#: detection is visible rather than inferred.
_COLLAPSES = (1.0, 0.8, 0.6, 0.5, 0.4)

_SHAPES = ("normal", "logistic", "beta2_2")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--evalsets-dir", default=str(PROJECT_ROOT / "evalsets"))
    parser.add_argument(
        "--reference", default="triviaqa-2400-t960",
        help="the envelope the probe was fitted on; its TEST split is the denominator",
    )
    parser.add_argument(
        "--n-reference", type=int, default=None,
        help=(
            "override the reference test-split size. Normally read from the "
            "reference eval set so the simulated denominator is the real one."
        ),
    )
    parser.add_argument(
        "--repeats", type=int, default=20000,
        help="simulation draws; 20,000 puts the p5 inside about +-0.01",
    )
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "results" / "pilot_null_band.json"),
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)

    # The denominator is the reference TEST split, because that is what
    # 13_pilot_run.py divides by. Simulating against a different size would
    # produce a band for a comparison nobody runs.
    n_reference = args.n_reference
    if n_reference is None:
        reference_path = Path(args.evalsets_dir) / f"{args.reference}.json"
        if not reference_path.is_file():
            _LOG.error(
                "no reference eval set at %s and no --n-reference given. The "
                "band is not computed rather than computed against a guess.",
                reference_path,
            )
            return 1
        reference_set = load_evalset(reference_path)
        n_reference = sum(1 for i in reference_set.items if i.split == TEST)
    _LOG.info("reference %s: %d items in the TEST split", args.reference, n_reference)

    null_band = {}
    for shape in _SHAPES:
        for n in _SIZES:
            band = simulate_null_iqr_ratio(
                n_pilot=n,
                n_reference=n_reference,
                threshold=SATURATION_IQR_RATIO,
                shape=shape,
                repeats=args.repeats,
                seed=config.seed,
            )
            null_band[f"{shape}/n{n}"] = band.to_payload()
            if shape == "normal":
                _LOG.info(
                    "n=%3d  95%% null band [%.3f, %.3f]  p5 %.3f  "
                    "false alarm at %.3f = %.4f",
                    n, band.p2_5, band.p97_5, band.p5,
                    SATURATION_IQR_RATIO, band.false_alarm_rate,
                )

    # Two power tables, and the difference between them is the finding. The
    # frozen threshold was calibrated AT n=12; reused at a larger n it holds
    # still while the null band tightens around it, so it grows more
    # conservative and less powerful at the same time.
    power_frozen: dict[str, float] = {}
    power_recalibrated: dict[str, float] = {}
    recalibrated_threshold: dict[str, float] = {}
    for n in _SIZES:
        p5 = simulate_null_iqr_ratio(
            n_pilot=n, n_reference=n_reference, threshold=SATURATION_IQR_RATIO,
            shape="normal", repeats=args.repeats, seed=config.seed,
        ).p5
        recalibrated_threshold[f"n{n}"] = p5
        for collapse in _COLLAPSES:
            key = f"n{n}/collapse{collapse}"
            power_frozen[key] = iqr_ratio_power(
                n_pilot=n, n_reference=n_reference,
                threshold=SATURATION_IQR_RATIO, collapse=collapse,
                shape="normal", repeats=args.repeats, seed=config.seed,
            )
            power_recalibrated[key] = iqr_ratio_power(
                n_pilot=n, n_reference=n_reference,
                threshold=p5, collapse=collapse,
                shape="normal", repeats=args.repeats, seed=config.seed,
            )
        _LOG.info(
            "n=%3d  frozen %.3f: P(fire | 0.6x)=%.3f   recalibrated %.3f: "
            "P(fire | 0.6x)=%.3f",
            n, SATURATION_IQR_RATIO, power_frozen[f"n{n}/collapse0.6"], p5,
            power_recalibrated[f"n{n}/collapse0.6"],
        )

    payload = {
        "quantity": "pilot score IQR / reference test-split score IQR",
        "reference_eval_set_id": args.reference,
        "n_reference": n_reference,
        "frozen_threshold": SATURATION_IQR_RATIO,
        "frozen_threshold_source": (
            "controlplane/evalsets/banking.py SATURATION_IQR_RATIO, derived by "
            "hand in the correction to DECISIONS 090 at 12 clusters"
        ),
        "effective_n_is_clusters": (
            "The pilot is 24 items over 12 questions and the two items of a "
            "question share it, so the effective n is 12. Simulating at 24 "
            "gives a p5 of about 0.60 and would call a healthy pilot saturated."
        ),
        "null_band": null_band,
        "power_frozen_threshold": power_frozen,
        "power_recalibrated_threshold": power_recalibrated,
        "recalibrated_threshold": recalibrated_threshold,
        "collapse_is": (
            "a multiplicative shrinkage of the pilot's score spread about its "
            "median. collapse=1.0 is the null, so that column is the "
            "false-alarm rate rather than power."
        ),
        "preregistered_in": "DECISIONS.md 090 (corrected), 101, 103",
    }
    write_json_artifact(Path(args.out), payload, config)
    _LOG.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
