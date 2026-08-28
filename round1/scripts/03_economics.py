"""Stage 5a: turn the measured test numbers into the three-policy comparison.

Parses arguments, calls ``src/``, writes files. No logic lives here.

Reads ``results/probe_test.json`` -- it never recomputes a metric, so the lift
reported here cannot drift from the one the probe stage measured.

Output:
    results/economics.json
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    load_config,
    read_json_artifact,
    set_seeds,
    setup_logging,
    write_json_artifact,
)
from src.economics import compare_policies  # noqa: E402

LOGGER = logging.getLogger("03_economics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)

    probe_test = read_json_artifact(config.results_path("probe_test.json"))
    test = probe_test["test"]
    boot = probe_test["bootstrap"]

    if not test["flag_rate"]:
        raise SystemExit(
            "measured test flag rate is zero: the probe flagged nothing, so lift "
            "is undefined. Lower the threshold or raise target_flag_rate."
        )

    # CLAUDE.md invariant 6: the MEASURED test flag rate, never the target.
    economics = compare_policies(
        n_responses=config.economics.n_responses,
        error_rate=config.economics.reference_error_rate,
        flag_rate=test["flag_rate"],
        recall=test["recall"],
        judge_accuracy=config.economics.judge_accuracy,
        recall_ci=(boot["recall"]["ci_low"], boot["recall"]["ci_high"]),
        flag_rate_ci=(boot["flag_rate"]["ci_low"], boot["flag_rate"]["ci_high"]),
        measured_base_rate=test["base_rate"],
        precision=test["precision"],
        roc=probe_test.get("roc"),
        projection_base_rates=config.economics.projection_base_rates,
    )
    economics["lift_ci"] = {
        "ci_low": boot["lift"]["ci_low"],
        "ci_high": boot["lift"]["ci_high"],
        "ci": boot["ci"],
        "n_samples": boot["n_samples"],
    }

    LOGGER.info(
        "measured f=%.4f  R=%.4f  ->  lift %.2fx [%.2f, %.2f]",
        economics["measured_flag_rate"],
        economics["measured_recall"],
        economics["lift"],
        boot["lift"]["ci_low"],
        boot["lift"]["ci_high"],
    )
    for row in economics["policies"]:
        # Pre-formatted: logging uses printf-style %-formatting, which has no
        # thousands separator, and we want one on six-figure judge-call counts.
        LOGGER.info(
            "  %-20s calls=%12s  coverage=%6s  caught=%10s  cost=%5.1fx",
            row["label"],
            f"{row['judge_calls']:,.0f}",
            f"{row['coverage'] * 100:.1f}%",
            f"{row['errors_caught']:,.0f}",
            row["relative_cost"],
        )
    LOGGER.info(
        "invariance: lift is identical across error rates %s and judge "
        "accuracies %s (spread %.1e)",
        economics["invariance"]["error_rates_tested"],
        economics["invariance"]["judge_accuracies_tested"],
        economics["invariance"]["spread"],
    )
    ceiling = economics.get("ceiling")
    if ceiling:
        LOGGER.info(
            "ceiling: base rate %.4f caps lift at %.2fx; measured %.2fx is %.1f%% "
            "of that maximum",
            ceiling["measured_base_rate"],
            ceiling["max_attainable_lift"],
            economics["lift"],
            100 * ceiling["fraction_of_ceiling_achieved"],
        )
    for row in economics.get("projection", {}).get("rows", []):
        LOGGER.info(
            "  PROJECTION at base rate %.2f: f=%.4f R=%.4f -> lift %.2fx "
            "(ceiling %.1fx)",
            row["base_rate"],
            row["flag_rate"],
            row["recall"],
            row["lift"],
            row["ceiling"],
        )

    write_json_artifact(
        config.results_path("economics.json"), {"economics": economics}, config
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
