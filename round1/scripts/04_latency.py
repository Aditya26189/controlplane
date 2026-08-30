"""Stage 5b: measure the probe's wall-clock cost against generation's.

Parses arguments, calls ``src/``, writes files. No logic lives here.

The probe is timed on a real activation vector taken from the extraction run,
one at a time, because the deployed cost is per response. Generation and
prefill times come from ``extract_meta.json`` -- the same calls that produced
the labels, not a fresh benchmark.

Output:
    results/latency.json
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    load_config,
    read_json_artifact,
    set_seeds,
    setup_logging,
    write_json_artifact,
)
from src.extract import load_activations  # noqa: E402
from src.latency import measure  # noqa: E402
from src.probe import load_probe  # noqa: E402

LOGGER = logging.getLogger("04_latency")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)

    probe = load_probe(config.results_path("probe.joblib"))
    extract_meta = read_json_artifact(config.results_path("extract_meta.json"))
    labels_frame = pd.read_parquet(config.paths.labels)
    activations = load_activations(
        config.paths.activations,
        expected_question_ids=labels_frame["question_id"].tolist(),
    )

    # A real vector from the run, not a random one: the timing should reflect
    # the actual dtype and memory layout the probe sees in practice.
    activation = activations[probe.layer][0]

    latency = measure(
        probe,
        activation,
        median_generate_seconds=extract_meta["extraction"][
            "median_generate_seconds_per_response"
        ],
        median_prefill_seconds=extract_meta["extraction"][
            "median_prefill_seconds_per_response"
        ],
        config=config,
    )

    comparison = latency["comparison"]
    LOGGER.info(
        "probe %.1f us (p95 %.1f) vs generation %.1f ms per response",
        comparison["probe_median_us"],
        comparison["probe_p95_us"],
        comparison["generation_median_ms"],
    )
    LOGGER.info(
        "ratio probe/generation = %.2e (%.1f orders of magnitude); "
        "probe adds no additional forward pass",
        comparison["probe_over_generation"],
        comparison["orders_of_magnitude"],
    )

    write_json_artifact(
        config.results_path("latency.json"), {"latency": latency}, config
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
