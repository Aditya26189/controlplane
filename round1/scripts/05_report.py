"""Stage 5c: render results/ into RESULTS.md, README.md, and two plots.

Parses arguments, calls ``src/``, writes files. No logic lives here.

Every number in both documents is read from a JSON artifact. Prose in the
README template can be edited by hand; numbers cannot (CLAUDE.md,
"Documentation"). If a number is wrong, the pipeline is wrong.

Outputs:
    results/RESULTS.md
    results/layer_sweep.png
    results/roc_curve.png
    README.md              (unless --no-readme)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, setup_logging  # noqa: E402
from src.report import (  # noqa: E402
    load_artifacts,
    plot_layer_sweep,
    plot_roc,
    render_readme,
    render_results_md,
)

LOGGER = logging.getLogger("05_report")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument(
        "--template",
        default="README_TEMPLATE.md",
        help="source template for README.md",
    )
    parser.add_argument(
        "--readme",
        default="README.md",
        help="destination for the rendered README",
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="render RESULTS.md and the plots only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)

    artifacts = load_artifacts(config)

    results_md = render_results_md(artifacts, config)
    results_path = config.results_path("RESULTS.md")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(results_md, encoding="utf-8")
    LOGGER.info("wrote %s (%d lines)", results_path, len(results_md.splitlines()))

    plot_layer_sweep(artifacts["probe_sweep"], config.results_path("layer_sweep.png"))
    plot_roc(artifacts["probe_test"], config.results_path("roc_curve.png"))

    if not args.no_readme:
        template = Path(args.template).read_text(encoding="utf-8")
        readme = render_readme(template, artifacts, config)
        Path(args.readme).write_text(readme, encoding="utf-8")
        LOGGER.info("wrote %s", args.readme)

    economics = artifacts["economics"]["economics"]
    test = artifacts["probe_test"]["test"]
    boot = artifacts["probe_test"]["bootstrap"]
    LOGGER.info(
        "HEADLINE  lift = R/f = %.2fx [%.2f, %.2f]   (R=%.4f, f=%.4f, AUROC=%.4f)",
        economics["lift"],
        boot["lift"]["ci_low"],
        boot["lift"]["ci_high"],
        test["recall"],
        test["flag_rate"],
        test["auroc"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
