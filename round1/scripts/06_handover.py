"""Stage 6 (documentation): render docs/HANDOVER.md from results/.

Parses arguments, calls ``src/``, writes files. No logic lives here.

The handover complements ``results/RESULTS.md`` rather than repeating it.
RESULTS.md is the formal record in SPEC.md §13's order; this is orientation for
someone joining cold who may have to present the work — what the numbers mean,
what bounds them, and which framings to avoid.

Output:
    docs/HANDOVER.md
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, setup_logging  # noqa: E402
from src.report import load_artifacts, render_handover_md  # noqa: E402

LOGGER = logging.getLogger("06_handover")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument(
        "--out",
        default="docs/HANDOVER.md",
        help="destination for the rendered handover",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)

    artifacts = load_artifacts(config)
    document = render_handover_md(artifacts, config)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    LOGGER.info("wrote %s (%d lines)", out, len(document.splitlines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
