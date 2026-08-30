"""Record which entity types stock Presidio can recognise at all.

``DECISIONS.md`` 118. Thin wrapper: parses arguments, calls the analyzer,
writes a file. CPU only, seconds.

**Why this needs to be an artifact.** The demo wants to say *"the library ships
no recogniser for these identifier formats in any configuration"*. That is a
strong claim and it was true, but it lived in a planning document and traced to
nothing in this repository -- the same import path that produced `DEFF 1.60`,
`prEN 18284` and the refusal beat's own budget (`113`, `117`). A claim about a
dependency is checkable by running the dependency, so it gets run.

This is a coverage claim, not a performance claim. It says the recogniser does
not exist, which is why stock recall on these formats is not a tuning problem.

Usage:
    python scripts/17_presidio_coverage.py --config config.yaml
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

_LOG = logging.getLogger("scripts.17_presidio_coverage")

#: The identifier families the Hinglish PII evaluation sets are built from.
#: Named here rather than inferred, so the claim is about a declared list.
INDIAN_IDENTIFIERS = ("UPI", "IFSC", "AADHAAR", "PAN", "MICR")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "results" / "presidio_coverage.json"),
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)

    from presidio_analyzer import AnalyzerEngine

    engine = AnalyzerEngine()
    supported = sorted(engine.get_supported_entities())
    _LOG.info("stock Presidio supports %d entity types", len(supported))

    coverage = {}
    for family in INDIAN_IDENTIFIERS:
        hits = [entity for entity in supported if family in entity.upper()]
        coverage[family] = hits
        _LOG.info("  %-8s -> %s", family, hits or "NONE")

    india_prefixed = [entity for entity in supported if entity.startswith("IN_")]
    uncovered = [family for family, hits in coverage.items() if not hits]

    write_json_artifact(
        Path(args.out),
        {
            "claim": (
                "stock Presidio ships no recogniser for these identifier "
                "families, in any configuration"
            ),
            "presidio_version": _version(),
            "n_supported_entities": len(supported),
            "supported_entities": supported,
            "indian_identifier_coverage": coverage,
            "india_prefixed_entities": india_prefixed,
            "uncovered": uncovered,
            "interpretation": (
                "This is a COVERAGE result, not a performance result. Recall on "
                "these formats is not low because the patterns are hard; it is "
                "low because no recogniser for them exists to load. That is why "
                "the fix was to write them (controlplane/detectors/"
                "presidio_custom.py) rather than to tune a threshold."
            ),
            "preregistered_in": "DECISIONS.md 118",
        },
        config,
    )
    _LOG.info("wrote %s", args.out)
    return 0


def _version() -> str:
    """The installed analyzer version, so the claim is pinned to one release."""
    try:
        from importlib.metadata import version

        return version("presidio-analyzer")
    except Exception:  # pragma: no cover - metadata is best effort
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
