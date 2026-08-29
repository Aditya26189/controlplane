"""The abstention floor and the review sizing, derived from what is measured.

Thin wrapper: parses arguments, calls ``controlplane/``, writes files. No logic
(``CLAUDE.md``).

Two things the proposal needs and does not have to declare as estimates:

1. **The abstention floor.** Given the measured base error rate of an envelope
   and a declared target risk, the minimum fraction of traffic that *any*
   selector must abstain on. An impossibility result — it answers "why not just
   tighten the threshold" without appealing to how good our detector is.

2. **Review sizing.** What each profile's measured operating point costs in
   items per month against one declared workload, and how many reviewed items a
   recall claim needs to reach a declared margin.

Neither is a cost figure. The price list is still not built; see
``DECISIONS.md`` 096 and 099.

Usage:
    python scripts/11_feasibility.py --config config.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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
from controlplane.economics import (  # noqa: E402
    achieved_risk,
    feasibility_curve,
    recall_sample_size,
    review_volume,
)
from controlplane.report.claims import resolve  # noqa: E402

_LOG = logging.getLogger("scripts.11_feasibility")

#: Declared risk appetites the floor is reported at. A curve, not a point:
#: alpha is a choice, and the shape is what carries the argument.
TARGET_RISKS = (0.20, 0.10, 0.05, 0.02, 0.01)

#: Where the measured base rate and the three operating points come from.
POLICY_ARTIFACT = "results/policy-triviaqa-2400-t960.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "results" / "feasibility.json")
    )
    parser.add_argument("--policy-artifact", default=POLICY_ARTIFACT)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)

    artifact = PROJECT_ROOT / args.policy_artifact
    if not artifact.is_file():
        _LOG.error("no %s; nothing measured to derive from", args.policy_artifact)
        return 1
    document = json.loads(artifact.read_text(encoding="utf-8"))

    first = document["operating_points"][0]
    base_rate = float(first["base_rate"])
    envelope_id = first["envelope_id"]
    eval_set_id = first["eval_set_id"]
    _LOG.info(
        "measured base error rate %.4f on %s [%s]", base_rate, eval_set_id, envelope_id
    )

    floors = feasibility_curve(
        base_rate, TARGET_RISKS, envelope_id=envelope_id, source=args.policy_artifact
    )
    for floor in floors:
        _LOG.info("  %s", floor.render())

    profiles = []
    for name, profile in config.profiles.items():
        point = profile.operating_point
        path = f"operating_points[operating_point.operating_point_id={point}]"
        try:
            op = resolve(document, path)
        except KeyError:
            _LOG.warning("no measured operating point %s; skipping %s", point, name)
            continue
        recall = float(op["metrics"]["recall"]["value"])
        flag_rate = float(op["metrics"]["flag_rate"]["value"])

        volume = review_volume(
            flag_rate=flag_rate,
            recall=recall,
            base_error_rate=config.workload.base_error_rate,
            monthly_interactions=config.workload.monthly_interactions,
            operating_point_id=point,
        )
        sizes = [
            recall_sample_size(
                recall=recall,
                margin=margin,
                design_effect=config.sampling.expected_design_effect,
                operating_point_id=point,
            )
            for margin in config.sampling.recall_margin_tiers
        ]
        achieved = achieved_risk(
            operating_point_id=point,
            base_error_rate=base_rate,
            recall=recall,
            flag_rate=flag_rate,
        )
        _LOG.info("%s @ %s", name, point)
        _LOG.info("    %s", achieved.render())
        for sizing in list(volume.values()) + sizes:
            for line in sizing.render().splitlines():
                _LOG.info("    %s", line)

        profiles.append(
            {
                "profile": name,
                "operating_point_id": point,
                "measured_recall": recall,
                "measured_flag_rate": flag_rate,
                "achieved_risk": dataclasses.asdict(achieved),
                "review_volume": {
                    k: dataclasses.asdict(v) for k, v in volume.items()
                },
                "recall_sample_size": [dataclasses.asdict(s) for s in sizes],
            }
        )

    payload = {
        "measured": {
            "base_error_rate": base_rate,
            "eval_set_id": eval_set_id,
            "envelope_id": envelope_id,
            "source": args.policy_artifact,
        },
        "declared": {
            "workload": config.workload.name,
            "monthly_interactions": config.workload.monthly_interactions,
            "production_base_error_rate": config.workload.base_error_rate,
            "target_risks": list(TARGET_RISKS),
            "recall_margin_tiers": list(config.sampling.recall_margin_tiers),
            "expected_design_effect": config.sampling.expected_design_effect,
        },
        "abstention_floor": [dataclasses.asdict(f) for f in floors],
        "profiles": profiles,
        "not_derived_here": (
            "No cost, headcount or saving figure. The price list "
            "(controlplane/economics/sizing.py) is specified in five contracts "
            "and not built -- DECISIONS 096 and 099. Any money figure in the "
            "proposal or the deck is a declared estimate and must say so."
        ),
    }
    write_json_artifact(Path(args.out), payload, config)
    _LOG.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
