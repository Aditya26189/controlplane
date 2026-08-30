"""Stage 4: sweep layers x C on validation, then score the test set once.

Parses arguments, calls ``src/``, writes files. No logic lives here.

Order of operations is fixed by TASKS.md Stage 4 and is the point of this
script: sweep on validation, pick the winner on validation, freeze the
threshold on validation, and only then open the test set -- at exactly one
place, marked below.

Outputs:
    results/probe_sweep.json    every (layer, C) tried, with validation AUROC
    results/probe.joblib        the fitted probe, threshold frozen
    results/probe_test.json     the single test scoring, with bootstrap CIs
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    load_config,
    provenance,
    set_seeds,
    setup_logging,
    write_json_artifact,
)
from src.evaluate import (  # noqa: E402
    abstention_analysis,
    append_test_scoring,
    auroc_floor_status,
    bootstrap_metrics,
    evaluate_at_threshold,
    roc_points,
)
from src.extract import load_activations  # noqa: E402
from src.probe import (  # noqa: E402
    assert_polarity,
    fit_selected_probe,
    run_sweep,
    save_probe,
)

LOGGER = logging.getLogger("02_train_probe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)
    LOGGER.info("config hash %s, seed %d", config.config_hash, config.seed)

    labels_frame = pd.read_parquet(config.paths.labels)
    activations = load_activations(
        config.paths.activations,
        expected_question_ids=labels_frame["question_id"].tolist(),
    )
    labels = labels_frame["label"].to_numpy().astype(int)
    split = labels_frame["split"].to_numpy()

    # Polarity is checked here, at the boundary between extraction and the
    # probe: 1 == the generated answer was incorrect (DECISIONS.md 004).
    assert_polarity(labels, labels_frame["correct"].to_numpy())
    LOGGER.info(
        "loaded %d examples across %d layers; base rate (incorrect) %.4f",
        len(labels_frame),
        len(activations),
        float(labels.mean()),
    )

    # ---- 1-2. sweep and select, on validation only -------------------------- #
    sweep = run_sweep(activations, labels, split, config)
    best = sweep["best"]

    LOGGER.info("layer sweep (validation AUROC), full table:")
    table = pd.DataFrame(sweep["sweep"]).pivot(
        index="layer", columns="C", values="val_auroc"
    )
    for line in table.round(4).to_string().splitlines():
        LOGGER.info("  %s", line)

    # SPEC.md §5: a winner at the edge of the grid means the search stopped at
    # the boundary rather than at an optimum. Reported loudly, on validation
    # evidence only, so the grid can be widened before anything is published.
    grid = sorted(config.probe.C_grid)
    at_boundary = best["C"] in (grid[0], grid[-1]) and len(grid) > 1
    if at_boundary:
        LOGGER.warning(
            "selected C=%g is at the %s edge of the grid %s. The search found a "
            "boundary, not an optimum -- widen probe.C_grid and re-select on "
            "validation (SPEC.md §5, DECISIONS.md 016).",
            best["C"],
            "lower" if best["C"] == grid[0] else "upper",
            grid,
        )

    write_json_artifact(
        config.results_path("probe_sweep.json"),
        {
            "sweep": sweep["sweep"],
            "best": best,
            "layer_fractions": list(config.model.layer_fractions),
            "C_grid": list(config.probe.C_grid),
            "selected_on": "validation",
            "winner_at_grid_boundary": bool(at_boundary),
            "layers_selecting_smallest_C": sorted(
                {
                    row["layer"]
                    for row in sweep["sweep"]
                    if row["C"] == grid[0]
                    and row["val_auroc"]
                    == max(
                        r["val_auroc"]
                        for r in sweep["sweep"]
                        if r["layer"] == row["layer"]
                    )
                }
            ),
        },
        config,
    )

    # ---- 3. freeze the threshold, on validation only ------------------------ #
    probe = fit_selected_probe(activations, labels, split, best, config)
    save_probe(probe, config.results_path("probe.joblib"))
    LOGGER.info(
        "probe: layer %d, C=%g, threshold %.6f, val AUROC %.4f, val flag rate %.4f",
        probe.layer,
        probe.c_value,
        probe.threshold,
        probe.val_auroc,
        probe.val_flag_rate,
    )

    # ======================================================================== #
    # 4. THE TEST SET IS OPENED HERE. THIS IS THE ONLY PLACE.                  #
    #                                                                          #
    # Everything above this line saw train and validation only. Nothing below  #
    # feeds back into layer choice, C, or the threshold -- those are frozen in #
    # `probe` already (CLAUDE.md invariant 2). Every scoring is appended to     #
    # results/test_scoring_log.json so the count is auditable, not asserted.    #
    # ======================================================================== #
    test_mask = split == "test"
    if not test_mask.any():
        raise AssertionError("no test rows; the split assignment is wrong")
    x_test = activations[probe.layer][test_mask]
    y_test = labels[test_mask]
    test_scores = probe.score(x_test)

    point = evaluate_at_threshold(y_test, test_scores, probe.threshold)
    intervals = bootstrap_metrics(
        y_test,
        test_scores,
        probe.threshold,
        config.evaluation.bootstrap_samples,
        config.evaluation.ci,
        config.seed,
    )
    floor = auroc_floor_status(point["auroc"], config)
    abstention = abstention_analysis(
        test_scores, labels_frame.loc[test_mask, "abstained"].tolist(), config
    )

    LOGGER.info(
        "TEST  n=%d  AUROC %.4f [%.4f, %.4f]  f=%.4f  R=%.4f  precision=%.4f  "
        "base rate=%.4f",
        point["n"],
        point["auroc"],
        intervals["auroc"]["ci_low"],
        intervals["auroc"]["ci_high"],
        point["flag_rate"],
        point["recall"],
        point["precision"] if point["precision"] is not None else float("nan"),
        point["base_rate"],
    )

    write_json_artifact(
        config.results_path("probe_test.json"),
        {
            "probe": probe.to_meta(),
            "test": point,
            "bootstrap": intervals,
            "auroc_floor": floor,
            "abstention": abstention,
            "roc": roc_points(y_test, test_scores, probe.threshold),
            "strict_em": {
                # None, not NaN: the column is all-NaN when
                # labeling.record_strict_em is off, and NaN is not valid JSON.
                "test_accuracy_strict": (
                    None
                    if labels_frame.loc[test_mask, "exact_match"].isna().all()
                    else float(labels_frame.loc[test_mask, "exact_match"].mean())
                ),
                "test_accuracy_lenient": float(
                    labels_frame.loc[test_mask, "correct"].mean()
                ),
            },
            "split_sizes": {
                name: int((split == name).sum()) for name in ("train", "val", "test")
            },
            "test_scores_summary": {
                "min": float(np.min(test_scores)),
                "median": float(np.median(test_scores)),
                "max": float(np.max(test_scores)),
            },
        },
        config,
    )

    # Append-only record of every test scoring (DECISIONS.md 016).
    scoring_log = append_test_scoring(
        config.results_path("test_scoring_log.json"),
        {
            "timestamp_utc": provenance(config)["timestamp_utc"],
            "config_hash": config.config_hash,
            "git_commit": provenance(config)["git_commit"],
            "selected_layer": probe.layer,
            "selected_C": probe.c_value,
            "C_grid": list(config.probe.C_grid),
            "threshold": probe.threshold,
            "auroc": point["auroc"],
            "flag_rate": point["flag_rate"],
            "recall": point["recall"],
            "precision": point["precision"],
            "lift": point["lift"],
            "n_test": point["n"],
        },
    )
    write_json_artifact(
        config.results_path("test_scoring_log.json"), scoring_log, config
    )
    if scoring_log["n_scorings"] > 1:
        LOGGER.warning(
            "the test set has now been scored %d times. This is disclosed in "
            "RESULTS.md; every scoring stays in the log (DECISIONS.md 016).",
            scoring_log["n_scorings"],
        )

    if floor["below_floor"]:
        LOGGER.warning(
            "AUROC is at or below the floor. Report this as measured; do not "
            "tune on test (TASKS.md Stage 4)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
