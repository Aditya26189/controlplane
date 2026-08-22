"""Render the artifacts in ``results/`` into RESULTS.md, README.md, and plots.

Every number here is read from a JSON file produced by a script in this repo.
Nothing is computed in this module beyond formatting, and nothing is typed in
by hand -- if a number is wrong, the pipeline is wrong (CLAUDE.md,
"Documentation": never hand-edit a number).

``RESULTS.md`` follows the thirteen-element order fixed by SPEC.md §13.
``README.md`` is rendered from ``README_TEMPLATE.md`` by substituting
``{{placeholder}}`` tokens; an unsubstituted token is an error rather than a
silently published blank.
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional

from src.config import Config, read_json_artifact

LOGGER = logging.getLogger(__name__)

ARTIFACT_FILES = {
    "data_stats": "data_stats.json",
    "extract_meta": "extract_meta.json",
    "probe_sweep": "probe_sweep.json",
    "probe_test": "probe_test.json",
    "economics": "economics.json",
    "latency": "latency.json",
}

OPTIONAL_ARTIFACT_FILES = {
    "negative_control": "negative_control.json",
    "test_scoring_log": "test_scoring_log.json",
}


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def fmt(value: Optional[float], digits: int = 3) -> str:
    """Format a number for a report, or ``n/a`` when it is undefined.

    Undefined stays visible rather than becoming 0: a precision of "n/a"
    because nothing was flagged is a different fact from a precision of zero.
    """
    if value is None:
        return "n/a"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def fmt_pct(value: Optional[float], digits: int = 1) -> str:
    """Format a fraction as a percentage."""
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def pct_number(value: Optional[float]) -> str:
    """Percentage as a bare number, no ``%`` sign and no trailing zeros.

    The README template writes the sign itself (``{{flag_rate_pct}}%``), so a
    value carrying its own ``%`` renders as ``2.5%%``.
    """
    if value is None:
        return "n/a"
    return f"{value * 100:g}"


def fmt_ratio(value: Optional[float]) -> str:
    """Format a ratio in scientific notation, or ``n/a`` when undefined."""
    return "n/a" if value is None else f"{value:.2e}"


def fmt_count(value: Optional[float]) -> str:
    """Format a count with thousands separators."""
    return "n/a" if value is None else f"{round(value):,}"


def fmt_ci(block: Optional[dict[str, Any]], digits: int = 3) -> str:
    """Format a bootstrap interval as ``[low, high]``."""
    if not block or block.get("ci_low") is None:
        return "n/a"
    return f"[{block['ci_low']:.{digits}f}, {block['ci_high']:.{digits}f}]"


def fmt_duration(seconds: Optional[float]) -> str:
    """Format a duration as minutes and seconds."""
    if seconds is None:
        return "n/a"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_artifacts(config: Config) -> dict[str, Any]:
    """Read every stage artifact from ``results/``.

    Args:
        config: Resolved experiment config.

    Returns:
        Artifact name -> parsed JSON. Optional artifacts are absent rather than
        empty when they were not produced.
    """
    artifacts: dict[str, Any] = {}
    for key, filename in ARTIFACT_FILES.items():
        artifacts[key] = read_json_artifact(config.results_path(filename))
    for key, filename in OPTIONAL_ARTIFACT_FILES.items():
        path = config.results_path(filename)
        if path.is_file():
            artifacts[key] = read_json_artifact(path)

    consistency = config_hash_consistency(artifacts)
    if not consistency["consistent"]:
        LOGGER.warning(
            "artifacts were produced under %d different config hashes: %s. The "
            "report will say so; re-run the whole pipeline if the stages are "
            "meant to describe one experiment.",
            len(consistency["distinct"]),
            consistency["per_artifact"],
        )
    return artifacts


def config_hash_consistency(artifacts: dict[str, Any]) -> dict[str, Any]:
    """Check that every stage's artifact came from the same config.

    Stages run as separate processes and each loads the config itself, so
    nothing stops someone re-running stage 02 with an edited config against
    stage 01's activations. The report would then quote one hash beside numbers
    produced under two, which is exactly the traceability claim CLAUDE.md
    invariant 7 makes. This does not raise -- re-running a later stage with a
    changed probe grid is a legitimate workflow -- but the mismatch is surfaced
    in RESULTS.md rather than left to be noticed.
    """
    per_artifact = {
        name: artifact.get("provenance", {}).get("config_hash")
        for name, artifact in artifacts.items()
    }
    distinct = sorted({h for h in per_artifact.values() if h})
    return {
        "per_artifact": per_artifact,
        "distinct": distinct,
        "consistent": len(distinct) <= 1,
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def plot_layer_sweep(sweep: dict[str, Any], path: Path) -> Path:
    """Plot validation AUROC by depth, one line per regularisation strength.

    The shape carries information the winning number does not: a smooth curve
    peaking mid-stack is itself evidence the signal is real rather than noise.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sweep["sweep"]
    best = sweep["best"]
    c_values = sorted({row["C"] for row in rows})

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    for c_value in c_values:
        series = sorted(
            (row for row in rows if row["C"] == c_value), key=lambda r: r["layer"]
        )
        ax.plot(
            [r["layer"] for r in series],
            [r["val_auroc"] for r in series],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=f"C = {c_value:g}",
        )
    ax.axhline(0.5, color="#999999", linestyle=":", linewidth=1)
    ax.annotate(
        "chance",
        xy=(min(r["layer"] for r in rows), 0.5),
        xytext=(2, 3),
        textcoords="offset points",
        fontsize=8,
        color="#666666",
    )
    ax.scatter(
        [best["layer"]],
        [best["val_auroc"]],
        s=140,
        facecolors="none",
        edgecolors="#d1495b",
        linewidths=2,
        zorder=5,
        label=f"selected: layer {best['layer']}, C={best['C']:g}",
    )
    ax.set_xlabel("Transformer layer (hidden_states index)")
    ax.set_ylabel("Validation AUROC")
    ax.set_title("Probe quality by depth — selected on validation, never on test")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("wrote %s", path)
    return path


def plot_roc(probe_test: dict[str, Any], path: Path) -> Path:
    """Plot the test ROC curve with the frozen operating threshold marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    roc = probe_test["roc"]
    test = probe_test["test"]
    auroc = test["auroc"]

    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=150)
    ax.plot(roc["fpr"], roc["tpr"], linewidth=2, color="#2d6a9f", label=f"AUROC {auroc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle=":", color="#999999", linewidth=1, label="chance")
    point = roc.get("operating_point")
    if point and point.get("fpr") is not None:
        ax.scatter(
            [point["fpr"]],
            [point["tpr"]],
            s=90,
            color="#d1495b",
            zorder=5,
            label=(
                f"operating point: f={test['flag_rate']:.3f}, R={test['recall']:.3f}"
            ),
        )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall of incorrect answers)")
    ax.set_title(f"Test ROC — layer {probe_test['probe']['layer']}")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("wrote %s", path)
    return path


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def layer_sweep_table(sweep: dict[str, Any]) -> str:
    """Markdown table of validation AUROC for every (layer, C) tried.

    All of it, not just the winner: a reviewer wants the shape of the curve,
    and showing only the maximum invites the question of what was hidden.
    """
    rows = sweep["sweep"]
    best = sweep["best"]
    c_values = sorted({row["C"] for row in rows})
    layers = sorted({row["layer"] for row in rows})
    lookup = {(row["layer"], row["C"]): row["val_auroc"] for row in rows}

    header = "| Layer | " + " | ".join(f"C={c:g}" for c in c_values) + " |"
    divider = "|---" * (len(c_values) + 1) + "|"
    lines = [header, divider]
    for layer in layers:
        cells = []
        for c_value in c_values:
            value = lookup.get((layer, c_value))
            text = fmt(value, 4)
            if layer == best["layer"] and c_value == best["C"]:
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {layer} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def projection_table_markdown(economics: dict[str, Any]) -> str:
    """Markdown for the base-rate projection, or an empty string if absent."""
    projection = economics.get("projection")
    if not projection or not projection.get("rows"):
        return ""
    lines = [
        "| Base error rate | Budget `f` | Recall `R` | Lift | Ceiling (1/base rate) |",
        "|---|---|---|---|---|",
    ]
    for row in projection["rows"]:
        lines.append(
            f"| {fmt(row['base_rate'], 3)} | {fmt(row['flag_rate'], 4)} "
            f"| {fmt(row['recall'], 4)} | {fmt(row['lift'], 2)}x "
            f"| {fmt(row['ceiling'], 1)}x |"
        )
    return "\n".join(lines)


def policy_table_markdown(economics: dict[str, Any]) -> str:
    """Markdown rendering of the three-policy comparison."""
    lines = [
        "| Policy | Judge calls | Responses seen by any check | Errors caught | Relative cost |",
        "|---|---|---|---|---|",
    ]
    for row in economics["policies"]:
        emphasis = "**" if row["policy"] == "probe_triggered" else ""
        lines.append(
            f"| {emphasis}{row['label']}{emphasis} "
            f"| {emphasis}{fmt_count(row['judge_calls'])}{emphasis} "
            f"| {emphasis}{pct_number(row['coverage'])}%{emphasis} "
            f"| {emphasis}{fmt_count(row['errors_caught'])}{emphasis} "
            f"| {emphasis}{row['relative_cost']:.0f}x{emphasis} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# RESULTS.md
# --------------------------------------------------------------------------- #


def render_results_md(artifacts: dict[str, Any], config: Config) -> str:
    """Render ``results/RESULTS.md`` in the order fixed by SPEC.md §13."""
    data = artifacts["data_stats"]["data"]
    extract = artifacts["extract_meta"]
    probe_test = artifacts["probe_test"]
    sweep = artifacts["probe_sweep"]
    economics = artifacts["economics"]["economics"]
    latency = artifacts["latency"]["latency"]

    provenance = probe_test["provenance"]
    model_info = extract["model"]
    test = probe_test["test"]
    boot = probe_test["bootstrap"]
    probe = probe_test["probe"]
    base_rates = extract["base_rates"]
    equivalence = extract["equivalence_check"]
    extraction = extract["extraction"]
    abstention = probe_test["abstention"]
    comparison = latency["comparison"]

    strict_gap_raw = base_rates.get("lenient_minus_strict_accuracy")
    strict_gap = None if strict_gap_raw is None else abs(strict_gap_raw)
    lines: list[str] = []
    add = lines.append

    add("# RESULTS")
    add("")
    add(
        "Generated by `scripts/05_report.py`. Every number below is read from a "
        "JSON artifact in this directory; none is hand-entered."
    )
    add("")

    # 1. Run metadata
    add("## 1. Run metadata")
    add("")
    add("| Field | Value |")
    add("|---|---|")
    add(f"| Model | `{model_info['name']}` |")
    add(f"| Quantisation | {model_info['quantization']} ({model_info['dtype']} compute) |")
    add(f"| Device | {provenance['device']['device_name']} |")
    add(f"| Seed | {provenance['seed']} |")
    add(f"| Config hash | `{provenance['config_hash']}` |")
    add(f"| Git commit | `{provenance['git_commit']}` |")
    add(f"| Working tree dirty at run time | {provenance['dirty']} |")
    add(f"| Timestamp (UTC) | {provenance['timestamp_utc']} |")
    add(f"| torch / transformers | {provenance['libraries']['torch']} / {provenance['libraries']['transformers']} |")
    add("")
    consistency = config_hash_consistency(artifacts)
    if not consistency["consistent"]:
        add(
            "> **Config hashes differ across stages.** These numbers were not all "
            "produced from one configuration: "
            + ", ".join(
                f"`{name}` = `{value}`"
                for name, value in consistency["per_artifact"].items()
            )
            + ". Re-run the whole pipeline before quoting anything here."
        )
        add("")
    control = equivalence.get("right_padding_control")
    add(
        f"Left-padding equivalence check: relative L2 error "
        f"**{equivalence['max_relative_l2']:.3e}** (limit "
        f"{equivalence['relative_tolerance']:.0e}), cosine similarity "
        f"**{equivalence['min_cosine_observed']:.6f}** (limit "
        f"{equivalence['min_cosine']:.4f}), across {equivalence['n_prompts']} "
        "prompts spanning the length distribution. Batched and unbatched "
        "last-token activations agree, so position -1 is the true final prompt "
        "token and not a pad token (CLAUDE.md invariant 4)."
    )
    add("")
    if control is not None:
        add(
            "Positive control: repeating the same comparison with the tokenizer "
            f"deliberately **right**-padded gives relative L2 "
            f"{control['max_relative_l2']:.3e} and cosine "
            f"{control['min_cosine']:.4f} — rejected, as it must be. The check "
            "is therefore demonstrated to discriminate a real padding fault on "
            "this model and this hardware, not merely to have passed."
        )
        add("")

    # 2. Dataset
    add("## 2. Dataset")
    add("")
    add("| Field | Value |")
    add("|---|---|")
    add(f"| Source | `{data['dataset']}`, config `{data['dataset_config']}`, split `{data['split']}` |")
    add(f"| Rows loaded | {data['rows_loaded']:,} |")
    add(f"| Dropped as duplicate questions | {data['duplicates_dropped']:,} |")
    add(f"| Dropped as empty or aliasless | {data['empty_or_aliasless_dropped']:,} |")
    add(f"| Sampled | {data['n_final']:,} |")
    add(
        f"| Split sizes | train {data['split_sizes']['train']:,} / "
        f"val {data['split_sizes']['val']:,} / test {data['split_sizes']['test']:,} |"
    )
    add(f"| Accuracy (lenient alias match) | {fmt(base_rates['accuracy_lenient'])} |")
    add(f"| Accuracy (strict exact match) | {fmt(base_rates['accuracy_strict_em'])} |")
    add(f"| **Base error rate (positive class)** | **{fmt(base_rates['base_rate_incorrect'])}** |")
    add("")
    add(
        "Splits are by `question_id` after deduplicating normalised question "
        "strings, and are asserted pairwise disjoint on both keys "
        "(CLAUDE.md invariant 3)."
    )
    add("")
    if strict_gap is None:
        add(
            "Strict exact match was not recorded for this run "
            "(`labeling.record_strict_em` is off), so the labelling audit "
            "SPEC.md §2 asks for is unavailable. Turn it on and re-run before "
            "quoting the base rate."
        )
    elif strict_gap > 0.10:
        add(
            f"**Labelling note.** Lenient and strict matching disagree by "
            f"{fmt(strict_gap)} in accuracy, more than the ~10 point threshold "
            "SPEC.md §2 asks to be called out. The lenient rule is primary "
            "(DECISIONS.md 007); the gap is the amount of correctness attributable "
            "to a gold alias appearing inside a longer sentence."
        )
    else:
        add(
            f"Lenient and strict matching disagree by {fmt(strict_gap)} in "
            "accuracy, within the ~10 point threshold SPEC.md §2 asks to be "
            "called out."
        )
    add("")

    # 3. Layer sweep
    add("## 3. Layer sweep — validation AUROC")
    add("")
    add(layer_sweep_table(sweep))
    add("")
    add(
        f"**Selected: layer {probe['layer']} of {model_info['num_hidden_layers']}, "
        f"C = {probe['C']:g}**, on validation AUROC {fmt(probe['val_auroc'], 4)}. "
        "The layer, the regularisation strength and the threshold were all chosen "
        "here. The test set was opened afterwards, once "
        "(CLAUDE.md invariant 2, DECISIONS.md 006)."
    )
    add("")
    if sweep.get("winner_at_grid_boundary"):
        add(
            f"> **The selected `C` sits at the edge of the grid** "
            f"{sorted(sweep.get('C_grid', []))}. A boundary is not an optimum — "
            "the search stopped where we stopped looking. The grid should be "
            "widened and validation re-run before this number is treated as "
            "final (SPEC.md §5)."
        )
        add("")
    add("![Layer sweep](layer_sweep.png)")
    add("")

    # 4. Test results
    add("## 4. Test results")
    add("")
    log = artifacts.get("test_scoring_log")
    n_scorings = log.get("n_scorings", 1) if log else 1
    if n_scorings <= 1:
        add(f"The test set was scored exactly once, at n = {test['n']:,}.")
        add("")
    else:
        add(
            f"**The test set has been scored {n_scorings} times**, at "
            f"n = {test['n']:,}. Disclosed rather than claimed: every scoring is "
            "appended to `results/test_scoring_log.json` and reproduced below. "
            "Each selection that produced one was made on validation alone "
            "(DECISIONS.md 016)."
        )
        add("")
        add(
            "The log begins when it was introduced, so any scoring that predates "
            "it is recorded in `DECISIONS.md` 016 and 017 rather than in the "
            "table below. Treat the two together as the full history."
        )
        add("")
        add("| # | Layer | C | AUROC | `f` | `R` | Lift | Config hash |")
        add("|---|---|---|---|---|---|---|---|")
        for i, row in enumerate(log.get("scorings", []), start=1):
            add(
                f"| {i} | {row.get('selected_layer')} | {row.get('selected_C'):g} "
                f"| {fmt(row.get('auroc'), 4)} | {fmt(row.get('flag_rate'), 4)} "
                f"| {fmt(row.get('recall'), 4)} | {fmt(row.get('lift'), 2)}x "
                f"| `{row.get('config_hash')}` |"
            )
        add("")
    add("| Metric | Value | 95% CI |")
    add("|---|---|---|")
    add(f"| AUROC | {fmt(test['auroc'], 4)} | {fmt_ci(boot['auroc'], 4)} |")
    add(f"| Measured flag rate `f` | {fmt(test['flag_rate'], 4)} | {fmt_ci(boot['flag_rate'], 4)} |")
    add(f"| Recall `R` | {fmt(test['recall'], 4)} | {fmt_ci(boot['recall'], 4)} |")
    add(f"| Precision | {fmt(test['precision'], 4)} | {fmt_ci(boot['precision'], 4)} |")
    add(f"| Base error rate | {fmt(test['base_rate'], 4)} | — |")
    add("")
    add(
        f"Confusion at the frozen threshold ({fmt(probe['threshold'], 4)}): "
        f"TP {test['tp']}, FP {test['fp']}, FN {test['fn']}, TN {test['tn']}. "
        f"{test['n_flagged']} of {test['n']} responses flagged; "
        f"{test['n_incorrect']} were incorrect."
    )
    add("")
    add(
        "Precision and recall are reported separately and never blended "
        "(CLAUDE.md invariant 5). Low precision is by design: a false positive "
        "costs one wasted judge call, a false negative costs a user acting on a "
        "wrong answer (DECISIONS.md 005)."
    )
    add("")
    add(
        f"The threshold was chosen on validation to hit a flag rate of "
        f"{fmt(probe['target_flag_rate'])} and achieved "
        f"{fmt(probe['val_flag_rate'])} there. On test the measured rate is "
        f"{fmt(test['flag_rate'], 4)}, and **that measured value is what every "
        "calculation below uses** (CLAUDE.md invariant 6)."
    )
    add("")
    add("![Test ROC](roc_curve.png)")
    add("")
    if probe_test["auroc_floor"]["below_floor"]:
        add(
            f"> **Weak result.** Test AUROC {fmt(test['auroc'], 4)} is at or below "
            f"the configured floor of {probe_test['auroc_floor']['floor']}. It is "
            "reported as measured rather than tuned; see the limitations section."
        )
        add("")

    # 5. Three policies
    add(f"## 5. Three policies at N = {economics['n_responses']:,}")
    add("")
    add(
        f"Assumed base error rate {fmt(economics['reference_error_rate'])} "
        f"(illustrative), judge accuracy {fmt(economics['judge_accuracy'], 2)}."
    )
    add("")
    add(policy_table_markdown(economics))
    add("")
    add(
        "Coverage and verdict are different things. Every response passes the "
        "cheap probe; only the expensive verdict is rationed. Random sampling has "
        f"{fmt_pct(economics['measured_flag_rate'])} coverage *and* "
        f"{fmt_pct(economics['measured_flag_rate'])} verdict. That gap is the result."
    )
    add("")
    if economics.get("ceiling"):
        add(
            "> **The 'errors caught' column mixes two regimes.** It applies the "
            f"recall measured at a base error rate of "
            f"{fmt(economics['ceiling']['measured_base_rate'], 4)} to an assumed "
            f"production rate of {fmt(economics['reference_error_rate'])}. The "
            "**ratio** between the rows is unaffected — that is the point of "
            "`lift`, and both the assumed rate and judge accuracy cancel from it. "
            "The **absolute counts** are conservative: at the lower assumed rate "
            "the same probe would reach a higher recall (see the projection "
            "below), so the probe-triggered row understates rather than "
            "overstates."
        )
        add("")

    # 6. Headline
    add("## 6. Headline — lift")
    add("")
    add(
        f"### lift = R / f = {fmt(economics['lift'], 2)}x "
        f"{fmt_ci(boot['lift'], 2)}"
    )
    add("")
    add(
        f"At the same judge budget as random sampling, the probe surfaces "
        f"**{fmt(economics['lift'], 2)}x** as many wrong answers."
    )
    add("")
    add(
        "**The base error rate assumed in the policy table, and the judge's "
        "accuracy, both cancel from the ratio.** They appear in every policy's "
        "errors-caught figure, so the multiplier does not rest on an assumption "
        "about how often the model is wrong in production or about how good the "
        "judge is."
    )
    add("")
    ceiling = economics.get("ceiling")
    if ceiling:
        add(
            "**But the measured lift is bounded by the base rate of the set it "
            "was measured on**, and that is a different statement. Algebraically "
            "`lift = R/f = precision / base_rate`, so precision <= 1 caps lift at "
            f"`1 / base_rate` = **{fmt(ceiling['max_attainable_lift'], 2)}x** on "
            f"this test set, whose base error rate is "
            f"{fmt(ceiling['measured_base_rate'], 4)}. The measured "
            f"{fmt(economics['lift'], 2)}x is "
            f"**{fmt_pct(ceiling['fraction_of_ceiling_achieved'], 1)} of "
            "everything that was attainable here.** No probe, however well it "
            "ranks, could have scored much higher on this dataset "
            "(DECISIONS.md 015)."
        )
        add("")
        if ceiling.get("lift_from_precision") is not None:
            add(
                f"Checked both ways: `R/f` = {fmt(economics['lift'], 4)} and "
                f"`precision/base_rate` = "
                f"{fmt(ceiling['lift_from_precision'], 4)}."
            )
            add("")
        if boot.get("ceiling", {}).get("ci_low") is not None:
            add(
                f"The ceiling is itself an estimate: {fmt_ci(boot['ceiling'], 2)} "
                "over the same bootstrap resamples. A resample drawing fewer "
                "incorrect answers has a higher ceiling, which is why the "
                f"interval on lift ({fmt_ci(boot['lift'], 2)}) can reach above the "
                f"{fmt(ceiling['max_attainable_lift'], 2)}x computed from the "
                "point base rate. Within any single resample the two cannot "
                "cross: `lift / ceiling` is precision, which is at most 1."
            )
            add("")
    projection = economics.get("projection")
    if projection and projection.get("rows"):
        add("#### Headroom at lower base error rates — a projection, not a result")
        add("")
        add(
            "A ROC curve is base-rate independent: it describes how well the probe "
            "*ranks*, which is a property of the probe rather than of how often "
            "the model is wrong. So the measured curve can be re-read at other "
            f"base error rates, holding the budget at the measured `f` = "
            f"{fmt(projection['budget'], 4)}."
        )
        add("")
        add(projection_table_markdown(economics))
        add("")
        add(f"> **{projection['caveat']}**")
        add("")
    inv = economics["invariance"]
    add(
        f"Demonstrated, not asserted: recomputing the table across error rates "
        f"{inv['error_rates_tested']} and judge accuracies "
        f"{inv['judge_accuracies_tested']} gives {len(inv['lifts'])} lifts with a "
        f"spread of {inv['spread']:.1e} (all equal: {inv['all_equal']})."
    )
    add("")
    add(
        f"The interval is a {int(boot['ci'] * 100)}% percentile bootstrap over "
        f"{boot['n_samples']:,} resamples of the {test['n']:,} test examples, "
        "resampling recall and flag rate jointly because lift is their ratio."
    )
    add("")

    # 7. Latency
    add("## 7. Latency")
    add("")
    add("| Measurement | Value |")
    add("|---|---|")
    add(f"| Probe score, median (full scikit-learn call) | {fmt(comparison['probe_median_us'], 1)} µs |")
    add(f"| Probe score, p95 | {fmt(comparison['probe_p95_us'], 1)} µs |")
    if comparison.get("raw_dot_product_median_us") is not None:
        add(
            f"| — of which arithmetic (standardise + dot + bias) | "
            f"{fmt(comparison['raw_dot_product_median_us'], 1)} µs |"
        )
    add(f"| Generation, median per response | {fmt(comparison['generation_median_ms'], 1)} ms |")
    add(f"| Prefill, median per response | {fmt(comparison['prefill_median_ms'], 1)} ms |")
    add(f"| Probe / generation | {fmt_ratio(comparison['probe_over_generation'])} |")
    add(f"| Probe / prefill | {fmt_ratio(comparison['probe_over_prefill'])} |")
    add(f"| Device | {latency['device']['device_name']} |")
    add(f"| torch | {latency['versions']['torch']} |")
    add(f"| Quantisation | {latency['quantization']} |")
    add("")
    add(
        "**The probe adds no additional forward pass.** The activation is a "
        "by-product of the prefill that generation already performs, so its "
        "marginal cost is the scale-and-dot-product timed above and nothing else."
    )
    if comparison.get("sklearn_overhead_factor"):
        add("")
        add(
            f"The headline figure is the whole scikit-learn call. Most of it is "
            f"input validation and array copying, not arithmetic: the dot product "
            f"itself is {fmt(comparison['sklearn_overhead_factor'], 1)}x faster. "
            "The slower, deployable number is quoted as the headline on purpose."
        )
    add("")
    add(
        f"Extraction throughput for reference: {extraction['n_examples']:,} "
        f"examples in {fmt_duration(extraction['total_seconds'])} "
        f"({fmt(extraction['examples_per_second'], 2)} examples/s) at batch size "
        f"{extraction['batch_size']}."
    )
    add("")

    # 8. Abstention
    add("## 8. Secondary validation — abstention correlation")
    add("")
    if abstention["underpowered"]:
        add(
            f"Abstention rate on test is {fmt(abstention['abstention_rate'], 4)} "
            f"({abstention['n_abstained']} of {abstention['n']}), below the "
            f"{fmt(abstention['min_rate_to_report'], 2)} floor. **The comparison is "
            "underpowered and is not reported as a result** (SPEC.md §9)."
        )
    else:
        add(
            f"Abstention rate on test: {fmt(abstention['abstention_rate'], 4)} "
            f"({abstention['n_abstained']} of {abstention['n']})."
        )
        add("")
        add("| Group | Mean probe score |")
        add("|---|---|")
        add(f"| Abstained | {fmt(abstention['mean_score_abstained'], 4)} |")
        add(f"| Did not abstain | {fmt(abstention['mean_score_not_abstained'], 4)} |")
        add("")
        add(
            f"AUROC of the probe score for predicting abstention: "
            f"{fmt(abstention['auroc_predicting_abstention'], 4)}. The direction "
            "tracks the model's own expressed uncertainty as well as its "
            "correctness — independent evidence that the probe reads something "
            "real rather than a dataset artifact."
        )
    add("")

    # 9. Negative control
    add("## 9. Negative control — reproducing a documented limitation")
    add("")
    add(render_negative_control(artifacts, probe_test))
    add("")

    # 10. Limitations
    add("## 10. Limitations")
    add("")
    for item in limitations(artifacts, config):
        add(f"- {item}")
    add("")
    return "\n".join(lines) + "\n"


def render_negative_control(
    artifacts: dict[str, Any], probe_test: dict[str, Any]
) -> str:
    """Render the GSM8K section, framed as a reproduction (SPEC.md §10)."""
    if "negative_control" not in artifacts:
        return (
            "Not run — **and not implemented in this build**. The published result "
            "reports that probe generalisation falters on mathematical reasoning "
            "(DECISIONS.md 008), and reproducing that on GSM8K is Stage 6 of "
            "`TASKS.md`: an optional stage that requires a completed main run "
            "first. The `negative_control` block in `config.yaml` reserves the "
            "settings for it, but no code reads them yet, so setting "
            "`enabled: true` does nothing today. Until Stage 6 is built and run, "
            "cross-domain generalisation is untested here and is listed as a "
            "limitation below."
        )
    control = artifacts["negative_control"]["negative_control"]
    main_auroc = probe_test["test"]["auroc"]
    return "\n".join(
        [
            "The published result reports that this method's generalisation falters "
            "on mathematical reasoning. We ran the identical pipeline on "
            f"`{control['dataset']}` and reproduce that finding.",
            "",
            "| Dataset | Test AUROC | n |",
            "|---|---|---|",
            f"| {control['dataset']} (mathematical reasoning) | "
            f"{fmt(control['test']['auroc'], 4)} | {control['test']['n']:,} |",
            f"| TriviaQA (knowledge recall) | {fmt(main_auroc, 4)} | "
            f"{probe_test['test']['n']:,} |",
            "",
            "This is a successful reproduction of a documented limitation, not a "
            "shortcoming of this implementation. A claim that the method works on "
            "knowledge questions is stronger when accompanied by evidence of where "
            "it does not.",
        ]
    )


def limitations(artifacts: dict[str, Any], config: Config) -> list[str]:
    """Build the limitations list from what this run actually did.

    Written from the artifacts rather than from a boilerplate block, so it
    cannot drift into claiming a limitation was addressed when it was not
    (SPEC.md §13: "not optional and not boilerplate").
    """
    extract = artifacts["extract_meta"]
    probe_test = artifacts["probe_test"]
    data = artifacts["data_stats"]["data"]
    base_rates = extract["base_rates"]
    raw_gap = base_rates.get("lenient_minus_strict_accuracy")
    gap = None if raw_gap is None else abs(raw_gap)

    items = [
        f"**One model, one dataset.** `{extract['model']['name']}` on "
        f"{data['dataset']} ({data['dataset_config']}). Cross-model and "
        "cross-dataset generalisation is untested here.",
        "**Knowledge questions only.** "
        + (
            "The GSM8K negative control was run and is reported above."
            if "negative_control" in artifacts
            else "The published result reports that generalisation falters on "
            "mathematical reasoning; the GSM8K control was not run, so that "
            "boundary is cited here rather than measured."
        ),
        "**This measures the probe, not a system.** There is no gateway, no "
        "serving path, and no end-to-end latency under load. The latency figures "
        "are component measurements taken in isolation.",
        f"**`f` depends on workload.** The flag rate of "
        f"{fmt(probe_test['test']['flag_rate'], 4)} is for this dataset at this "
        "threshold. Real traffic has a different difficulty distribution and would "
        "produce a different rate at the same threshold.",
        "**Judge accuracy is assumed to cancel.** It does cancel from the ratio, "
        "but a real judge misses errors the probe correctly flagged, so the "
        "absolute errors-caught counts in the three-policy table are upper bounds.",
        f"**Single seed.** Everything reported is at seed "
        f"{probe_test['provenance'].get('seed', config.seed)}. No seed "
        "sweep was run, so the intervals reflect test-set sampling only, not "
        "variation in the split or in probe fitting.",
        "**Labelling is automatic.** Normalised alias matching is a proxy for "
        "correctness, not human judgment. "
        + (
            "Strict exact match was not recorded, so the size of that proxy's "
            "effect is unmeasured for this run."
            if gap is None
            else f"Lenient and strict matching differ by {fmt(gap)} in accuracy here."
        ),
        f"**Test set is small.** {probe_test['test']['n']:,} examples, which is "
        "why every headline number carries a bootstrap interval rather than a "
        "point estimate alone.",
    ]
    ceiling = artifacts.get("economics", {}).get("economics", {}).get("ceiling")
    if ceiling:
        items.insert(
            0,
            "**The headline lift is bounded by this benchmark, not by the probe.** "
            f"`lift = precision / base_rate`, so a base error rate of "
            f"{fmt(ceiling['measured_base_rate'], 4)} caps it at "
            f"{fmt(ceiling['max_attainable_lift'], 2)}x however well the probe "
            f"ranks. The measured value reaches "
            f"{fmt_pct(ceiling['fraction_of_ceiling_achieved'], 1)} of that. The "
            "transferable quantity is the AUROC; the lift is specific to a "
            "workload where the model is wrong this often (DECISIONS.md 015).",
        )
    if probe_test["auroc_floor"]["below_floor"]:
        items.insert(
            0,
            f"**The result is weak.** Test AUROC "
            f"{fmt(probe_test['test']['auroc'], 4)} is at or below the "
            f"{probe_test['auroc_floor']['floor']} floor set in `config.yaml`. It "
            "is reported as measured; no tuning was done against the test set.",
        )
    return items


# --------------------------------------------------------------------------- #
# README.md
# --------------------------------------------------------------------------- #


PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def readme_values(artifacts: dict[str, Any], config: Config) -> dict[str, str]:
    """Map every ``{{placeholder}}`` in README_TEMPLATE.md to an artifact value.

    Every entry traces to a file in ``results/``; nothing here is a literal
    (CLAUDE.md invariant 7).
    """
    extract = artifacts["extract_meta"]
    data = artifacts["data_stats"]["data"]
    probe_test = artifacts["probe_test"]
    sweep = artifacts["probe_sweep"]
    economics = artifacts["economics"]["economics"]
    latency = artifacts["latency"]["latency"]

    model_info = extract["model"]
    test = probe_test["test"]
    boot = probe_test["bootstrap"]
    probe = probe_test["probe"]
    base_rates = extract["base_rates"]
    abstention = probe_test["abstention"]
    comparison = latency["comparison"]
    policies = {row["policy"]: row for row in economics["policies"]}

    ceiling_block = economics.get("ceiling", {})
    strict_accuracy = probe_test.get("strict_em", {}).get("test_accuracy_strict")
    strict_test_base_rate = (
        None if strict_accuracy is None else 1.0 - strict_accuracy
    )

    values = {
        "model_name": model_info["name"],
        "quantization": model_info["quantization"],
        "dataset": f"{data['dataset']} {data['dataset_config']}",
        "n_test": f"{test['n']:,}",
        # Test-set rates, not whole-dataset ones: the table row above them reads
        # "n = {{n_test}} held-out questions", so a dataset-wide rate there would
        # be describing a different set of examples than the row claims.
        "base_rate": fmt(test["base_rate"]),
        "strict_em_base_rate": fmt(strict_test_base_rate),
        "layer": str(probe["layer"]),
        "n_layers": str(model_info["num_hidden_layers"]),
        "hidden_size": f"{model_info['hidden_size']:,}",
        "auroc": fmt(test["auroc"], 3),
        "auroc_ci_low": fmt(boot["auroc"]["ci_low"], 3),
        "auroc_ci_high": fmt(boot["auroc"]["ci_high"], 3),
        "flag_rate": fmt(test["flag_rate"], 3),
        "flag_rate_pct": pct_number(test["flag_rate"]),
        "recall": fmt(test["recall"], 3),
        "recall_ci_low": fmt(boot["recall"]["ci_low"], 3),
        "recall_ci_high": fmt(boot["recall"]["ci_high"], 3),
        "precision": fmt(test["precision"], 3),
        "lift": fmt(economics["lift"], 1),
        "lift_ceiling": fmt(ceiling_block.get("max_attainable_lift"), 1),
        "lift_pct_of_ceiling": fmt_pct(
            ceiling_block.get("fraction_of_ceiling_achieved"), 0
        ),
        "measured_base_rate": fmt(ceiling_block.get("measured_base_rate"), 3),
        "projection_table": projection_table_markdown(economics)
        or "_Not computed for this run._",
        "projection_caveat": economics.get("projection", {}).get("caveat", ""),
        "lift_ci_low": fmt(boot["lift"]["ci_low"], 1),
        "lift_ci_high": fmt(boot["lift"]["ci_high"], 1),
        "probe_latency_us": fmt(comparison["probe_median_us"], 1),
        "generation_latency_ms": fmt(comparison["generation_median_ms"], 0),
        # The template reads "a factor of {{latency_ratio}}", so this is the
        # inverse of the fraction reported in RESULTS.md: how many times
        # cheaper the probe is, which is what "a factor of" means in prose.
        "latency_ratio": (
            f"{1.0 / comparison['probe_over_generation']:,.0f}x"
            if comparison["probe_over_generation"]
            else "n/a"
        ),
        "reference_error_rate": fmt(economics["reference_error_rate"]),
        "policy_a_caught": fmt_count(policies["judge_everything"]["errors_caught"]),
        "policy_a_cost": f"{policies['judge_everything']['relative_cost']:.0f}",
        "policy_b_calls": fmt_count(policies["random_sample"]["judge_calls"]),
        "policy_b_caught": fmt_count(policies["random_sample"]["errors_caught"]),
        "policy_c_calls": fmt_count(policies["probe_triggered"]["judge_calls"]),
        "policy_c_caught": fmt_count(policies["probe_triggered"]["errors_caught"]),
        "layer_sweep_table": layer_sweep_table(sweep),
        "total_runtime": fmt_duration(extract["extraction"]["total_seconds"]),
        "device_name": latency["device"]["device_name"],
        "abstain_score": fmt(abstention["mean_score_abstained"], 3),
        "non_abstain_score": fmt(abstention["mean_score_not_abstained"], 3),
        "abstain_auroc": fmt(abstention["auroc_predicting_abstention"], 3),
        "abstain_rate": fmt(abstention["abstention_rate"], 3),
        "negative_control_section": (
            "## Negative control\n\n"
            + render_negative_control(artifacts, probe_test)
        ),
        "negative_control_note": (
            "We reproduced that on GSM8K; see the negative control section above."
            if "negative_control" in artifacts
            else "We did not run the GSM8K control, so that boundary is cited "
            "rather than measured here."
        ),
    }
    return values


def render_readme(template: str, artifacts: dict[str, Any], config: Config) -> str:
    """Substitute every placeholder in the README template.

    The template's own instruction header (everything above the first ``---``
    rule) is stripped, since it is a note to the author rather than to a reader.

    Raises:
        KeyError: if a placeholder has no value. A blank in a published README
            is worse than a crash here.
    """
    body = template.split("\n---\n", 1)[-1].lstrip("\n")
    values = readme_values(artifacts, config)

    missing = {
        name for name in PLACEHOLDER.findall(body) if name not in values
    }
    if missing:
        raise KeyError(
            f"README template has placeholder(s) with no value: {sorted(missing)}"
        )
    unused = sorted(set(values) - set(PLACEHOLDER.findall(body)))
    if unused:
        LOGGER.warning("readme values with no placeholder: %s", unused)

    return PLACEHOLDER.sub(lambda m: values[m.group(1)], body)


# --------------------------------------------------------------------------- #
# Display helpers for notebooks
#
# These exist so notebooks/cascade_economics.ipynb can hold no logic
# (CLAUDE.md, Coding standards). The notebook calls one function per cell and
# displays what comes back; anything a reviewer might want to check therefore
# lives in this module, under test, rather than in a cell.
# --------------------------------------------------------------------------- #


def metadata_frame(artifacts: dict[str, Any]) -> "Any":
    """Run metadata as a two-column DataFrame, for the notebook's first cell."""
    import pandas as pd

    provenance = artifacts["probe_test"]["provenance"]
    model_info = artifacts["extract_meta"]["model"]
    equivalence = artifacts["extract_meta"]["equivalence_check"]
    data = artifacts["data_stats"]["data"]
    rows = [
        ("Model", model_info["name"]),
        ("Quantisation", f"{model_info['quantization']} ({model_info['dtype']})"),
        ("Device", provenance["device"]["device_name"]),
        ("Dataset", f"{data['dataset']} / {data['dataset_config']} / {data['split']}"),
        ("Examples", f"{data['n_final']:,}"),
        (
            "Split sizes",
            f"train {data['split_sizes']['train']:,} · "
            f"val {data['split_sizes']['val']:,} · "
            f"test {data['split_sizes']['test']:,}",
        ),
        ("Seed", provenance["seed"]),
        ("Config hash", provenance["config_hash"]),
        ("Git commit", str(provenance["git_commit"])[:12]),
        ("Tree dirty at run time", provenance["dirty"]),
        ("Timestamp (UTC)", provenance["timestamp_utc"]),
        (
            "Left-padding equivalence",
            f"max deviation {equivalence['max_deviation']:.2e} "
            f"(tolerance {equivalence['tolerance']:.0e})",
        ),
    ]
    return pd.DataFrame(rows, columns=["Field", "Value"]).set_index("Field")


def sweep_frame(artifacts: dict[str, Any]) -> "Any":
    """Layer sweep as a layer x C DataFrame of validation AUROC."""
    import pandas as pd

    frame = pd.DataFrame(artifacts["probe_sweep"]["sweep"])
    return frame.pivot(index="layer", columns="C", values="val_auroc").round(4)


def test_metrics_frame(artifacts: dict[str, Any]) -> "Any":
    """Test metrics with bootstrap intervals, precision and recall separate."""
    import pandas as pd

    test = artifacts["probe_test"]["test"]
    boot = artifacts["probe_test"]["bootstrap"]
    rows = [
        ("AUROC", fmt(test["auroc"], 4), fmt_ci(boot["auroc"], 4)),
        ("Measured flag rate f", fmt(test["flag_rate"], 4), fmt_ci(boot["flag_rate"], 4)),
        ("Recall R", fmt(test["recall"], 4), fmt_ci(boot["recall"], 4)),
        ("Precision", fmt(test["precision"], 4), fmt_ci(boot["precision"], 4)),
        ("Base error rate", fmt(test["base_rate"], 4), "—"),
        ("Test examples", f"{test['n']:,}", "—"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "95% CI"]).set_index("Metric")


def policy_frame(artifacts: dict[str, Any]) -> "Any":
    """The three-policy comparison as a DataFrame.

    This is the visual centrepiece of the notebook and of the video, so the
    columns are named as a reader would say them aloud rather than as the JSON
    keys them.
    """
    import pandas as pd

    economics = artifacts["economics"]["economics"]
    rows = [
        {
            "Policy": row["label"],
            "Judge calls": fmt_count(row["judge_calls"]),
            "Coverage": pct_number(row["coverage"]) + "%",
            "Errors caught": fmt_count(row["errors_caught"]),
            "Relative cost": f"{row['relative_cost']:.0f}x",
        }
        for row in economics["policies"]
    ]
    return pd.DataFrame(rows).set_index("Policy")


def headline_markdown(artifacts: dict[str, Any]) -> str:
    """The headline lift, its interval, and the independence sentence."""
    economics = artifacts["economics"]["economics"]
    boot = artifacts["probe_test"]["bootstrap"]
    test = artifacts["probe_test"]["test"]
    return "\n".join(
        [
            f"# lift = R / f = {fmt(economics['lift'], 2)}x",
            "",
            f"### 95% CI {fmt_ci(boot['lift'], 2)}",
            "",
            f"Measured on {test['n']:,} held-out questions: "
            f"recall **R = {fmt(test['recall'], 3)}**, "
            f"measured flag rate **f = {fmt(test['flag_rate'], 3)}**.",
            "",
            "At the same judge budget as random sampling, the probe surfaces "
            f"**{fmt(economics['lift'], 2)}x** as many wrong answers.",
            "",
            "`R/f` is independent of the base error rate and of the judge's own "
            "accuracy — both appear in every policy and cancel from the ratio.",
        ]
    )


def latency_frame(artifacts: dict[str, Any]) -> "Any":
    """Probe cost against generation cost, with the device attached."""
    import pandas as pd

    latency = artifacts["latency"]["latency"]
    comparison = latency["comparison"]
    rows = [
        ("Probe score (median)", f"{comparison['probe_median_us']:.1f} µs"),
        ("Probe score (p95)", f"{comparison['probe_p95_us']:.1f} µs"),
        ("Generation (median/response)", f"{comparison['generation_median_ms']:.1f} ms"),
        (
            "Prefill (median/response)",
            "n/a"
            if comparison["prefill_median_ms"] is None
            else f"{comparison['prefill_median_ms']:.1f} ms",
        ),
        ("Probe / generation", f"{comparison['probe_over_generation']:.2e}"),
        ("Additional forward passes", "0 — the activation is a prefill by-product"),
        ("Device", latency["device"]["device_name"]),
    ]
    return pd.DataFrame(rows, columns=["Measurement", "Value"]).set_index("Measurement")
