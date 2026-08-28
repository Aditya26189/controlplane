"""Cascade economics: lift is R/f, its ceiling, and what cancels from it.

The invariance tests are the point. "But you assumed a 3% error rate" is the
most natural attack on this analysis, and the answer is that the assumption
cancels out of the ratio. These make that demonstrable rather than asserted.
"""

from pathlib import Path

import pytest

from src.economics import (
    compare_policies,
    invariance_check,
    lift,
    lift_ceiling,
    lift_from_precision,
    policy_table,
    project_lift_at_base_rate,
)
from src.evaluate import evaluate_at_threshold

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_lift_is_recall_over_flag_rate():
    """The definition, pinned exactly (SPEC.md §7)."""
    assert lift(0.61, 0.052) == pytest.approx(0.61 / 0.052)
    assert lift(0.5, 0.05) == pytest.approx(10.0)


def test_lift_of_one_means_no_better_than_random():
    assert lift(0.05, 0.05) == pytest.approx(1.0)


def test_lift_undefined_at_zero_flag_rate():
    """No flagged responses means no budget to compare against."""
    with pytest.raises(ValueError, match="undefined"):
        lift(0.5, 0.0)


def test_random_sampling_policy_has_lift_exactly_one():
    """A probe whose recall equals its flag rate is exactly random sampling."""
    f = 0.05
    table = {
        row["policy"]: row
        for row in policy_table(1_000_000, error_rate=0.03, flag_rate=f, recall=f)
    }
    assert table["probe_triggered"]["errors_caught"] == pytest.approx(
        table["random_sample"]["errors_caught"]
    )
    assert lift(f, f) == pytest.approx(1.0)


@pytest.mark.parametrize("error_rate", [0.001, 0.03, 0.25, 0.9])
def test_lift_is_invariant_to_the_base_error_rate(error_rate):
    """e scales every policy's errors-caught identically, so it cancels."""
    result = compare_policies(1_000_000, error_rate, flag_rate=0.05, recall=0.60)
    assert result["lift"] == pytest.approx(12.0)
    assert result["lift_from_policy_table"] == pytest.approx(12.0)


@pytest.mark.parametrize("judge_accuracy", [0.4, 0.75, 1.0])
def test_lift_is_invariant_to_judge_accuracy(judge_accuracy):
    """a multiplies every policy's errors-caught identically, so it cancels."""
    result = compare_policies(
        1_000_000, 0.03, flag_rate=0.05, recall=0.60, judge_accuracy=judge_accuracy
    )
    assert result["lift"] == pytest.approx(12.0)
    assert result["lift_from_policy_table"] == pytest.approx(12.0)


def test_invariance_block_reports_all_equal():
    """The artifact carries the demonstration, not just the claim."""
    block = invariance_check(flag_rate=0.05, recall=0.60)
    assert block["all_equal"] is True
    assert block["spread"] == pytest.approx(0.0, abs=1e-9)
    assert len(block["lifts"]) == len(block["error_rates_tested"]) * len(
        block["judge_accuracies_tested"]
    )
    assert all(row["lift"] == pytest.approx(12.0) for row in block["lifts"])


def test_two_ways_of_computing_lift_must_agree():
    """R/f and the table's errors-caught ratio are the same claim."""
    result = compare_policies(1_000_000, 0.03, flag_rate=0.052, recall=0.61)
    assert result["lift"] == pytest.approx(result["lift_from_policy_table"])


def test_probe_and_random_spend_the_same_budget():
    """Equal judge calls is what makes the comparison fair at all."""
    table = {
        row["policy"]: row
        for row in policy_table(1_000_000, 0.03, flag_rate=0.05, recall=0.60)
    }
    assert table["probe_triggered"]["judge_calls"] == table["random_sample"]["judge_calls"]
    assert table["probe_triggered"]["relative_cost"] == pytest.approx(1.0)
    assert table["random_sample"]["relative_cost"] == pytest.approx(1.0)


def test_coverage_differs_even_though_cost_does_not():
    """Coverage and verdict are different things; that gap is the whole result."""
    table = {
        row["policy"]: row
        for row in policy_table(1_000_000, 0.03, flag_rate=0.05, recall=0.60)
    }
    assert table["random_sample"]["coverage"] == pytest.approx(0.05)
    assert table["probe_triggered"]["coverage"] == pytest.approx(1.0)


def test_judge_everything_costs_one_over_f():
    table = {
        row["policy"]: row
        for row in policy_table(1_000_000, 0.03, flag_rate=0.05, recall=0.60)
    }
    assert table["judge_everything"]["relative_cost"] == pytest.approx(20.0)
    assert table["judge_everything"]["judge_calls"] == pytest.approx(1_000_000)


def test_flag_rate_out_of_range_raises():
    with pytest.raises(ValueError, match="flag_rate"):
        policy_table(1000, 0.03, flag_rate=0.0, recall=0.5)
    with pytest.raises(ValueError, match="flag_rate"):
        policy_table(1000, 0.03, flag_rate=1.5, recall=0.5)


def test_lift_matches_the_metrics_module_end_to_end():
    """evaluate_at_threshold and economics must not drift apart."""
    import numpy as np

    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    scores = np.array([9.0, 8.0, 0.1, 0.0, 7.0, 0.5, 0.4, 0.3, 0.2, 0.1])
    result = evaluate_at_threshold(labels, scores, threshold=7.0)

    assert result["recall"] == pytest.approx(2 / 4)
    assert result["flag_rate"] == pytest.approx(3 / 10)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["lift"] == pytest.approx(lift(result["recall"], result["flag_rate"]))


# --- the ceiling: lift = precision/base_rate (DECISIONS.md 015) ------------ #


def test_lift_equals_precision_over_base_rate():
    """The identity that makes the ceiling real, checked against real numbers.

    From the Kaggle run: TP 33, FP 4, FN 200, TN 363 on 600 examples.
    """
    tp, fp, fn, n = 33, 4, 200, 600
    positives, flagged = tp + fn, tp + fp
    base_rate, flag_rate = positives / n, flagged / n
    recall, precision = tp / positives, tp / flagged

    assert lift(recall, flag_rate) == pytest.approx(
        lift_from_precision(precision, base_rate)
    )
    assert lift(recall, flag_rate) == pytest.approx(2.2967, abs=1e-4)


def test_ceiling_is_one_over_base_rate():
    assert lift_ceiling(0.3883) == pytest.approx(2.5753, abs=1e-3)
    assert lift_ceiling(0.03) == pytest.approx(33.333, abs=1e-2)
    assert lift_ceiling(1.0) == pytest.approx(1.0)


def test_lift_can_never_exceed_the_ceiling():
    """Precision <= 1 is the whole argument; assert it holds across the range."""
    for base_rate in [0.05, 0.2, 0.3883, 0.7]:
        for precision in [0.1, 0.5, 0.9, 1.0]:
            assert lift_from_precision(precision, base_rate) <= lift_ceiling(
                base_rate
            ) + 1e-12


def test_measured_run_achieved_89_percent_of_its_ceiling():
    """The headline is ceiling-bound, not probe-bound. Pinned so it stays stated."""
    result = compare_policies(
        1_000_000,
        0.03,
        flag_rate=37 / 600,
        recall=33 / 233,
        measured_base_rate=233 / 600,
        precision=33 / 37,
    )
    ceiling = result["ceiling"]
    assert ceiling["max_attainable_lift"] == pytest.approx(2.5751, abs=1e-3)
    assert ceiling["fraction_of_ceiling_achieved"] == pytest.approx(0.892, abs=1e-3)
    assert ceiling["lift_from_precision"] == pytest.approx(result["lift"])


def test_ceiling_absent_when_base_rate_not_supplied():
    """Backwards compatible: the block only appears when it can be computed."""
    assert "ceiling" not in compare_policies(1000, 0.03, 0.05, 0.6)


# --- projecting the measured ROC onto other base rates -------------------- #


def perfect_roc():
    return [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]


def chance_roc(points: int = 101):
    """A diagonal ROC, finely sampled so small budgets have operating points."""
    grid = [i / (points - 1) for i in range(points)]
    return grid, list(grid)


def test_projection_of_a_perfect_ranker_hits_the_ceiling():
    fpr, tpr = perfect_roc()
    row = project_lift_at_base_rate(fpr, tpr, base_rate=0.03, budget=0.05)
    assert row["lift"] == pytest.approx(row["ceiling"], rel=1e-9)


def test_projection_of_a_chance_ranker_gives_lift_one():
    """A probe no better than random must project to exactly 1.0."""
    fpr, tpr = chance_roc()
    row = project_lift_at_base_rate(fpr, tpr, base_rate=0.03, budget=0.5)
    assert row["lift"] == pytest.approx(1.0, abs=1e-9)


def test_projection_respects_the_budget():
    fpr, tpr = chance_roc()
    row = project_lift_at_base_rate(fpr, tpr, base_rate=0.03, budget=0.05)
    assert row["flag_rate"] <= 0.05


def test_projection_returns_none_when_no_point_fits_the_budget():
    """A coarse curve may have no operating point inside a small budget.

    None is the honest answer there; inventing an interpolated point would be
    reporting an operating point the probe cannot actually be set to.
    """
    fpr, tpr = [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]
    assert project_lift_at_base_rate(fpr, tpr, base_rate=0.03, budget=0.05) is None


def test_projection_is_labelled_as_a_projection():
    """It must never be mistaken for a measurement in an artifact."""
    fpr, tpr = perfect_roc()
    row = project_lift_at_base_rate(fpr, tpr, base_rate=0.03, budget=0.05)
    assert row["projected"] is True

    result = compare_policies(
        1000,
        0.03,
        0.05,
        0.6,
        roc={"fpr": fpr, "tpr": tpr},
        projection_base_rates=(0.03,),
    )
    assert "PROJECTION, NOT MEASUREMENT" in result["projection"]["caveat"]
    assert "NOT tested" in result["projection"]["caveat"]


def test_projection_reproduces_the_measured_row():
    """Projecting onto the test set's own base rate must recover what was measured."""
    import json

    bundle = REPO_ROOT / "results_bundle (1)" / "results" / "probe_test.json"
    if not bundle.is_file():
        pytest.skip("real-run bundle not present")
    probe_test = json.loads(bundle.read_text(encoding="utf-8"))
    test = probe_test["test"]
    row = project_lift_at_base_rate(
        probe_test["roc"]["fpr"],
        probe_test["roc"]["tpr"],
        base_rate=test["base_rate"],
        budget=test["flag_rate"],
    )
    # Same regime, same budget: the projection must land near the measurement.
    assert row["lift"] == pytest.approx(test["lift"], rel=0.10)
