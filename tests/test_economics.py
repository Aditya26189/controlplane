"""Cascade economics: lift is R/f, and it depends on nothing else.

The invariance tests are the point. "But you assumed a 3% error rate" is the
most natural attack on this analysis, and the answer is that the assumption
cancels out of the ratio. These make that demonstrable rather than asserted.
"""

import pytest

from src.economics import compare_policies, invariance_check, lift, policy_table
from src.evaluate import evaluate_at_threshold


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
