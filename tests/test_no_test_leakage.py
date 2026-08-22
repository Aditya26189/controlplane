"""The test set is never used for selection (CLAUDE.md invariant 2).

Selecting the layer, C, or the threshold on test is the single most common way
to produce an inflated headline number, and it is the first thing a reviewer
checks. These tests make the claim checkable rather than asserted:

* the scaler's fitted statistics equal the training rows' statistics exactly,
  so it cannot have seen val or test;
* corrupting every test row leaves the sweep and the chosen threshold
  bit-identical, so neither can depend on test.

The count of scorings is disclosed rather than capped; what must never happen is
a *selection* that consulted test (DECISIONS.md 016, 017).
"""

import numpy as np
import pytest

from src.probe import (
    fit_probe,
    fit_selected_probe,
    probe_scores,
    run_sweep,
    select_threshold,
)


@pytest.fixture
def split_data():
    """Activations where each split has a deliberately different distribution.

    Val and test are shifted far from train. If the scaler ever saw them, its
    mean would move measurably, so the equality check below has real power.
    """
    rng = np.random.RandomState(7)
    n_train, n_val, n_test = 120, 40, 40
    n = n_train + n_val + n_test
    split = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    labels = np.tile([0, 1], n // 2)

    activations = {}
    for layer, strength in [(1, 1.2), (2, 0.6), (3, 0.2)]:
        x = rng.randn(n, 10)
        x[labels == 1, :2] += strength
        x[split == "val"] += 5.0
        x[split == "test"] += 50.0
        activations[layer] = x
    return activations, labels, split


def test_scaler_is_fit_on_train_rows_only(config, split_data):
    """The scaler's mean must equal the training mean, to floating point."""
    activations, labels, split = split_data
    x = activations[1]
    train_mask = split == "train"

    scaler, _ = fit_probe(x[train_mask], labels[train_mask], 1.0, config)

    np.testing.assert_allclose(scaler.mean_, x[train_mask].mean(axis=0), rtol=1e-12)
    # And must NOT equal the statistics of any set that includes val or test.
    assert not np.allclose(scaler.mean_, x.mean(axis=0), rtol=1e-6)
    assert not np.allclose(
        scaler.mean_, x[split != "test"].mean(axis=0), rtol=1e-6
    )


def test_sweep_is_unchanged_when_test_rows_are_corrupted(config, split_data):
    """Destroy every test row; the layer/C selection must not move at all."""
    activations, labels, split = split_data

    clean = run_sweep(activations, labels, split, config)

    corrupted = {
        layer: np.where((split == "test")[:, None], 1e6, x.copy())
        for layer, x in activations.items()
    }
    corrupted_labels = labels.copy()
    corrupted_labels[split == "test"] = 1 - corrupted_labels[split == "test"]

    after = run_sweep(corrupted, corrupted_labels, split, config)

    assert clean["best"] == after["best"]
    assert clean["sweep"] == after["sweep"]


def test_threshold_is_unchanged_when_test_rows_are_corrupted(config, split_data):
    """The threshold is frozen on validation and must not see test either."""
    activations, labels, split = split_data
    sweep = run_sweep(activations, labels, split, config)

    clean = fit_selected_probe(activations, labels, split, sweep["best"], config)

    corrupted = {
        layer: np.where((split == "test")[:, None], -1e6, x.copy())
        for layer, x in activations.items()
    }
    after = fit_selected_probe(corrupted, labels, split, sweep["best"], config)

    assert clean.threshold == after.threshold
    assert clean.val_auroc == after.val_auroc
    np.testing.assert_array_equal(
        clean.classifier.coef_, after.classifier.coef_
    )


def test_threshold_hits_the_target_flag_rate_on_validation(config):
    """Threshold selection is a validation-set decision with a stated target."""
    rng = np.random.RandomState(3)
    scores = rng.randn(600)
    threshold = select_threshold(scores, 0.05)
    achieved = float(np.mean(scores >= threshold))
    assert achieved == pytest.approx(0.05, abs=0.005)


def test_measured_flag_rate_may_differ_from_the_target(config, split_data):
    """Invariant 6: downstream maths uses the measured rate, not the target.

    The threshold is chosen to hit the target on validation; on a different
    split the realised rate differs, and this test documents that it is
    expected to.
    """
    activations, labels, split = split_data
    sweep = run_sweep(activations, labels, split, config)
    probe = fit_selected_probe(activations, labels, split, sweep["best"], config)

    test_scores = probe.score(activations[probe.layer][split == "test"])
    measured = float(np.mean(test_scores >= probe.threshold))

    assert probe.val_flag_rate == pytest.approx(
        config.economics.target_flag_rate, abs=0.05
    )
    assert isinstance(measured, float)  # it is measured, whatever it turns out to be


def test_sweep_requires_both_train_and_val(config, split_data):
    activations, labels, split = split_data
    only_train = np.array(["train"] * len(split))
    with pytest.raises(ValueError, match="non-empty train and validation"):
        run_sweep(activations, labels, only_train, config)
