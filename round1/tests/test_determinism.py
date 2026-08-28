"""Two runs at one seed produce identical numbers (CLAUDE.md definition of done).

Reproducibility here is not a nicety: the README states a config hash and a
seed next to every number, and that claim is only worth something if re-running
at that seed reproduces the number exactly.
"""

import numpy as np
import pytest

from src.config import set_seeds
from src.evaluate import bootstrap_metrics, evaluate_at_threshold
from src.probe import fit_probe, fit_selected_probe, run_sweep


@pytest.fixture
def data():
    rng = np.random.RandomState(11)
    n = 200
    split = np.array(["train"] * 120 + ["val"] * 40 + ["test"] * 40)
    labels = np.tile([0, 1], n // 2)
    activations = {}
    for layer, strength in [(1, 0.9), (2, 0.5)]:
        x = rng.randn(n, 8)
        x[labels == 1, :2] += strength
        activations[layer] = x
    return activations, labels, split


def test_probe_coefficients_are_identical_across_runs(config, data):
    """Same seed, same coefficients, bit for bit."""
    activations, labels, split = data
    x, y = activations[1][split == "train"], labels[split == "train"]

    set_seeds(config.seed)
    _, first = fit_probe(x, y, 0.1, config)
    set_seeds(config.seed)
    _, second = fit_probe(x, y, 0.1, config)

    np.testing.assert_array_equal(first.coef_, second.coef_)
    np.testing.assert_array_equal(first.intercept_, second.intercept_)


def test_sweep_is_reproducible(config, data):
    activations, labels, split = data
    first = run_sweep(activations, labels, split, config)
    second = run_sweep(activations, labels, split, config)
    assert first == second


def test_selected_probe_is_reproducible(config, data):
    activations, labels, split = data
    sweep = run_sweep(activations, labels, split, config)
    first = fit_selected_probe(activations, labels, split, sweep["best"], config)
    second = fit_selected_probe(activations, labels, split, sweep["best"], config)

    assert first.to_meta() == second.to_meta()
    np.testing.assert_array_equal(first.classifier.coef_, second.classifier.coef_)


def test_bootstrap_is_reproducible(config, data):
    """The CI is seeded too, so an interval quoted in the README is stable."""
    activations, labels, split = data
    sweep = run_sweep(activations, labels, split, config)
    probe = fit_selected_probe(activations, labels, split, sweep["best"], config)

    test_mask = split == "test"
    scores = probe.score(activations[probe.layer][test_mask])
    y = labels[test_mask]

    first = bootstrap_metrics(y, scores, probe.threshold, 200, 0.95, config.seed)
    second = bootstrap_metrics(y, scores, probe.threshold, 200, 0.95, config.seed)
    assert first == second


def test_bootstrap_changes_with_a_different_seed(config, data):
    """Reproducibility must come from the seed, not from a degenerate resampler."""
    activations, labels, split = data
    sweep = run_sweep(activations, labels, split, config)
    probe = fit_selected_probe(activations, labels, split, sweep["best"], config)

    test_mask = split == "test"
    scores = probe.score(activations[probe.layer][test_mask])
    y = labels[test_mask]

    first = bootstrap_metrics(y, scores, probe.threshold, 200, 0.95, 1)
    second = bootstrap_metrics(y, scores, probe.threshold, 200, 0.95, 2)
    assert first["auroc"]["ci_low"] != second["auroc"]["ci_low"]


def test_tie_break_in_selection_is_deterministic(config):
    """Equal AUROCs must resolve the same way every time, not by dict order."""
    from src.probe import select_best

    rows = [
        {"layer": 20, "C": 0.1, "val_auroc": 0.70},
        {"layer": 8, "C": 1.0, "val_auroc": 0.70},
        {"layer": 8, "C": 0.01, "val_auroc": 0.70},
        {"layer": 14, "C": 0.1, "val_auroc": 0.69},
    ]
    best = select_best(rows)
    assert (best["layer"], best["C"]) == (8, 0.01)
    assert select_best(list(reversed(rows))) == best


def test_evaluate_is_a_pure_function_of_its_inputs(config, data):
    activations, labels, split = data
    sweep = run_sweep(activations, labels, split, config)
    probe = fit_selected_probe(activations, labels, split, sweep["best"], config)
    test_mask = split == "test"
    scores = probe.score(activations[probe.layer][test_mask])

    first = evaluate_at_threshold(labels[test_mask], scores, probe.threshold)
    second = evaluate_at_threshold(labels[test_mask], scores, probe.threshold)
    assert first == second
