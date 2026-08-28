"""Label polarity: the positive class is "incorrect" (DECISIONS.md 004).

Inverting the polarity produces ``1 - AUROC``. At AUROC 0.70 that reads as
0.30 -- a strong negative result, not an obvious bug -- and sends debugging in
entirely the wrong direction. So it is asserted at the boundary and pinned here.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.probe import PolarityError, assert_polarity, fit_probe, probe_scores


@pytest.fixture
def separable_data():
    """Activations carrying a real signal about a binary label.

    Class 1 ("incorrect") sits at a different mean, so a probe should score it
    higher and AUROC should be well above 0.5.
    """
    rng = np.random.RandomState(0)
    n = 200
    labels = np.array([0, 1] * (n // 2))
    x = rng.randn(n, 12)
    x[labels == 1, :3] += 1.4
    return x, labels


def test_positive_class_is_incorrect(config, separable_data):
    """A probe trained with 1 == incorrect must score incorrect rows higher."""
    x, labels = separable_data
    scaler, classifier = fit_probe(x, labels, 1.0, config)
    scores = probe_scores(scaler, classifier, x)
    assert scores[labels == 1].mean() > scores[labels == 0].mean()
    assert roc_auc_score(labels, scores) > 0.5


def test_inverted_labels_give_one_minus_auroc(config, separable_data):
    """The documented failure signature, pinned so it is recognisable.

    Training on inverted labels and scoring against the true ones yields
    exactly 1 - AUROC. If a run ever reports an AUROC well below 0.5, this is
    the first thing to suspect.
    """
    x, labels = separable_data

    scaler, classifier = fit_probe(x, labels, 1.0, config)
    correct_auroc = roc_auc_score(labels, probe_scores(scaler, classifier, x))

    inverted = 1 - labels
    inv_scaler, inv_classifier = fit_probe(x, inverted, 1.0, config)
    inverted_auroc = roc_auc_score(labels, probe_scores(inv_scaler, inv_classifier, x))

    assert inverted_auroc == pytest.approx(1.0 - correct_auroc, abs=1e-9)
    assert inverted_auroc < 0.5 < correct_auroc


def test_assert_polarity_accepts_matching_columns():
    labels = np.array([0, 1, 1, 0])
    correct = np.array([True, False, False, True])
    assert_polarity(labels, correct)


def test_assert_polarity_rejects_inverted_columns():
    """label must be the complement of correct, not a copy of it."""
    labels = np.array([0, 1, 1, 0])
    correct = np.array([False, True, True, False])
    with pytest.raises(PolarityError, match="complement"):
        assert_polarity(labels, correct)


def test_assert_polarity_rejects_single_class():
    """Every-answer-correct means the matcher or the prompt is broken."""
    with pytest.raises(PolarityError, match="only class"):
        assert_polarity(np.zeros(10, dtype=int))


def test_assert_polarity_rejects_non_binary():
    with pytest.raises(PolarityError, match="0/1"):
        assert_polarity(np.array([0, 1, 2]))


def test_label_frame_polarity_matches_the_assertion(config):
    """The extraction stage's labels must satisfy the probe stage's guard."""
    import pandas as pd

    from src.data import label_frame

    frame = pd.DataFrame(
        {
            "question_id": ["a", "b"],
            "question": ["q1", "q2"],
            "question_norm": ["q1", "q2"],
            "answer_value": ["paris", "paris"],
            "aliases": [["Paris"], ["Paris"]],
        }
    )
    labelled = label_frame(frame, ["The answer is Paris.", "Rome, definitely."], config)
    assert labelled["label"].tolist() == [0, 1]
    assert_polarity(labelled["label"].to_numpy(), labelled["correct"].to_numpy())
