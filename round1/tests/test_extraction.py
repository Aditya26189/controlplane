"""Extraction: batch ordering, shapes, persistence, and the base-rate gate.

Runs the whole extraction loop on the tiny offline Qwen2. The completions are
nonsense -- the weights are random -- but the plumbing under test is the same
plumbing the T4 run uses: length-sorted batching, order restoration, shape and
finiteness assertions, and fp16 round-tripping.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.data import label_frame, normalize_question
from src.extract import (
    BaseRateError,
    assert_base_rate,
    base_rate_summary,
    load_activations,
    run_extraction,
    save_activations,
)
from src.model import configure_tokenizer

LAYERS = (1, 2, 3, 4)


@pytest.fixture
def left_padded(tiny_tokenizer):
    configure_tokenizer(tiny_tokenizer)
    return tiny_tokenizer


@pytest.fixture
def small_config(config):
    """Config with a batch size that forces several ragged batches."""
    return replace(config, generation=replace(config.generation, batch_size=3))


@pytest.fixture
def questions_frame():
    """Nine questions of deliberately varied length, so batches are ragged."""
    questions = [
        "who wrote the iliad ?",
        "what is the capital of france ?",
        "what is paris",
        "who wrote the iliad of homer in the capital of france ?",
        "what is homer",
        "who is homer of the iliad ?",
        "what is the capital",
        "who wrote the question of paris and homer in france ?",
        "what is france",
    ]
    return pd.DataFrame(
        {
            "question_id": [f"q{i}" for i in range(len(questions))],
            "question": questions,
            "question_norm": [normalize_question(q) for q in questions],
            "answer_value": ["homer"] * len(questions),
            "aliases": [["homer", "Homer"] for _ in questions],
            "split": ["train"] * 6 + ["val", "test", "test"],
        }
    )


def test_run_extraction_shapes_and_order(tiny_model, left_padded, small_config, questions_frame):
    """Activations come back in frame order, one row per question, all finite."""
    acts, labelled, meta = run_extraction(
        tiny_model, left_padded, questions_frame, small_config, LAYERS, progress=False
    )
    n = len(questions_frame)
    for layer in LAYERS:
        assert acts[layer].shape == (n, tiny_model.config.hidden_size)
        assert np.isfinite(acts[layer]).all()
    assert labelled["question_id"].tolist() == questions_frame["question_id"].tolist()
    assert meta["n_examples"] == n
    assert meta["layers"] == list(LAYERS)
    assert meta["median_generate_seconds_per_response"] > 0


def test_length_sorting_does_not_reorder_the_output(
    tiny_model, left_padded, small_config, questions_frame
):
    """Sorted and unsorted batching must give the same rows in the same order.

    Length sorting is a padding-waste optimisation. If it ever changed which
    activation sits beside which label, every downstream number would be
    garbage while looking entirely normal.
    """
    unsorted_config = replace(
        small_config, generation=replace(small_config.generation, sort_by_length=False)
    )
    sorted_acts, sorted_labels, _ = run_extraction(
        tiny_model, left_padded, questions_frame, small_config, LAYERS, progress=False
    )
    plain_acts, plain_labels, _ = run_extraction(
        tiny_model, left_padded, questions_frame, unsorted_config, LAYERS, progress=False
    )
    assert sorted_labels["question_id"].tolist() == plain_labels["question_id"].tolist()
    assert sorted_labels["completion"].tolist() == plain_labels["completion"].tolist()
    for layer in LAYERS:
        np.testing.assert_allclose(
            sorted_acts[layer], plain_acts[layer], atol=1e-3, rtol=0
        )


def test_labels_have_the_documented_polarity(
    tiny_model, left_padded, small_config, questions_frame
):
    """label == 1 means the answer was wrong (DECISIONS.md 004)."""
    _, labelled, _ = run_extraction(
        tiny_model, left_padded, questions_frame, small_config, LAYERS, progress=False
    )
    assert set(labelled["label"].unique()) <= {0, 1}
    assert (labelled["label"] == (~labelled["correct"]).astype(int)).all()


# --- persistence ----------------------------------------------------------- #


def test_activations_round_trip_through_npz(tmp_path):
    rng = np.random.RandomState(0)
    acts = {8: rng.randn(5, 16).astype(np.float32), 14: rng.randn(5, 16).astype(np.float32)}
    ids = [f"q{i}" for i in range(5)]
    path = save_activations(acts, ids, tmp_path / "activations.npz")

    back = load_activations(path, expected_question_ids=ids)
    assert set(back) == {8, 14}
    for layer in acts:
        # fp16 storage: agreement to ~1e-3 is the documented precision cost.
        np.testing.assert_allclose(back[layer], acts[layer], atol=1e-2, rtol=0)


def test_loading_rejects_misaligned_question_ids(tmp_path):
    """A stale npz beside a fresh labels frame must fail, not silently mispair."""
    acts = {8: np.zeros((3, 4), dtype=np.float32)}
    path = save_activations(acts, ["a", "b", "c"], tmp_path / "activations.npz")
    with pytest.raises(AssertionError, match="not aligned"):
        load_activations(path, expected_question_ids=["a", "b", "z"])


# --- the base-rate gate ---------------------------------------------------- #


def make_labelled(config, n_correct: int, n_total: int) -> pd.DataFrame:
    """Build a labelled frame with a chosen accuracy."""
    completions = ["homer"] * n_correct + ["definitely not the answer"] * (n_total - n_correct)
    frame = pd.DataFrame(
        {
            "question_id": [f"q{i}" for i in range(n_total)],
            "question": [f"question {i}" for i in range(n_total)],
            "question_norm": [f"question {i}" for i in range(n_total)],
            "answer_value": ["homer"] * n_total,
            "aliases": [["homer"] for _ in range(n_total)],
            "split": ["train"] * n_total,
        }
    )
    return label_frame(frame, completions, config)


def test_base_rate_gate_accepts_a_healthy_distribution(config):
    labelled = make_labelled(config, n_correct=60, n_total=100)
    summary = assert_base_rate(labelled, config)
    assert summary["accuracy_lenient"] == pytest.approx(0.60)
    assert summary["base_rate_incorrect"] == pytest.approx(0.40)


@pytest.mark.parametrize("n_correct", [0, 10, 95, 100])
def test_base_rate_gate_rejects_a_degenerate_distribution(config, n_correct):
    """0/20 or 20/20 correct means the prompt or the matcher is broken."""
    labelled = make_labelled(config, n_correct=n_correct, n_total=100)
    with pytest.raises(BaseRateError, match="sanity band"):
        assert_base_rate(labelled, config)


def test_base_rate_summary_reports_both_matching_rules(config):
    """The strict-EM audit column travels beside the lenient one (SPEC.md §2)."""
    labelled = make_labelled(config, n_correct=50, n_total=100)
    summary = base_rate_summary(labelled)
    assert summary["accuracy_lenient"] >= summary["accuracy_strict_em"]
    assert "lenient_minus_strict_accuracy" in summary
    assert summary["base_rate_incorrect"] == pytest.approx(
        1.0 - summary["accuracy_lenient"]
    )
