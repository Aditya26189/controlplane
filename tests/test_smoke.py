"""End-to-end pipeline smoke: every stage runs and writes every artifact.

Covers the chain in two halves, because the reference model needs a GPU that CI
does not have:

* **Stage 01** runs against the tiny offline Qwen2, confirming that argument
  parsing, data preparation, the equivalence check, batching, labelling and
  persistence all work as one program.
* **Stages 02-05** run against a synthetic extraction with a planted mid-stack
  signal, confirming the sweep, the single test scoring, the economics, the
  latency measurement and both rendered documents.

The synthetic half exists because a randomly initialised model answers nothing
correctly, so its labels are single-class and no probe can be fitted from them.
The planted signal is not evidence about the method -- it is a fixture that
exercises the plumbing. The real numbers come from a GPU run.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import load_config
from src.data import label_frame
from src.extract import save_activations

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_N = 240


def write_config(base_path: Path, destination_dir: Path, **sections) -> Path:
    """Copy config.yaml into a temp dir with paths redirected and edits applied."""
    with base_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["paths"] = {
        "results_dir": str(destination_dir),
        "activations": str(destination_dir / "activations.npz"),
        "labels": str(destination_dir / "labels.parquet"),
        "splits": str(destination_dir / "splits.parquet"),
    }
    for section, changes in sections.items():
        raw[section].update(changes)
    path = destination_dir / "config.yaml"
    destination_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)
    return path


def run_script(script: str, config_path: Path, *extra: str) -> subprocess.CompletedProcess:
    """Run a stage script and fail the test with its output if it errors."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), "--config", str(config_path), *extra],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"{script} exited {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return result


# --------------------------------------------------------------------------- #
# Stages 02-05, against a synthetic extraction
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    """Write a synthetic activations/labels pair with a planted mid-stack signal."""
    results = tmp_path_factory.mktemp("smoke-results")
    config_path = write_config(
        REPO_ROOT / "config.yaml",
        results,
        data={"n_examples": SMOKE_N},
        evaluation={"bootstrap_samples": 100},
        economics={"target_flag_rate": 0.10},
    )
    config = load_config(config_path)

    rng = np.random.RandomState(1729)
    n = SMOKE_N
    split = np.array(["train"] * 144 + ["val"] * 48 + ["test"] * 48)
    correct = rng.rand(n) < 0.6

    activations = {}
    for layer, strength in [(8, 0.1), (11, 0.3), (14, 0.6), (17, 1.0), (20, 0.7), (23, 0.3), (26, 0.1)]:
        x = rng.randn(n, 48).astype(np.float32)
        x[~correct, :6] += strength
        activations[layer] = x

    frame = pd.DataFrame(
        {
            "question_id": [f"q{i:04d}" for i in range(n)],
            "question": [f"Question number {i}?" for i in range(n)],
            "question_norm": [f"question number {i}" for i in range(n)],
            "answer_value": ["Paris"] * n,
            "aliases": [["Paris"] for _ in range(n)],
            "split": split,
        }
    )
    completions = [
        "The answer is Paris." if ok else ("I don't know." if i % 7 == 0 else "Rome.")
        for i, ok in enumerate(correct)
    ]
    labelled = label_frame(frame, completions, config)

    save_activations(activations, labelled["question_id"].tolist(), config.paths.activations)
    labelled.to_parquet(config.paths.labels, index=False)
    frame.to_parquet(config.paths.splits, index=False)

    # extract_meta.json is normally written by stage 01; the fields the later
    # stages read are the timings, the model description and the base rates.
    from src.config import write_json_artifact
    from src.extract import base_rate_summary

    write_json_artifact(
        results / "data_stats.json",
        {
            "data": {
                "dataset": config.data.dataset,
                "dataset_config": config.data.config,
                "split": config.data.split,
                "rows_loaded": 17944,
                "duplicates_dropped": 7983,
                "empty_or_aliasless_dropped": 0,
                "n_examples_requested": n,
                "n_final": n,
                "split_sizes": {name: int((split == name).sum()) for name in ("train", "val", "test")},
                "seed": config.seed,
            }
        },
        config,
    )
    write_json_artifact(
        results / "extract_meta.json",
        {
            "model": {
                "name": "synthetic-fixture",
                "quantization": "none",
                "dtype": "float32",
                "num_hidden_layers": 28,
                "hidden_size": 48,
                "probe_layers": sorted(activations),
                "layer_fractions": list(config.model.layer_fractions),
                "padding_side": "left",
                "pad_token": "<eos>",
                "eos_token": "<eos>",
                "peak_memory_gb_after_load": None,
                "example_prompt": "<QUESTION>",
                "system_prompt": config.prompt.system,
            },
            "equivalence_check": {
                "max_deviation": 4.17e-07,
                "max_relative_l2": 1.8e-07,
                "min_cosine_observed": 1.0,
                "relative_tolerance": 0.10,
                "min_cosine": 0.999,
                "positive_control_ran": True,
                "positive_control_rejected": True,
                "right_padding_control": {
                    "max_relative_l2": 1.357,
                    "min_cosine": 0.036,
                },
                "n_prompts": 4,
                "distinct_prompt_lengths": 4,
            },
            "extraction": {
                "n_examples": n,
                "layers": sorted(activations),
                "hidden_size": 48,
                "batch_size": 8,
                "sort_by_length": True,
                "total_seconds": 120.0,
                "examples_per_second": 2.0,
                "median_prefill_seconds_per_response": 0.02,
                "median_generate_seconds_per_response": 0.4,
                "max_padded_length": 64.0,
            },
            "base_rates": base_rate_summary(labelled),
            "base_rate_by_split": {},
        },
        config,
    )
    return results, config_path


def test_probe_stage_writes_its_artifacts(synthetic_run):
    results, config_path = synthetic_run
    run_script("02_train_probe.py", config_path)

    assert (results / "probe_sweep.json").is_file()
    assert (results / "probe.joblib").is_file()
    assert (results / "probe_test.json").is_file()

    sweep = json.loads((results / "probe_sweep.json").read_text(encoding="utf-8"))
    assert len(sweep["sweep"]) == 7 * 4, "every layer x C combination must appear"
    assert sweep["selected_on"] == "validation"

    probe_test = json.loads((results / "probe_test.json").read_text(encoding="utf-8"))
    assert probe_test["test"]["n"] == 48
    assert probe_test["probe"]["positive_class"] == "incorrect"
    # The planted signal peaks at layer 17; the sweep should find it.
    assert probe_test["probe"]["layer"] == 17


def test_economics_and_latency_stages(synthetic_run):
    results, config_path = synthetic_run
    run_script("02_train_probe.py", config_path)
    run_script("03_economics.py", config_path)
    run_script("04_latency.py", config_path)

    economics = json.loads((results / "economics.json").read_text(encoding="utf-8"))["economics"]
    probe_test = json.loads((results / "probe_test.json").read_text(encoding="utf-8"))

    # The economics stage must reuse the measured numbers, not recompute them.
    assert economics["measured_flag_rate"] == probe_test["test"]["flag_rate"]
    assert economics["measured_recall"] == probe_test["test"]["recall"]
    assert economics["lift"] == pytest.approx(
        probe_test["test"]["recall"] / probe_test["test"]["flag_rate"]
    )
    assert economics["invariance"]["all_equal"] is True

    latency = json.loads((results / "latency.json").read_text(encoding="utf-8"))["latency"]
    assert latency["comparison"]["probe_median_us"] > 0
    assert latency["comparison"]["adds_a_forward_pass"] is False


def test_report_stage_renders_both_documents(synthetic_run, tmp_path):
    results, config_path = synthetic_run
    run_script("02_train_probe.py", config_path)
    run_script("03_economics.py", config_path)
    run_script("04_latency.py", config_path)
    readme = tmp_path / "README.md"
    run_script("05_report.py", config_path, "--readme", str(readme))

    assert (results / "RESULTS.md").is_file()
    assert (results / "layer_sweep.png").is_file()
    assert (results / "roc_curve.png").is_file()
    assert readme.is_file()

    text = (results / "RESULTS.md").read_text(encoding="utf-8")
    for heading in [
        "## 1. Run metadata",
        "## 2. Dataset",
        "## 3. Layer sweep",
        "## 4. Test results",
        "## 5. Three policies",
        "## 6. Headline",
        "## 7. Latency",
        "## 8. Secondary validation",
        "## 9. Negative control",
        "## 10. Limitations",
    ]:
        assert heading in text, f"RESULTS.md is missing {heading!r} (SPEC.md §13)"

    rendered = readme.read_text(encoding="utf-8")
    assert "{{" not in rendered, "an unsubstituted placeholder reached README.md"
    assert "%%" not in rendered, "double percent sign in rendered README"
    assert "Fill this in at Stage 7" not in rendered, "template header leaked into README"


def test_report_refuses_an_unknown_placeholder(synthetic_run):
    """A blank in a published README is worse than a crash."""
    from src.report import load_artifacts, render_readme

    results, config_path = synthetic_run
    run_script("02_train_probe.py", config_path)
    run_script("03_economics.py", config_path)
    run_script("04_latency.py", config_path)

    config = load_config(config_path)
    artifacts = load_artifacts(config)
    with pytest.raises(KeyError, match="no value"):
        render_readme("---\nvalue: {{not_a_real_placeholder}}\n", artifacts, config)


# --------------------------------------------------------------------------- #
# Stage 01, against the tiny offline model
# --------------------------------------------------------------------------- #


def dataset_is_available() -> bool:
    """True when TriviaQA can be loaded (from cache or network)."""
    try:
        from datasets import load_dataset

        load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation[:2]")
        return True
    except Exception:  # noqa: BLE001 - absence is a skip, not a failure
        return False


@pytest.mark.skipif(
    not dataset_is_available(), reason="TriviaQA not available offline or online"
)
def test_extract_stage_runs_end_to_end(tmp_path, tiny_model, tiny_tokenizer):
    """Stage 01 as one program, on a 41k-parameter model.

    The base-rate band is widened for this test only: a randomly initialised
    model answers nothing correctly, so the real band would (correctly) abort.
    The band stays at its configured value everywhere else, including in
    ``run_all.py --smoke``.
    """
    model_dir = tmp_path / "tiny-model"
    tiny_model.save_pretrained(model_dir)
    tiny_tokenizer.save_pretrained(model_dir)

    results = tmp_path / "results"
    config_path = write_config(
        REPO_ROOT / "config.yaml",
        results,
        model={
            "name": str(model_dir),
            "quantization": "none",
            "dtype": "float32",
            "device_map": "cpu",
        },
        data={"n_examples": 16},
        generation={"max_new_tokens": 4, "batch_size": 4},
        labeling={"base_rate_min": 0.0, "base_rate_max": 1.0},
    )

    run_script("01_extract.py", config_path)

    assert (results / "splits.parquet").is_file()
    assert (results / "data_stats.json").is_file()
    assert (results / "activations.npz").is_file()
    assert (results / "labels.parquet").is_file()
    assert (results / "extract_meta.json").is_file()

    meta = json.loads((results / "extract_meta.json").read_text(encoding="utf-8"))
    check = meta["equivalence_check"]
    assert check["max_relative_l2"] < check["relative_tolerance"]
    assert check["min_cosine_observed"] > check["min_cosine"]
    # The positive control must have run and rejected right padding, so passing
    # is a demonstration rather than an assertion (DECISIONS.md 014).
    assert check["positive_control_rejected"] is True
    assert not (
        check["right_padding_control"]["max_relative_l2"] < check["relative_tolerance"]
    )
    assert meta["model"]["padding_side"] == "left"
    assert meta["extraction"]["n_examples"] == 16

    labels = pd.read_parquet(results / "labels.parquet")
    assert len(labels) == 16
    assert set(labels.columns) >= {"question_id", "completion", "correct", "label", "abstained", "split"}

    activations = np.load(results / "activations.npz", allow_pickle=True)
    layer_keys = [k for k in activations.files if k.startswith("layer_")]
    assert len(layer_keys) == len(meta["model"]["probe_layers"])
    for key in layer_keys:
        assert np.isfinite(activations[key].astype(np.float32)).all()
    assert [str(q) for q in activations["question_id"]] == labels["question_id"].tolist()
