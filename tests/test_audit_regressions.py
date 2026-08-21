"""Regressions for the defects found in the Stage 7 audit.

One test per finding, each named for the failure it prevents coming back.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import _json_default, load_config
from src.data import label_frame
from src.extract import base_rate_summary, select_equivalence_prompts
from src.report import config_hash_consistency, fmt_ratio

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- smoke mode must not crash, and must not clobber a real run ------------ #


def _run_all_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_all_module", REPO_ROOT / "scripts" / "run_all.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_config_builds_when_results_dir_does_not_exist(tmp_path, monkeypatch):
    """A clean clone has no results/ -- git does not track empty directories.

    This is the first thing the Stage 7 clone-and-run gate does, and it used to
    raise FileNotFoundError before any stage started.
    """
    run_all = _run_all_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_all, "REPO_ROOT", tmp_path)

    source = tmp_path / "config.yaml"
    source.write_text((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    assert not (tmp_path / "results").exists()

    destination = tmp_path / "results" / "smoke" / "config.smoke.yaml"
    run_all.build_smoke_config(source, destination)
    assert destination.is_file()


def test_smoke_config_redirects_every_output_path(tmp_path):
    """A smoke run must not overwrite artifacts that cost a GPU hour."""
    run_all = _run_all_module()
    source = tmp_path / "config.yaml"
    source.write_text((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    destination = tmp_path / "out" / "config.smoke.yaml"
    run_all.build_smoke_config(source, destination)
    with destination.open(encoding="utf-8") as fh:
        smoke = yaml.safe_load(fh)

    real = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    for key, path in smoke["paths"].items():
        assert run_all.SMOKE_SUBDIR in path, f"{key} still points outside the smoke dir"
        assert path != real["paths"][key]
    assert smoke["data"]["n_examples"] == run_all.SMOKE_N_EXAMPLES
    # The safety gate must survive smoke mode.
    assert smoke["labeling"]["base_rate_min"] == real["labeling"]["base_rate_min"]
    assert smoke["labeling"]["base_rate_max"] == real["labeling"]["base_rate_max"]


# --- strict exact match: "not recorded" is not "nothing matched" ----------- #


def _two_row_frame():
    return pd.DataFrame(
        {
            "question_id": ["a", "b"],
            "question": ["q1", "q2"],
            "question_norm": ["q1", "q2"],
            "answer_value": ["Paris", "Paris"],
            "aliases": [["Paris"], ["Paris"]],
        }
    )


def test_strict_em_off_reports_none_not_a_fabricated_gap(config):
    """With record_strict_em off, the audit figures must be None.

    Setting the column to False made every answer look strictly wrong, so the
    summary reported a 100-point lenient-vs-strict gap that never happened --
    and RESULTS.md then fired SPEC.md §2's ">10 point" warning about it.
    """
    off = replace(config, labeling=replace(config.labeling, record_strict_em=False))
    summary = base_rate_summary(label_frame(_two_row_frame(), ["Paris", "Paris"], off))

    assert summary["strict_em_recorded"] is False
    assert summary["accuracy_strict_em"] is None
    assert summary["lenient_minus_strict_accuracy"] is None
    assert summary["base_rate_incorrect_strict_em"] is None
    assert summary["accuracy_lenient"] == pytest.approx(1.0)


def test_strict_em_on_still_reports_a_real_gap(config):
    summary = base_rate_summary(
        label_frame(_two_row_frame(), ["The answer is Paris.", "Paris"], config)
    )
    assert summary["strict_em_recorded"] is True
    assert summary["accuracy_strict_em"] == pytest.approx(0.5)
    assert summary["lenient_minus_strict_accuracy"] == pytest.approx(0.5)


# --- the equivalence check must batch maximum padding --------------------- #


def test_equivalence_prompts_span_the_length_distribution(tiny_tokenizer):
    """Taking the first four prompts gave a batch with almost no padding.

    The check's sensitivity scales with padding: four near-equal prompts pad by
    a few tokens and would pass even with a subtly wrong read position.
    """
    prompts = [" ".join(["what"] * n) for n in [5, 6, 7, 8, 9, 40, 41]]
    selected = select_equivalence_prompts(tiny_tokenizer, prompts, 4)

    lengths = [len(tiny_tokenizer(p)["input_ids"]) for p in selected]
    naive = [len(tiny_tokenizer(p)["input_ids"]) for p in prompts[:4]]

    assert len(selected) == 4
    assert max(lengths) - min(lengths) > max(naive) - min(naive)
    assert min(lengths) == min(len(tiny_tokenizer(p)["input_ids"]) for p in prompts)
    assert max(lengths) == max(len(tiny_tokenizer(p)["input_ids"]) for p in prompts)


def test_equivalence_prompt_selection_handles_small_inputs(tiny_tokenizer):
    prompts = ["what is paris", "who wrote the iliad ?"]
    assert select_equivalence_prompts(tiny_tokenizer, prompts, 4) == prompts


# --- JSON artifacts must not stringify numbers ---------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.float32(0.25), 0.25),
        (np.int64(5), 5),
        (np.int32(5), 5),
        (np.bool_(True), True),
    ],
)
def test_numpy_scalars_serialise_as_numbers(value, expected):
    """default=str silently turned float32/int64 into JSON strings.

    They do not subclass Python builtins, so json fell through to the default
    hook. The value still looked like a number in the file and broke a format
    string three stages later.
    """
    assert json.loads(json.dumps(value, default=_json_default)) == expected


def test_unserialisable_value_raises_rather_than_stringifying():
    with pytest.raises(TypeError, match="not JSON-serialisable"):
        json.dumps(object(), default=_json_default)


# --- cross-stage config hash consistency ---------------------------------- #


def test_config_hash_mismatch_is_detected():
    """Stages load the config independently; nothing forced them to agree.

    RESULTS.md would then quote one hash beside numbers produced under two,
    which is precisely the traceability claim invariant 7 makes.
    """
    artifacts = {
        "extract_meta": {"provenance": {"config_hash": "aaaa1111"}},
        "probe_test": {"provenance": {"config_hash": "bbbb2222"}},
    }
    result = config_hash_consistency(artifacts)
    assert result["consistent"] is False
    assert result["distinct"] == ["aaaa1111", "bbbb2222"]


def test_config_hash_agreement_is_detected():
    artifacts = {
        "extract_meta": {"provenance": {"config_hash": "aaaa1111"}},
        "probe_test": {"provenance": {"config_hash": "aaaa1111"}},
    }
    assert config_hash_consistency(artifacts)["consistent"] is True


# --- report formatting must not crash on undefined values ----------------- #


def test_ratio_formatting_survives_a_missing_denominator():
    assert fmt_ratio(None) == "n/a"
    assert fmt_ratio(5.53e-04) == "5.53e-04"


# --- README must describe the split it names ------------------------------ #


def test_readme_base_rate_is_the_test_sets(tmp_path):
    """The README table row reads "n = <n_test> held-out questions".

    Quoting a whole-dataset base rate underneath it described a different set of
    examples than the row claimed, and disagreed with RESULTS.md §4.
    """
    from src.report import readme_values

    config = load_config(REPO_ROOT / "config.yaml")
    artifacts = {
        "extract_meta": {
            "model": {
                "name": "m",
                "quantization": "nf4",
                "dtype": "bfloat16",
                "num_hidden_layers": 28,
                "hidden_size": 3584,
            },
            "base_rates": {
                "base_rate_incorrect": 0.40,          # whole dataset
                "base_rate_incorrect_strict_em": 0.45,
            },
            "extraction": {"total_seconds": 100.0},
        },
        "data_stats": {"data": {"dataset": "d", "dataset_config": "c"}},
        "probe_test": {
            "provenance": {"seed": 1729},
            "test": {
                "n": 600,
                "auroc": 0.7,
                "flag_rate": 0.05,
                "recall": 0.6,
                "precision": 0.4,
                "base_rate": 0.33,                    # test set — differs on purpose
            },
            "bootstrap": {
                "auroc": {"ci_low": 0.6, "ci_high": 0.8},
                "recall": {"ci_low": 0.5, "ci_high": 0.7},
                "lift": {"ci_low": 10.0, "ci_high": 14.0},
            },
            "probe": {"layer": 17, "C": 0.01},
            "abstention": {
                "mean_score_abstained": 1.0,
                "mean_score_not_abstained": 0.0,
                "auroc_predicting_abstention": 0.7,
                "abstention_rate": 0.03,
            },
            "strict_em": {"test_accuracy_strict": 0.55},
        },
        "probe_sweep": {"sweep": [{"layer": 17, "C": 0.01, "val_auroc": 0.71}], "best": {"layer": 17, "C": 0.01}},
        "economics": {
            "economics": {
                "lift": 12.0,
                "reference_error_rate": 0.03,
                "policies": [
                    {"policy": "judge_everything", "judge_calls": 1e6, "errors_caught": 3e4, "relative_cost": 20.0},
                    {"policy": "random_sample", "judge_calls": 5e4, "errors_caught": 1500.0, "relative_cost": 1.0},
                    {"policy": "probe_triggered", "judge_calls": 5e4, "errors_caught": 1.8e4, "relative_cost": 1.0},
                ],
            }
        },
        "latency": {
            "latency": {
                "comparison": {
                    "probe_median_us": 200.0,
                    "generation_median_ms": 400.0,
                    "probe_over_generation": 5e-4,
                },
                "device": {"device_name": "Tesla T4"},
            }
        },
    }
    values = readme_values(artifacts, config)
    assert values["base_rate"] == "0.330", "must be the test-set rate the row names"
    assert values["strict_em_base_rate"] == "0.450"  # 1 - test_accuracy_strict
    assert values["n_test"] == "600"


# --- a failed base-rate gate must not discard the GPU hour ----------------- #


def test_extract_persists_artifacts_before_the_base_rate_gate():
    """The gate judges the labels; the activations cost 40-70 minutes.

    assert_base_rate used to raise before save_activations, so a degenerate
    label distribution destroyed the whole extraction and you had to re-run the
    GPU to investigate why the labels looked wrong.
    """
    source = (REPO_ROOT / "scripts" / "01_extract.py").read_text(encoding="utf-8")
    save_at = source.index("save_activations(acts")
    meta_at = source.index('config.results_path("extract_meta.json")')
    gate_at = source.rindex("assert_base_rate(labelled, config)")

    assert save_at < gate_at, "activations must be saved before the gate can raise"
    assert meta_at < gate_at, "extract_meta must be written before the gate can raise"


# --- the dirty flag must describe the code, not the pipeline's own output --- #


def test_dirty_flag_ignores_the_results_directory(monkeypatch):
    """The pipeline writes into results/, a committed path.

    On the first real run that made stage 01 dirty the tree and every later
    artifact recorded dirty=true regardless of the code, which is the opposite
    of what the flag is for.
    """
    import src.config as config_module

    monkeypatch.setattr(
        config_module,
        "_git",
        lambda *args: "?? results/\n?? results/probe_test.json",
    )
    assert config_module.working_tree_changes("results") == []


def test_dirty_flag_still_reports_real_code_changes(monkeypatch):
    import src.config as config_module

    monkeypatch.setattr(
        config_module, "_git", lambda *args: " M src/extract.py\n?? results/x.json"
    )
    assert config_module.working_tree_changes("results") == ["src/extract.py"]


def test_porcelain_parsing_survives_the_stripped_first_line(monkeypatch):
    """_git() strips the whole output, removing the leading space of line 1 only.

    A fixed-offset slice therefore ate one character of the first path and no
    others -- ' M src/config.py' became 'rc/config.py'. Caught while verifying
    the fix above, which had exactly that bug.
    """
    import src.config as config_module

    # Exactly what _git returns: leading space of the first line already gone.
    monkeypatch.setattr(
        config_module,
        "_git",
        lambda *args: "M src/config.py\n M src/evaluate.py\nMM src/report.py",
    )
    assert config_module.working_tree_changes("results") == [
        "src/config.py",
        "src/evaluate.py",
        "src/report.py",
    ]


def test_porcelain_parsing_handles_renames_and_quotes(monkeypatch):
    import src.config as config_module

    monkeypatch.setattr(
        config_module,
        "_git",
        lambda *args: 'R  src/old.py -> src/new.py\n?? "src/with space.py"',
    )
    assert config_module.working_tree_changes("results") == [
        "src/new.py",
        "src/with space.py",
    ]


# --- the ceiling and the bootstrap must not appear to contradict ----------- #


def test_lift_never_exceeds_ceiling_within_a_resample():
    """lift/ceiling == precision <= 1, so they cannot cross inside one resample.

    The point-estimate ceiling (2.575) is below the lift CI upper bound (2.655)
    on the real run, which reads as a contradiction until you see that the
    ceiling is resampled too.
    """
    import numpy as np

    from src.evaluate import bootstrap_metrics, evaluate_at_threshold

    rng = np.random.RandomState(0)
    labels = (rng.rand(600) < 0.39).astype(int)
    scores = rng.randn(600) + labels * 1.2
    threshold = float(np.quantile(scores, 0.94))

    point = evaluate_at_threshold(labels, scores, threshold)
    assert point["ceiling"] == pytest.approx(1.0 / point["base_rate"])
    assert point["lift"] <= point["ceiling"] + 1e-9
    assert point["lift"] / point["ceiling"] == pytest.approx(point["precision"])

    boot = bootstrap_metrics(labels, scores, threshold, 200, 0.95, 1729)
    assert boot["ceiling"]["ci_low"] is not None
    assert boot["ceiling"]["point"] == pytest.approx(point["ceiling"])


def test_limitations_names_the_ceiling():
    """SPEC.md §13 calls the limitations section 'not optional and not boilerplate'.

    It omitted the single most important caveat about the headline number.
    """
    from src.config import load_config
    from src.report import limitations

    config = load_config(REPO_ROOT / "config.yaml")
    artifacts = {
        "extract_meta": {
            "model": {"name": "m"},
            "base_rates": {"lenient_minus_strict_accuracy": 0.48},
        },
        "data_stats": {"data": {"dataset": "d", "dataset_config": "c"}},
        "probe_test": {
            "provenance": {"seed": 1729},
            "test": {"n": 600, "flag_rate": 0.0617, "auroc": 0.85},
            "auroc_floor": {"below_floor": False, "floor": 0.55},
        },
        "economics": {
            "economics": {
                "ceiling": {
                    "measured_base_rate": 0.3883,
                    "max_attainable_lift": 2.5751,
                    "fraction_of_ceiling_achieved": 0.892,
                }
            }
        },
    }
    text = " ".join(limitations(artifacts, config))
    assert "ceiling" in text.lower() or "bounded by this benchmark" in text.lower()
    assert "2.58" in text
    assert "AUROC" in text, "must name AUROC as the transferable quantity"
