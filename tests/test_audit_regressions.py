"""Regressions for the defects found in the Stage 7 audit.

One test per finding, each named for the failure it prevents coming back.
"""

import json
import re
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


# --- the test-scoring audit trail (DECISIONS.md 016) ---------------------- #


def test_scoring_log_appends_never_replaces(tmp_path):
    """A disappointing scoring cannot be erased by re-running until one improves."""
    from src.evaluate import append_test_scoring

    path = tmp_path / "test_scoring_log.json"

    first = append_test_scoring(path, {"auroc": 0.8545, "selected_C": 0.001})
    assert first["n_scorings"] == 1
    path.write_text(json.dumps(first), encoding="utf-8")

    second = append_test_scoring(path, {"auroc": 0.9, "selected_C": 1e-05})
    assert second["n_scorings"] == 2
    assert second["scorings"][0]["auroc"] == 0.8545, "history must survive"
    assert second["scorings"][1]["auroc"] == 0.9


def test_scoring_log_refuses_to_overwrite_a_corrupt_file(tmp_path):
    """Losing the history silently would defeat the point of keeping it."""
    from src.evaluate import append_test_scoring

    path = tmp_path / "test_scoring_log.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        append_test_scoring(path, {"auroc": 0.9})


def test_report_discloses_repeat_scorings():
    """RESULTS.md must say so, rather than leaving it to the log file."""
    from src.config import load_config
    from src.report import render_results_md

    config = load_config(REPO_ROOT / "config.yaml")
    artifacts = _minimal_artifacts()
    artifacts["test_scoring_log"] = {
        "n_scorings": 2,
        "scorings": [
            {"selected_layer": 23, "selected_C": 0.001, "auroc": 0.8545,
             "flag_rate": 0.0617, "recall": 0.1416, "lift": 2.297,
             "config_hash": "cbac792afcf74bc3"},
            {"selected_layer": 20, "selected_C": 1e-05, "auroc": 0.87,
             "flag_rate": 0.06, "recall": 0.16, "lift": 2.4,
             "config_hash": "c429ce5e92da9a22"},
        ],
    }
    text = render_results_md(artifacts, config)
    assert "has been scored 2 times" in text
    assert "cbac792afcf74bc3" in text, "the earlier scoring must remain visible"
    assert "0.8545" in text
    assert "scored once" not in text, "a repeat scoring must not be described as one"


def test_report_warns_when_the_winner_is_at_a_grid_boundary():
    from src.config import load_config
    from src.report import render_results_md

    config = load_config(REPO_ROOT / "config.yaml")
    artifacts = _minimal_artifacts()
    artifacts["probe_sweep"]["winner_at_grid_boundary"] = True
    text = render_results_md(artifacts, config)
    assert "edge of the grid" in text
    assert "boundary is not an optimum" in text.lower()


def _minimal_artifacts():
    """Smallest artifact set render_results_md will accept."""
    return {
        "data_stats": {"data": {
            "dataset": "d", "dataset_config": "c", "split": "validation",
            "rows_loaded": 17944, "duplicates_dropped": 7983,
            "empty_or_aliasless_dropped": 0, "n_final": 3000,
            "split_sizes": {"train": 1800, "val": 600, "test": 600},
        }},
        "extract_meta": {
            "model": {"name": "m", "quantization": "nf4", "dtype": "bfloat16",
                      "num_hidden_layers": 28, "hidden_size": 3584},
            "equivalence_check": {
                "max_relative_l2": 0.0156, "relative_tolerance": 0.10,
                "min_cosine_observed": 0.99988, "min_cosine": 0.999,
                "n_prompts": 4, "right_padding_control": None,
            },
            "extraction": {"n_examples": 3000, "total_seconds": 8304.0,
                           "examples_per_second": 0.36, "batch_size": 8},
            "base_rates": {"accuracy_lenient": 0.594, "accuracy_strict_em": 0.106,
                           "base_rate_incorrect": 0.406,
                           "lenient_minus_strict_accuracy": 0.488},
        },
        "probe_sweep": {
            "sweep": [{"layer": 23, "C": 0.001, "val_auroc": 0.8377}],
            "best": {"layer": 23, "C": 0.001},
            "C_grid": [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0],
        },
        "probe_test": {
            "provenance": {"seed": 1729, "config_hash": "h", "git_commit": "c",
                           "dirty": False, "timestamp_utc": "t", "python": "3.12",
                           "libraries": {"torch": "2.10", "transformers": "5.0"},
                           "device": {"device_name": "Tesla T4"}},
            "test": {"n": 600, "auroc": 0.8545, "flag_rate": 0.0617,
                     "recall": 0.1416, "precision": 0.8919, "base_rate": 0.3883,
                     "tp": 33, "fp": 4, "fn": 200, "tn": 363,
                     "n_flagged": 37, "n_incorrect": 233, "lift": 2.297},
            "bootstrap": {"auroc": {"ci_low": 0.82, "ci_high": 0.89},
                          "flag_rate": {"ci_low": 0.04, "ci_high": 0.08},
                          "recall": {"ci_low": 0.10, "ci_high": 0.19},
                          "precision": {"ci_low": 0.7, "ci_high": 1.0},
                          "lift": {"ci_low": 2.02, "ci_high": 2.65},
                          "ci": 0.95, "n_samples": 1000},
            "probe": {"layer": 23, "C": 0.001, "threshold": 2.794,
                      "target_flag_rate": 0.05, "val_flag_rate": 0.05,
                      "val_auroc": 0.8377},
            "auroc_floor": {"below_floor": False, "floor": 0.55},
            "abstention": {"underpowered": True, "abstention_rate": 0.0,
                           "n_abstained": 0, "n": 600, "min_rate_to_report": 0.02},
            "roc": {"fpr": [0, 1], "tpr": [0, 1], "operating_point": None},
        },
        "economics": {"economics": {
            "n_responses": 1000000, "reference_error_rate": 0.03,
            "judge_accuracy": 1.0, "measured_flag_rate": 0.0617,
            "measured_recall": 0.1416, "lift": 2.297,
            "policies": [
                {"policy": "judge_everything", "label": "Judge everything",
                 "judge_calls": 1e6, "coverage": 1.0, "errors_caught": 30000.0,
                 "relative_cost": 16.2},
                {"policy": "random_sample", "label": "Random 6.2% sample",
                 "judge_calls": 61700.0, "coverage": 0.0617,
                 "errors_caught": 1851.0, "relative_cost": 1.0},
                {"policy": "probe_triggered", "label": "Probe-triggered",
                 "judge_calls": 61700.0, "coverage": 1.0,
                 "errors_caught": 4249.0, "relative_cost": 1.0},
            ],
            "invariance": {"error_rates_tested": [0.03], "judge_accuracies_tested": [1.0],
                           "lifts": [{"lift": 2.297}], "all_equal": True, "spread": 0.0},
        }},
        "latency": {"latency": {
            "comparison": {"probe_median_us": 258.5, "probe_p95_us": 334.3,
                           "generation_median_ms": 2518.4, "prefill_median_ms": 353.2,
                           "probe_over_generation": 1.03e-4, "probe_over_prefill": 7.3e-4},
            "device": {"device_name": "Tesla T4"}, "versions": {"torch": "2.10"},
            "quantization": "nf4",
        }},
    }


# --- the "scored once" claim must not creep back into published prose ------ #

FORBIDDEN_ONCE = re.compile(
    r"(scored|opened|touched)[^.\n]{0,25}(exactly )?once", re.IGNORECASE
)


@pytest.mark.parametrize(
    "relative_path",
    ["README_TEMPLATE.md", "SPEC.md", "CLAUDE.md", "scripts/build_notebooks.py"],
)
def test_published_prose_does_not_claim_a_single_scoring(relative_path):
    """The test set has been scored three times; nothing shipped may say once.

    DECISIONS.md 016 said the claim "must not be repeated anywhere" and it then
    survived in eight places, two of which rendered into the published README
    and the screen-recorded notebook. This is the guard against that returning.

    CLAUDE.md and DECISIONS.md may *describe* the retired wording, so a line is
    only a violation if it asserts the claim rather than quoting its history.
    """
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in text.splitlines()
        if FORBIDDEN_ONCE.search(line)
        and "once read" not in line
        and "original form" not in line
        and "no longer" not in line
    ]
    assert not offenders, f"{relative_path} still claims a single scoring: {offenders}"


def test_rendered_readme_does_not_claim_a_single_scoring():
    """The template renders into a public, judged artifact."""
    text = (REPO_ROOT / "README_TEMPLATE.md").read_text(encoding="utf-8")
    assert "Test was scored once" not in text
    assert "opened once" not in text
