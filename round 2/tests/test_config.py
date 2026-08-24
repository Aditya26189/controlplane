"""Config loading, hashing, layer resolution and provenance.

The Phase 0 gate is two claims: the config hash is stable across runs, and
``provenance()`` reports ``dirty`` correctly after a file is touched. Both are
asserted here, along with the load-time invariant assertions that make several
of ``CLAUDE.md``'s silent failure modes into crashes.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.config import (
    Config,
    ConfigError,
    _from_mapping,
    load_config,
    provenance,
    project_root,
    read_json_artifact,
    repo_root,
    working_tree_changes,
    write_json_artifact,
)


@pytest.fixture(scope="module")
def raw_config(project_root: Path) -> dict:
    """The config file as plain data, for tests that need to corrupt one key."""
    with (project_root / "config.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_mutated(raw: dict, path: str, value) -> Config:
    """Load a config with one dotted key replaced, without touching the file."""
    data = copy.deepcopy(raw)
    cursor = data
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return _from_mapping(Config, data)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def test_committed_config_loads(config: Config) -> None:
    """The committed config.yaml satisfies every load-time assertion."""
    assert config.seed > 0
    assert config.workload.name
    assert len(config.evalsets) == 5
    assert set(config.profiles) == {
        "customer_support",
        "internal_knowledge",
        "decision_support",
    }


def test_unknown_key_is_an_error(raw_config: dict) -> None:
    """A mistyped knob must crash, not be silently dropped.

    Silently dropping it produces a run whose settings differ from the config
    recorded beside its results, with nothing raised.
    """
    data = copy.deepcopy(raw_config)
    data["workload"]["montly_interactions"] = 1000  # deliberate typo
    with pytest.raises(ConfigError, match="unknown key"):
        _from_mapping(Config, data)


def test_missing_key_is_an_error(raw_config: dict) -> None:
    data = copy.deepcopy(raw_config)
    del data["workload"]["base_error_rate"]
    with pytest.raises(ConfigError, match="missing required key"):
        _from_mapping(Config, data)


def test_wrong_type_is_an_error(raw_config: dict) -> None:
    with pytest.raises(ConfigError, match="expected an integer"):
        _load_mutated(raw_config, "workload.monthly_interactions", "200000")


def test_optional_field_may_be_omitted(config: Config) -> None:
    """pad_tokens is declared only for the long-context set."""
    by_id = {spec.id: spec for spec in config.evalsets}
    assert by_id["triviaqa-600"].pad_tokens is None
    assert by_id["triviaqa-longctx-600"].pad_tokens == (4000, 16000)


def test_overrides_must_name_an_existing_key(project_root: Path) -> None:
    with pytest.raises(ConfigError, match="no such config"):
        load_config(project_root / "config.yaml", overrides={"validation.nope": 1})


# --------------------------------------------------------------------------- #
# Hashing — the Phase 0 gate
# --------------------------------------------------------------------------- #


def test_config_hash_is_stable_within_a_process(project_root: Path) -> None:
    a = load_config(project_root / "config.yaml")
    b = load_config(project_root / "config.yaml")
    assert a.config_hash == b.config_hash
    assert len(a.config_hash) == 16


def test_config_hash_is_stable_across_processes(
    project_root: Path, config: Config
) -> None:
    """Two separate interpreters must agree on the hash.

    Within one process a stable hash proves very little; dict ordering and
    PYTHONHASHSEED only differ across processes. The hash is what an artifact
    records to say which settings produced it, so it has to survive that.
    """
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from src.config import load_config;"
        "print(load_config(sys.argv[1] + '/config.yaml').config_hash)"
    )
    seen = set()
    for _ in range(2):
        out = subprocess.run(
            [sys.executable, "-c", code, str(project_root)],
            capture_output=True,
            text=True,
            check=True,
        )
        seen.add(out.stdout.strip())
    assert seen == {config.config_hash}


def test_config_hash_changes_with_any_value(
    project_root: Path, config: Config
) -> None:
    """An override is a different run and must produce a different hash."""
    other = load_config(
        project_root / "config.yaml", overrides={"validation.bootstrap_samples": 200}
    )
    assert other.config_hash != config.config_hash


def test_config_hash_is_taken_over_the_json_rendering(config: Config) -> None:
    """to_dict() must round-trip through JSON unchanged.

    The hash is taken over exactly this rendering, so a value that survives
    asdict() but not json.dumps() would hash differently on reload.
    """
    assert json.loads(json.dumps(config.to_dict())) == config.to_dict()


# --------------------------------------------------------------------------- #
# Invariants asserted at load time (DECISIONS.md 020)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("probe.positive_class", "correct", "positive_class"),
        ("probe.standardize", False, "standardize"),
        ("sampling.blind_queue", False, "blind_queue"),
        ("sampling.allocation_month_one", "neyman", "proportional"),
        ("policy.fail_closed_on_missing_warrant", False, "fail_closed"),
        ("store.hash_chain", False, "hash_chain"),
        ("store.retention_days", 30, "retention_days"),
        ("drift.psi_significant", 0.05, "psi_significant"),
        ("validation.null_control_band", [0.6, 0.8], "straddle"),
        ("workload.measured_tpr", 0.005, "chance line"),
    ],
)
def test_invariant_violations_are_rejected_at_load(
    raw_config: dict, key: str, value, match: str
) -> None:
    """Each of these produces plausible output if it is allowed through.

    Inverted polarity reads as 1 - AUROC; an unblinded queue biases the
    estimate towards flattery; fail-open policy loading makes the product
    theatre. None of them raises at the point of use, so they are refused here.
    """
    with pytest.raises(ConfigError, match=match):
        _load_mutated(raw_config, key, value)


def test_dropping_a_control_is_rejected(raw_config: dict) -> None:
    """All five controls run on every validation (SPEC.md §2.1)."""
    remaining = [c for c in raw_config["validation"]["controls"] if c != "padding_fault"]
    with pytest.raises(ConfigError, match="padding_fault"):
        _load_mutated(raw_config, "validation.controls", remaining)


def test_dropping_token_length_from_drift_is_rejected(raw_config: dict) -> None:
    """Long context is the documented probe failure mode (SPEC.md §5.1)."""
    remaining = [f for f in raw_config["drift"]["features"] if f != "token_length"]
    with pytest.raises(ConfigError, match="token_length"):
        _load_mutated(raw_config, "drift.features", remaining)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("model.name", "meta-llama/Llama-Guard-3-8B"),
        ("detectors.qwen3guard", "google/shieldgemma-2b"),
    ],
)
def test_non_permissive_licences_are_rejected(raw_config: dict, key: str, value) -> None:
    """The public claim is a fully open self-hostable stack (CLAUDE.md)."""
    with pytest.raises(ConfigError, match="permissive"):
        _load_mutated(raw_config, key, value)


def test_dropping_a_presidio_configuration_is_rejected(raw_config: dict) -> None:
    """Reporting only stock is open to 'you crippled it' (DECISIONS.md 008)."""
    with pytest.raises(ConfigError, match="enabled_plus_custom"):
        _load_mutated(raw_config, "detectors.presidio_configs", ["stock", "enabled"])


def test_duplicate_evalset_ids_are_rejected(raw_config: dict) -> None:
    """The eval set id is part of the warrant key (CLAUDE.md invariant 1)."""
    duplicated = raw_config["evalsets"] + [{"id": "triviaqa-600"}]
    with pytest.raises(ConfigError, match="duplicate"):
        _load_mutated(raw_config, "evalsets", duplicated)


def test_absolute_paths_are_rejected(raw_config: dict) -> None:
    """An absolute path in a committed config will not exist on a clean clone."""
    with pytest.raises(ConfigError, match="relative"):
        _load_mutated(raw_config, "paths.results_dir", str(Path.cwd() / "results"))


# --------------------------------------------------------------------------- #
# Layer resolution
# --------------------------------------------------------------------------- #


def test_layer_resolution_is_half_up_and_clamped(config: Config) -> None:
    """Fractions map to 1-based hidden-state indices, rounded half-up.

    Half-up rather than Python's bankers' rounding so the mapping is the obvious
    one when a reviewer checks it by hand: 0.5 x 28 = 14, not 14-or-13.
    """
    layers = config.resolve_layers(28)
    assert layers == (8, 11, 14, 17, 20, 23, 26)
    assert min(layers) >= 1 and max(layers) <= 28


def test_layer_resolution_collapses_on_a_shallow_model(config: Config) -> None:
    """Seven fractions cannot yield seven distinct layers on a 4-layer model."""
    layers = config.resolve_layers(4)
    assert layers == tuple(sorted(set(layers)))
    assert max(layers) <= 4


def test_layer_resolution_rejects_a_nonpositive_depth(config: Config) -> None:
    with pytest.raises(ConfigError, match="must be positive"):
        config.resolve_layers(0)


# --------------------------------------------------------------------------- #
# Provenance — the Phase 0 gate
# --------------------------------------------------------------------------- #


def test_provenance_records_the_run(config: Config) -> None:
    block = provenance(config)
    assert block["config_hash"] == config.config_hash
    assert block["seed"] == config.seed
    assert block["timestamp_utc"].endswith("+00:00")
    assert len(block["git_commit"]) == 40
    assert "numpy" in block["libraries"]
    assert "platform" in block["device"]


def test_dirty_flag_responds_to_touching_a_file(config: Config) -> None:
    """The gate check: touch a file, the flag flips; remove it, it flips back.

    Asserted as a delta against the tree's current state rather than against a
    clean tree, so the test is meaningful while a phase is in progress.
    """
    baseline = working_tree_changes(config.paths.results_dir)
    assert baseline is not None, "git unavailable; provenance cannot be verified"

    probe = project_root() / ".dirty_probe_test_config"
    probe.write_text("touched by test_dirty_flag_responds_to_touching_a_file\n")
    try:
        changed = working_tree_changes(config.paths.results_dir)
        assert changed is not None
        new = set(changed) - set(baseline)
        assert any(p.endswith(".dirty_probe_test_config") for p in new), new
        assert provenance(config)["dirty"] is True
    finally:
        probe.unlink()

    assert set(working_tree_changes(config.paths.results_dir)) == set(baseline)


def test_dirty_flag_ignores_generated_results(config: Config) -> None:
    """Writing an artifact must not make the next stage report dirty.

    Without this exclusion the first stage dirties the tree by doing its job and
    every later artifact records dirty: true regardless of the code, which
    drains the flag of the only meaning it has.
    """
    baseline = working_tree_changes(config.paths.results_dir)
    assert baseline is not None

    artifact = config.results_path(".dirty_probe_artifact.json")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{}\n")
    try:
        assert set(working_tree_changes(config.paths.results_dir)) == set(baseline)
    finally:
        artifact.unlink()


def test_project_root_and_repo_root_are_distinct() -> None:
    """Round 2 is a subdirectory: git runs at the repo root, paths at the project root."""
    assert (project_root() / "config.yaml").is_file()
    assert (repo_root() / ".git").exists()
    assert project_root().is_relative_to(repo_root())


# --------------------------------------------------------------------------- #
# Artifact I/O
# --------------------------------------------------------------------------- #


def test_artifact_round_trips_with_provenance(config: Config, tmp_path: Path) -> None:
    """No artifact can exist without the run that produced it (invariant 8)."""
    out = write_json_artifact(tmp_path / "sample.json", {"value": 3}, config)
    document = read_json_artifact(out)
    assert document["value"] == 3
    assert document["provenance"]["config_hash"] == config.config_hash


def test_artifact_write_refuses_unserialisable_values(
    config: Config, tmp_path: Path
) -> None:
    """default=str would turn np.float32(0.89) into the string "0.89".

    It still looks like a number in the file and breaks three stages later, so
    an unserialisable value crashes at the point it was produced instead.
    """
    with pytest.raises(TypeError, match="not JSON-serialisable"):
        write_json_artifact(tmp_path / "bad.json", {"value": object()}, config)


def test_numpy_scalars_survive_as_numbers(config: Config, tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    out = write_json_artifact(
        tmp_path / "np.json",
        {"auroc": np.float32(0.8551), "n": np.int64(600), "flag": np.bool_(True)},
        config,
    )
    document = read_json_artifact(out)
    assert isinstance(document["auroc"], float)
    assert isinstance(document["n"], int)
    assert document["flag"] is True


def test_missing_artifact_names_the_ordering(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run the earlier stage first"):
        read_json_artifact(tmp_path / "absent.json")
