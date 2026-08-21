"""Config loading, validation, hashing, layer resolution, and provenance."""

import json
import subprocess

import pytest
import yaml

from src.config import ConfigError, load_config, provenance


@pytest.fixture
def raw_config(repo_root):
    """The committed config as plain YAML data."""
    with (repo_root / "config.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_config(tmp_path, data):
    """Serialise config data to a temp file and return its path."""
    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)
    return path


def mutated(raw, section, **changes):
    """Deep-copy raw config data and apply changes to one section."""
    out = json.loads(json.dumps(raw))
    out[section].update(changes)
    return out


def test_config_hash_is_stable(repo_root):
    """Two loads of the same file must hash identically.

    The hash is what ties a number in RESULTS.md to the settings that produced
    it; if it moved between loads it would prove nothing.
    """
    first = load_config(repo_root / "config.yaml")
    second = load_config(repo_root / "config.yaml")
    assert first.config_hash == second.config_hash
    assert len(first.config_hash) == 16


def test_override_changes_hash(repo_root):
    """A smoke run is a different run and must not share the full run's hash."""
    full = load_config(repo_root / "config.yaml")
    smoke = load_config(repo_root / "config.yaml", {"data.n_examples": 100})
    assert smoke.data.n_examples == 100
    assert smoke.config_hash != full.config_hash


def test_unknown_key_raises(tmp_path, raw_config):
    """A mistyped knob must fail loudly, not be silently ignored."""
    path = write_config(tmp_path, mutated(raw_config, "model", layer_fracions=[0.5]))
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_missing_key_raises(tmp_path, raw_config):
    """A dropped key must fail rather than fall back to a hidden default."""
    data = json.loads(json.dumps(raw_config))
    del data["data"]["n_examples"]
    with pytest.raises(ConfigError, match="missing required key"):
        load_config(write_config(tmp_path, data))


def test_unknown_override_raises(repo_root):
    """An override naming a nonexistent key is a typo, not a new setting."""
    with pytest.raises(ConfigError, match="no such key"):
        load_config(repo_root / "config.yaml", {"data.n_exampels": 100})


def test_polarity_guard(tmp_path, raw_config):
    """probe.positive_class must stay 'incorrect' (DECISIONS.md 004).

    Inverting it silently produces 1 - AUROC, which reads as a strong negative
    result rather than as a bug, so the config refuses the flip outright.
    """
    path = write_config(tmp_path, mutated(raw_config, "probe", positive_class="correct"))
    with pytest.raises(ConfigError, match="positive_class"):
        load_config(path)


def test_sampling_guard(tmp_path, raw_config):
    """Labels must come from greedy decoding (SPEC.md §11)."""
    path = write_config(tmp_path, mutated(raw_config, "generation", do_sample=True))
    with pytest.raises(ConfigError, match="do_sample"):
        load_config(path)


def test_split_fractions_must_sum_to_one(tmp_path, raw_config):
    """A split that does not partition the data would silently drop examples."""
    path = write_config(tmp_path, mutated(raw_config, "data", train_frac=0.7))
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(path)


def test_wrong_type_raises(tmp_path, raw_config):
    """Types are checked so a quoted number cannot reach torch as a string."""
    path = write_config(tmp_path, mutated(raw_config, "generation", batch_size="eight"))
    with pytest.raises(ConfigError, match="expected an integer"):
        load_config(path)


def test_layer_resolution_matches_spec_default(config):
    """Fractional depths must resolve to SPEC.md §4's stated 28-layer default."""
    assert config.resolve_layers(28) == (8, 11, 14, 17, 20, 23, 26)


@pytest.mark.parametrize("n_layers", [4, 12, 24, 28, 32, 80])
def test_layer_resolution_in_range(config, n_layers):
    """Resolved indices must be valid hidden_states indices for any model size."""
    layers = config.resolve_layers(n_layers)
    assert layers == tuple(sorted(set(layers)))
    assert all(1 <= layer <= n_layers for layer in layers)


def test_provenance_reports_commit_and_dirty(repo_root, config):
    """Provenance must carry a real commit and an honest dirty flag.

    An artifact that records a commit hash while the tree was dirty is claiming
    a lineage it does not have (CONTRIBUTING.md, provenance ordering).
    """
    block = provenance(config)
    assert block["config_hash"] == config.config_hash
    assert block["seed"] == config.seed
    assert set(block) >= {
        "timestamp_utc",
        "git_commit",
        "dirty",
        "python",
        "libraries",
        "device",
    }
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode == 0:
        assert block["git_commit"] is not None
        assert len(block["git_commit"]) == 40
        assert block["dirty"] == bool(status.stdout.strip())


def test_provenance_is_json_serialisable(config):
    """Provenance is embedded in JSON artifacts, so it must survive json.dumps."""
    json.dumps(provenance(config))
