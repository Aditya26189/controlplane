"""Configuration loading, validation, layer resolution, hashing, provenance.

Every knob in this experiment lives in ``config.yaml``; nothing in ``src/``
hardcodes an experimental value (CLAUDE.md, "Coding standards"). This module is
the single place that turns that file into typed objects, and it validates
aggressively: an unknown key is a typo that would otherwise be silently
ignored, and a silently ignored knob is a run whose settings are not what the
config file recorded alongside its results says they are.

It also owns two run-level helpers -- ``set_seeds`` and ``setup_logging`` -- so
that every script starts from the same reproducibility posture (SPEC.md §11)
without adding a module outside the layout fixed in CLAUDE.md.
"""

import dataclasses
import hashlib
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, get_args, get_origin

import yaml


class ConfigError(ValueError):
    """Raised when the config is malformed, incomplete, or inconsistent.

    A distinct type so a reader of a stack trace can tell "you configured this
    wrong" apart from "the experiment failed", which are different problems.
    """


# --------------------------------------------------------------------------- #
# Dataclasses mirroring config.yaml, one per top-level section.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelConfig:
    """Model identity, quantisation, and the depths to probe.

    ``layer_fractions`` are fractional depths rather than absolute indices so
    the same config transfers across model sizes (SPEC.md §4). They are
    resolved against ``model.config.num_hidden_layers`` by
    :meth:`Config.resolve_layers` once the model is loaded.
    """

    name: str
    quantization: str
    dtype: str
    device_map: str
    layer_fractions: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.quantization not in {"nf4", "none"}:
            raise ConfigError(
                f"model.quantization must be 'nf4' or 'none', got {self.quantization!r}"
            )
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ConfigError(
                "model.dtype must be one of bfloat16|float16|float32, "
                f"got {self.dtype!r}"
            )
        if not self.layer_fractions:
            raise ConfigError("model.layer_fractions must not be empty")
        for frac in self.layer_fractions:
            if not 0.0 < frac <= 1.0:
                raise ConfigError(
                    f"model.layer_fractions entries must be in (0, 1], got {frac}"
                )


@dataclass(frozen=True)
class DataConfig:
    """Dataset identity, sample size, and split proportions."""

    dataset: str
    config: str
    split: str
    n_examples: int
    train_frac: float
    val_frac: float
    test_frac: float
    dedup_questions: bool

    def __post_init__(self) -> None:
        if self.n_examples <= 0:
            raise ConfigError(f"data.n_examples must be positive, got {self.n_examples}")
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-9:
            raise ConfigError(
                "data.train_frac + val_frac + test_frac must sum to 1.0, "
                f"got {total!r}"
            )
        for name in ("train_frac", "val_frac", "test_frac"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ConfigError(f"data.{name} must be in (0, 1), got {value}")


@dataclass(frozen=True)
class PromptConfig:
    """The fixed experimental condition: system prompt and template mode.

    The system prompt is part of the experimental condition and must not vary
    between runs (SPEC.md §3), which is why it is a config value recorded in
    every artifact rather than a literal inside ``model.py``.
    """

    system: str
    add_generation_prompt: bool

    def __post_init__(self) -> None:
        if not self.add_generation_prompt:
            raise ConfigError(
                "prompt.add_generation_prompt must be true: without the assistant "
                "turn header, the final prompt token is not the position this "
                "experiment probes (SPEC.md §3)"
            )


@dataclass(frozen=True)
class EquivalenceCheckConfig:
    """Thresholds for the left-padding equivalence check (CLAUDE.md invariant 4).

    Scale-invariant on purpose. An absolute tolerance conflates "the batch is
    read at the wrong position", which changes the vector entirely, with "the
    two forward passes rounded differently", which is unavoidable in bfloat16
    and grows with activation magnitude. Relative L2 error and cosine
    similarity separate the two cleanly.
    """

    batch: int
    relative_tolerance: float
    min_cosine: float
    positive_control: bool

    def __post_init__(self) -> None:
        if self.batch < 2:
            raise ConfigError(
                "equivalence_check.batch must be at least 2: a single prompt has "
                "no padding, so the comparison would prove nothing"
            )
        if not 0.0 < self.relative_tolerance < 1.0:
            raise ConfigError(
                "equivalence_check.relative_tolerance must be in (0, 1), got "
                f"{self.relative_tolerance}"
            )
        if not 0.0 < self.min_cosine <= 1.0:
            raise ConfigError(
                f"equivalence_check.min_cosine must be in (0, 1], got {self.min_cosine}"
            )


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding settings for the labelling pass."""

    max_new_tokens: int
    do_sample: bool
    batch_size: int
    sort_by_length: bool

    def __post_init__(self) -> None:
        if self.do_sample:
            raise ConfigError(
                "generation.do_sample must be false: labels come from greedy "
                "decoding so two runs at one seed agree (SPEC.md §11)"
            )
        if self.max_new_tokens <= 0:
            raise ConfigError("generation.max_new_tokens must be positive")
        if self.batch_size <= 0:
            raise ConfigError("generation.batch_size must be positive")


@dataclass(frozen=True)
class LabelingConfig:
    """Alias-matching rule and the sanity band for the resulting base rate."""

    min_alias_len_for_substring: int
    record_strict_em: bool
    base_rate_min: float
    base_rate_max: float

    def __post_init__(self) -> None:
        if self.min_alias_len_for_substring < 1:
            raise ConfigError("labeling.min_alias_len_for_substring must be >= 1")
        if not 0.0 <= self.base_rate_min < self.base_rate_max <= 1.0:
            raise ConfigError(
                "labeling requires 0 <= base_rate_min < base_rate_max <= 1, got "
                f"{self.base_rate_min} and {self.base_rate_max}"
            )


@dataclass(frozen=True)
class ProbeConfig:
    """Probe family, standardisation, regularisation grid, and polarity."""

    type: str
    standardize: bool
    class_weight: str
    max_iter: int
    C_grid: tuple[float, ...]
    positive_class: str

    def __post_init__(self) -> None:
        if self.type != "logistic_regression":
            raise ConfigError(
                "probe.type: only 'logistic_regression' is implemented, got "
                f"{self.type!r}"
            )
        if self.positive_class != "incorrect":
            raise ConfigError(
                "probe.positive_class must be 'incorrect' (DECISIONS.md 004). "
                "Inverting polarity silently yields 1 - AUROC, which reads as a "
                "strong negative result rather than as a bug."
            )
        if not self.C_grid:
            raise ConfigError("probe.C_grid must not be empty")
        for c in self.C_grid:
            if c <= 0:
                raise ConfigError(f"probe.C_grid entries must be positive, got {c}")
        if self.max_iter <= 0:
            raise ConfigError("probe.max_iter must be positive")


@dataclass(frozen=True)
class EconomicsConfig:
    """Budget, judge accuracy, and the illustrative worked-table inputs."""

    target_flag_rate: float
    judge_accuracy: float
    n_responses: int
    reference_error_rate: float

    def __post_init__(self) -> None:
        if not 0.0 < self.target_flag_rate < 1.0:
            raise ConfigError(
                "economics.target_flag_rate must be in (0, 1), got "
                f"{self.target_flag_rate}"
            )
        if not 0.0 < self.judge_accuracy <= 1.0:
            raise ConfigError(
                f"economics.judge_accuracy must be in (0, 1], got {self.judge_accuracy}"
            )
        if self.n_responses <= 0:
            raise ConfigError("economics.n_responses must be positive")
        if not 0.0 <= self.reference_error_rate <= 1.0:
            raise ConfigError(
                "economics.reference_error_rate must be in [0, 1], got "
                f"{self.reference_error_rate}"
            )


@dataclass(frozen=True)
class EvaluationConfig:
    """Bootstrap settings and the floor below which we report rather than tune."""

    bootstrap_samples: int
    ci: float
    min_auroc_to_proceed: float

    def __post_init__(self) -> None:
        if self.bootstrap_samples <= 0:
            raise ConfigError("evaluation.bootstrap_samples must be positive")
        if not 0.0 < self.ci < 1.0:
            raise ConfigError(f"evaluation.ci must be in (0, 1), got {self.ci}")
        if not 0.0 <= self.min_auroc_to_proceed <= 1.0:
            raise ConfigError("evaluation.min_auroc_to_proceed must be in [0, 1]")


@dataclass(frozen=True)
class LatencyConfig:
    """How many times to repeat the probe timing loop."""

    probe_timing_repeats: int

    def __post_init__(self) -> None:
        if self.probe_timing_repeats <= 0:
            raise ConfigError("latency.probe_timing_repeats must be positive")


@dataclass(frozen=True)
class AbstentionConfig:
    """Patterns marking an abstaining generation, and the power floor."""

    patterns: tuple[str, ...]
    min_rate_to_report: float

    def __post_init__(self) -> None:
        if not self.patterns:
            raise ConfigError("abstention.patterns must not be empty")
        if not 0.0 <= self.min_rate_to_report <= 1.0:
            raise ConfigError("abstention.min_rate_to_report must be in [0, 1]")


@dataclass(frozen=True)
class NegativeControlConfig:
    """Optional Stage 6 GSM8K control (SPEC.md §10)."""

    enabled: bool
    dataset: str
    config: str
    n_examples: int

    def __post_init__(self) -> None:
        if self.n_examples <= 0:
            raise ConfigError("negative_control.n_examples must be positive")


@dataclass(frozen=True)
class PathsConfig:
    """Output locations. Every stage writes here and reads from here."""

    results_dir: str
    activations: str
    labels: str
    splits: str


@dataclass(frozen=True)
class Config:
    """The fully validated experiment configuration.

    Frozen so no stage can mutate a setting after the hash recorded in an
    artifact was computed. That hash is the claim that a number came from these
    settings; a mutable config would make the claim unfalsifiable.
    """

    seed: int
    model: ModelConfig
    data: DataConfig
    prompt: PromptConfig
    equivalence_check: EquivalenceCheckConfig
    generation: GenerationConfig
    labeling: LabelingConfig
    probe: ProbeConfig
    economics: EconomicsConfig
    evaluation: EvaluationConfig
    latency: LatencyConfig
    abstention: AbstentionConfig
    negative_control: NegativeControlConfig
    paths: PathsConfig

    # -- serialisation ------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Return the config as plain JSON-serialisable data.

        Tuples become lists so a round-trip through JSON is stable, which
        matters because the config hash is taken over exactly this rendering.
        """
        return _jsonify(dataclasses.asdict(self))

    @property
    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON rendering, truncated to 16 hex chars.

        Recomputed on access rather than stored: the object is frozen so the
        value cannot drift, and a stored field would have to be excluded from
        its own hash.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    # -- layer resolution ---------------------------------------------------- #

    def resolve_layers(self, num_hidden_layers: int) -> tuple[int, ...]:
        """Turn fractional depths into absolute hidden-state indices.

        Indices are 1-based against ``outputs.hidden_states``, where index 0 is
        the embedding output and index L is the output of transformer block L
        (SPEC.md §4). Rounding is half-up rather than Python's bankers' rounding
        so the mapping is the obvious one when a reviewer checks it by hand.

        Args:
            num_hidden_layers: ``model.config.num_hidden_layers``.

        Returns:
            Sorted, deduplicated absolute layer indices in
            ``[1, num_hidden_layers]``.
        """
        if num_hidden_layers <= 0:
            raise ConfigError(
                f"num_hidden_layers must be positive, got {num_hidden_layers}"
            )
        resolved: list[int] = []
        for frac in self.model.layer_fractions:
            idx = int(math.floor(frac * num_hidden_layers + 0.5))
            resolved.append(max(1, min(num_hidden_layers, idx)))
        unique = tuple(sorted(set(resolved)))
        if len(unique) != len(resolved):
            logging.getLogger(__name__).warning(
                "layer_fractions %s collapsed to %d distinct layers on a "
                "%d-layer model: %s",
                list(self.model.layer_fractions),
                len(unique),
                num_hidden_layers,
                list(unique),
            )
        return unique

    # -- paths --------------------------------------------------------------- #

    def results_path(self, name: str) -> Path:
        """Resolve a filename inside the configured results directory."""
        return Path(self.paths.results_dir) / name


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _jsonify(value: Any) -> Any:
    """Recursively convert tuples to lists for a stable JSON rendering."""
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _coerce(value: Any, typ: Any, path: str) -> Any:
    """Coerce and type-check one config value, naming its dotted path on error."""
    if is_dataclass(typ):
        return _from_mapping(typ, value, path)
    origin = get_origin(typ)
    if origin is tuple:
        item_type = get_args(typ)[0]
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        return tuple(_coerce(v, item_type, f"{path}[{i}]") for i, v in enumerate(value))
    if typ is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a boolean, got {value!r}")
        return value
    if typ is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if typ is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if typ is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{path}: unsupported config field type {typ!r}")


def _from_mapping(cls: type, data: Any, path: str = "") -> Any:
    """Build a config dataclass from a mapping, rejecting unknown/missing keys.

    Unknown keys are an error rather than a warning: a mistyped knob that is
    silently dropped produces a run whose settings differ from the config file
    recorded next to its results.
    """
    where = path or "<root>"
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    field_map = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(field_map))
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {unknown}; known keys are {sorted(field_map)}"
        )
    missing = sorted(set(field_map) - set(data))
    if missing:
        raise ConfigError(f"{where}: missing required key(s) {missing}")
    kwargs = {
        name: _coerce(data[name], f.type, f"{path}.{name}" if path else name)
        for name, f in field_map.items()
    }
    return cls(**kwargs)


def _apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply dotted-key overrides (e.g. ``data.n_examples``) to raw config data.

    Overrides exist for ``--smoke`` and for the negative control, which need to
    change one or two values without a second config file drifting out of sync
    with the first. Overriding a key that does not exist is an error, and any
    override changes the config hash -- correctly, because it is a different run.
    """
    out = json.loads(json.dumps(raw))  # deep copy of plain YAML data
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        cursor: Any = out
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise ConfigError(f"override {dotted!r}: no such section {part!r}")
            cursor = cursor[part]
        leaf = parts[-1]
        if not isinstance(cursor, dict) or leaf not in cursor:
            raise ConfigError(f"override {dotted!r}: no such key")
        cursor[leaf] = value
    return out


def load_config(
    path: str | os.PathLike = "config.yaml",
    overrides: Optional[dict[str, Any]] = None,
) -> Config:
    """Load, override, and validate the experiment config.

    Args:
        path: Path to the YAML config.
        overrides: Optional dotted-key overrides applied before validation, so
            an override naming a nonexistent key fails loudly.

    Returns:
        A frozen, validated :class:`Config`.

    Raises:
        ConfigError: on a missing file, malformed YAML, unknown or missing keys,
            or a value violating a documented constraint.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raise ConfigError(f"config file is empty: {p}")
    if overrides:
        raw = _apply_overrides(raw, overrides)
    return _from_mapping(Config, raw)


# --------------------------------------------------------------------------- #
# Reproducibility and provenance
# --------------------------------------------------------------------------- #


def set_seeds(seed: int) -> None:
    """Seed ``random``, ``numpy``, and torch (CPU and CUDA) from one value.

    Called at the top of every script. Two runs at one seed must produce
    identical numbers (CLAUDE.md, "Definition of done"); this plus greedy
    decoding is the whole reproducibility story.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch absent only in doc builds
        pass


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured, timestamped logging to stdout."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def repo_root() -> Path:
    """Return the repository root, located by walking up to the ``.git`` entry."""
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / ".git").exists():
            return candidate
    return here.parent.parent


def _git(*args: str) -> Optional[str]:
    """Run a git command in the repo root, returning None if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def library_versions() -> dict[str, Optional[str]]:
    """Collect versions of libraries whose behaviour could move a measured number.

    Reads distribution metadata rather than importing each package. Importing
    bitsandbytes alone costs several seconds, and provenance() runs on every
    artifact write -- so importing here would add minutes to a pipeline run for
    a string that is already on disk.
    """
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, Optional[str]] = {}
    for distribution in [
        "torch",
        "transformers",
        "datasets",
        "scikit-learn",
        "numpy",
        "pandas",
        "scipy",
        "bitsandbytes",
    ]:
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = None

    # If torch is already imported, prefer its own string: it carries the build
    # tag ("+cpu", "+cu121") that distribution metadata drops, and that tag is
    # part of what makes a latency number reproducible.
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        versions["torch"] = getattr(torch_module, "__version__", versions["torch"])
    return versions


def device_info() -> dict[str, Any]:
    """Describe the compute device: latency numbers are meaningless without it."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cuda_available": False,
        "device_name": "cpu",
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda
    except ImportError:  # pragma: no cover
        pass
    return info


def provenance(config: Optional[Config] = None) -> dict[str, Any]:
    """Build the provenance block embedded in every artifact.

    Records ``git_commit`` as HEAD at the moment the script ran -- the commit of
    the *code* that produced the artifact, never the commit that contains it
    (CONTRIBUTING.md, "The provenance ordering problem").

    ``dirty`` comes from ``git status --porcelain``. An artifact generated from a
    dirty tree records a commit hash that does not describe the code that ran, so
    the flag has to travel with the hash rather than be checked once by hand.

    Args:
        config: If given, its hash, seed, and resolved values are embedded too.

    Returns:
        A JSON-serialisable provenance mapping.
    """
    status = _git("status", "--porcelain")
    block: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status.strip()),
        "python": sys.version.split()[0],
        "libraries": library_versions(),
        "device": device_info(),
    }
    if config is not None:
        block["config_hash"] = config.config_hash
        block["seed"] = config.seed
        block["config"] = config.to_dict()
    return block


def _json_default(value: Any) -> Any:
    """Serialise numpy scalars as numbers, and refuse anything else.

    ``default=str`` would be simpler and would silently turn a stray
    ``np.float32`` into the *string* ``"0.89"`` -- json only handles numpy types
    that subclass Python builtins, and ``float32``, ``int64`` and ``bool_`` do
    not. A number that becomes a string still looks like a number in the file,
    then breaks or renders wrong three stages later. Everything else raises, so
    an unserialisable value is a crash at the point it was produced.
    """
    import numpy as np

    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"{type(value).__name__} is not JSON-serialisable. Cast it explicitly at "
        "the point it is produced rather than letting it be stringified."
    )


def write_json_artifact(
    path: str | os.PathLike, payload: dict[str, Any], config: Optional[Config] = None
) -> Path:
    """Write a results artifact with its provenance block attached.

    Every stage writes through here so that no artifact can exist without the
    config hash, seed, git commit and library versions that produced it
    (CLAUDE.md invariant 7 -- a number in the README must be traceable to a
    script, a seed and a config hash).

    Args:
        path: Destination JSON path.
        payload: The stage's own results.
        config: Config to stamp into the provenance block.

    Returns:
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    document = {"provenance": provenance(config), **payload}
    with out.open("w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2, sort_keys=False, default=str)
        fh.write("\n")
    logging.getLogger(__name__).info("wrote %s", out)
    return out


def read_json_artifact(path: str | os.PathLike) -> dict[str, Any]:
    """Read an artifact written by :func:`write_json_artifact`.

    Raises:
        FileNotFoundError: with the stage that produces the file named, since
            the usual cause is running stage N before stage N-1.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} not found. Stages read each other's output from disk; run the "
            "earlier stage first (see scripts/run_all.py for the order)."
        )
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)
