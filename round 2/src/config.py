"""Configuration loading, validation, hashing and provenance.

Everything tunable in this project lives in ``config.yaml`` and reaches code
through this module. Two invariants from ``CLAUDE.md`` are enforced structurally
here rather than by discipline:

* **Invariant 6 — one declared workload.** :class:`WorkloadConfig` is a single
  frozen block and :mod:`src.economics.sizing` takes it as its only source.
  There is no second place to read a flag rate or a base rate from, so two
  scenarios cannot be mixed into one table.
* **Invariant 8 — every number in a document is computed by code.** The config
  hash defined here is half of what makes a number traceable to the run that
  produced it.

The loader is strict on purpose. An unknown key is an error, not a warning: a
mistyped knob that is silently dropped produces a run whose settings differ from
the config recorded beside its results, and nothing raises.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

import yaml

__all__ = [
    "Config",
    "ConfigError",
    "load_config",
]


class ConfigError(ValueError):
    """Raised when the config is missing, malformed, or self-inconsistent.

    A distinct type so callers can tell "your config is wrong" from "the code is
    wrong", which are fixed in different places by different people.
    """


# Detectors whose licences are not OSI-permissive. The project's public claim is
# a fully open, self-hostable stack, so a runtime dependency on either would make
# that claim false (CLAUDE.md, "Out of scope"). Checked against every configured
# model string at load time, so the violation surfaces as a crash on the first
# run rather than as a licence question during judging.
_LICENCE_DENYLIST = (
    "llama-guard",
    "llamaguard",
    "llama_guard",
    "shieldgemma",
    "shield-gemma",
)


def _reject_non_permissive(value: str, path: str) -> None:
    """Fail loudly if a configured model is under a non-permissive licence."""
    lowered = value.lower()
    for banned in _LICENCE_DENYLIST:
        if banned in lowered:
            raise ConfigError(
                f"{path}: {value!r} is not OSI-permissively licensed. The "
                "project's public claim is a fully open self-hostable stack; "
                "see CLAUDE.md, 'Out of scope'."
            )


def _project_root() -> Path:
    """The project root — the directory holding ``config.yaml``."""
    return Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class WorkloadConfig:
    """The single declared workload every economic figure derives from.

    ``CLAUDE.md`` invariant 6. This block exists once and
    :mod:`src.economics.sizing` reads nothing else, so a flag rate from one
    scenario cannot end up in a table beside a base rate from another. That
    failure is silent — the numbers look fine and do not survive a reviewer with
    a calculator.

    ``measured_tpr`` and ``measured_fpr`` are the Round 1 measured operating
    point carried forward. They are the only transcribed measurements in the
    project, and they are *inputs*: every derived quantity (flag rate, TP,
    unflagged pool size, prevalence, dR/dq) is computed from them by code.
    """

    name: str
    monthly_interactions: int
    base_error_rate: float
    review_minutes_per_item: float
    reviewer_monthly_cost_inr: float
    reviewer_productive_hours: float
    measured_tpr: float
    measured_fpr: float

    def __post_init__(self) -> None:
        if self.monthly_interactions <= 0:
            raise ConfigError(
                "workload.monthly_interactions must be positive, got "
                f"{self.monthly_interactions}"
            )
        for name in ("base_error_rate", "measured_tpr", "measured_fpr"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ConfigError(f"workload.{name} must be in (0, 1), got {value}")
        for name in (
            "review_minutes_per_item",
            "reviewer_monthly_cost_inr",
            "reviewer_productive_hours",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ConfigError(f"workload.{name} must be positive, got {value}")
        # A detector whose TPR does not exceed its FPR sits below the chance
        # line, and every economic figure derived from it would be a projection
        # of noise dressed as a measurement.
        if self.measured_tpr <= self.measured_fpr:
            raise ConfigError(
                f"workload.measured_tpr ({self.measured_tpr}) must exceed "
                f"measured_fpr ({self.measured_fpr}); otherwise the operating "
                "point is below the chance line and the economics are meaningless."
            )


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """The model the activation-tier probe is pinned to.

    Pinned, not chosen at runtime: a probe reads one specific model's residual
    stream, so a model change invalidates every activation-tier warrant until
    revalidation (``SPEC.md`` §5.4). Recording the name here is what lets the
    store detect that.
    """

    name: str
    quantization: str
    layer_fractions: tuple[float, ...]

    def __post_init__(self) -> None:
        _reject_non_permissive(self.name, "model.name")
        if self.quantization not in {"nf4", "int8", "fp16", "bf16", "none"}:
            raise ConfigError(
                "model.quantization must be one of nf4/int8/fp16/bf16/none, got "
                f"{self.quantization!r}"
            )
        if not self.layer_fractions:
            raise ConfigError("model.layer_fractions must not be empty")
        for frac in self.layer_fractions:
            if not 0.0 < frac <= 1.0:
                raise ConfigError(
                    f"model.layer_fractions entries must be in (0, 1], got {frac}"
                )


@dataclasses.dataclass(frozen=True)
class ProbeConfig:
    """Probe training knobs, including the polarity that must never invert.

    ``positive_class`` is pinned to ``"incorrect"``. Inverting it yields
    ``1 - AUROC``, which reads as a strong negative result and misdirects
    debugging for hours (``CLAUDE.md``, "Silent failures"). ``test_polarity``
    asserts the runtime behaviour; this assertion catches the config edit that
    would cause it.
    """

    aggregations: tuple[str, ...]
    standardize: bool
    class_weight: str
    C_grid: tuple[float, ...]
    positive_class: str

    def __post_init__(self) -> None:
        if self.positive_class != "incorrect":
            raise ConfigError(
                "probe.positive_class must be 'incorrect', got "
                f"{self.positive_class!r}. Inverting polarity produces 1 - AUROC, "
                "which reads as a strong negative result rather than as a bug."
            )
        known = {"mean_pool", "max_rolling_means", "last_token"}
        unknown = sorted(set(self.aggregations) - known)
        if unknown:
            raise ConfigError(
                f"probe.aggregations contains unknown strategies {unknown}; known "
                f"are {sorted(known)}"
            )
        if not self.aggregations:
            raise ConfigError("probe.aggregations must not be empty")
        if not self.C_grid:
            raise ConfigError("probe.C_grid must not be empty")
        if any(c <= 0 for c in self.C_grid):
            raise ConfigError(
                f"probe.C_grid entries must be positive, got {list(self.C_grid)}"
            )
        if self.class_weight not in {"balanced", "none"}:
            raise ConfigError(
                f"probe.class_weight must be 'balanced' or 'none', got "
                f"{self.class_weight!r}"
            )
        if not self.standardize:
            raise ConfigError(
                "probe.standardize must be true. Residual-stream vectors have "
                "large, layer-varying magnitudes; unstandardised features make "
                "the regularisation path meaningless (CLAUDE.md, 'Silent failures')."
            )


@dataclasses.dataclass(frozen=True)
class EvalSetSpec:
    """One evaluation set's declaration.

    Only identity and construction parameters live here. The content hash is
    computed from the built set (``SPEC.md`` §4) and stored in the registry, never
    declared in config — a hash you can type is a hash that can be wrong.
    """

    id: str
    pad_tokens: Optional[tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ConfigError("evalsets[].id must not be empty")
        if self.pad_tokens is not None:
            if len(self.pad_tokens) != 2:
                raise ConfigError(
                    f"evalsets[{self.id}].pad_tokens must be a [min, max] pair, got "
                    f"{list(self.pad_tokens)}"
                )
            low, high = self.pad_tokens
            if not 0 < low < high:
                raise ConfigError(
                    f"evalsets[{self.id}].pad_tokens must satisfy 0 < min < max, "
                    f"got {list(self.pad_tokens)}"
                )


@dataclasses.dataclass(frozen=True)
class ValidationConfig:
    """The thresholds a warrant is issued or refused against.

    These are the refusal criteria of ``SPEC.md`` §2.3 in data form. They are read
    by the issuance path and by nothing else; there is no second copy of a
    threshold anywhere, which is half of why no override path can exist.
    """

    bootstrap_samples: int
    ci: float
    min_n_test: int
    min_auroc_lower_ci: float
    warrant_ttl_hours: int
    controls: tuple[str, ...]
    null_control_band: tuple[float, ...]

    #: The five controls of SPEC.md §2.1. All five run on every validation and
    #: any failure refuses the warrant, so the set is fixed rather than a subset
    #: a config edit could quietly shrink.
    REQUIRED_CONTROLS = (
        "padding_fault",
        "label_shuffle",
        "null_feature",
        "canary",
        "determinism",
    )

    def __post_init__(self) -> None:
        if self.bootstrap_samples < 100:
            raise ConfigError(
                "validation.bootstrap_samples must be >= 100 for a usable "
                f"percentile interval, got {self.bootstrap_samples}"
            )
        if not 0.0 < self.ci < 1.0:
            raise ConfigError(f"validation.ci must be in (0, 1), got {self.ci}")
        if self.min_n_test < 1:
            raise ConfigError(
                f"validation.min_n_test must be positive, got {self.min_n_test}"
            )
        if not 0.0 <= self.min_auroc_lower_ci <= 1.0:
            raise ConfigError(
                "validation.min_auroc_lower_ci must be in [0, 1], got "
                f"{self.min_auroc_lower_ci}"
            )
        if self.warrant_ttl_hours <= 0:
            raise ConfigError(
                "validation.warrant_ttl_hours must be positive, got "
                f"{self.warrant_ttl_hours}"
            )
        missing = sorted(set(self.REQUIRED_CONTROLS) - set(self.controls))
        if missing:
            raise ConfigError(
                f"validation.controls is missing {missing}. All five controls of "
                "SPEC.md §2.1 run on every validation; dropping one would let a "
                "warrant issue without the check that exists to refuse it."
            )
        if len(self.null_control_band) != 2:
            raise ConfigError(
                "validation.null_control_band must be a [low, high] pair, got "
                f"{list(self.null_control_band)}"
            )
        low, high = self.null_control_band
        if not 0.0 < low < 0.5 < high < 1.0:
            raise ConfigError(
                "validation.null_control_band must straddle 0.5, got "
                f"{list(self.null_control_band)}. The negative controls assert the "
                "pipeline can produce a null result when there is no signal; a "
                "band excluding chance cannot express that."
            )


@dataclasses.dataclass(frozen=True)
class DriftConfig:
    """Envelope monitoring and the revocation ladder's thresholds.

    The PSI bands follow the convention used by Indian banking risk teams
    (``SPEC.md`` §5.2). That is deliberate: it is native vocabulary to the
    audience, not a threshold we invented.
    """

    window_size: int
    psi_stable: float
    psi_significant: float
    mmd_permutations: int
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ConfigError(f"drift.window_size must be >= 2, got {self.window_size}")
        if not 0.0 < self.psi_stable < self.psi_significant:
            raise ConfigError(
                f"drift thresholds must satisfy 0 < psi_stable ({self.psi_stable}) "
                f"< psi_significant ({self.psi_significant}); STALE is the band "
                "between them, and an inverted pair erases that state entirely."
            )
        if self.mmd_permutations < 100:
            raise ConfigError(
                "drift.mmd_permutations must be >= 100 for a usable permutation "
                f"p-value, got {self.mmd_permutations}"
            )
        if not self.features:
            raise ConfigError("drift.features must not be empty")
        if "token_length" not in self.features:
            raise ConfigError(
                "drift.features must include 'token_length'. Long context is the "
                "documented probe failure mode (SPEC.md §5.1) and it is the "
                "feature the drift demo turns on."
            )


@dataclasses.dataclass(frozen=True)
class SamplingConfig:
    """Stratified estimation, allocation and label-quality settings.

    ``allocation_month_one`` is proportional because Neyman allocation needs
    per-band prevalence ``q_h`` that does not exist before the first month of
    labels (``SPEC.md`` §6.3). ``expected_design_effect`` is a prior to compare
    the measured value against, never a substitute for measuring it.
    """

    allocation_month_one: str
    allocation_thereafter: str
    expected_design_effect: float
    score_bands: int
    double_label_fraction: float
    recall_margin_tiers: tuple[float, ...]
    blind_queue: bool

    def __post_init__(self) -> None:
        allowed = {"proportional", "neyman"}
        for name in ("allocation_month_one", "allocation_thereafter"):
            value = getattr(self, name)
            if value not in allowed:
                raise ConfigError(
                    f"sampling.{name} must be one of {sorted(allowed)}, got {value!r}"
                )
        if self.allocation_month_one != "proportional":
            raise ConfigError(
                "sampling.allocation_month_one must be 'proportional'. Neyman "
                "allocation needs per-band prevalence q_h that does not exist "
                "until the first month of labels is in (SPEC.md §6.3)."
            )
        if not 0.0 < self.expected_design_effect <= 1.0:
            raise ConfigError(
                "sampling.expected_design_effect must be in (0, 1], got "
                f"{self.expected_design_effect}. It is a variance ratio: 1.0 is no "
                "gain, and a value above 1 would claim Neyman is worse than SRS."
            )
        if self.score_bands < 2:
            raise ConfigError(
                "sampling.score_bands must be >= 2 for stratification to mean "
                f"anything, got {self.score_bands}"
            )
        if not 0.0 <= self.double_label_fraction <= 1.0:
            raise ConfigError(
                "sampling.double_label_fraction must be in [0, 1], got "
                f"{self.double_label_fraction}"
            )
        if not self.recall_margin_tiers:
            raise ConfigError("sampling.recall_margin_tiers must not be empty")
        for margin in self.recall_margin_tiers:
            if not 0.0 < margin < 1.0:
                raise ConfigError(
                    "sampling.recall_margin_tiers entries must be in (0, 1), got "
                    f"{margin}"
                )
        if not self.blind_queue:
            raise ConfigError(
                "sampling.blind_queue must be true. An unblinded queue gives the "
                "two strata systematically different label distributions and "
                "biases the estimate in the direction that flatters us "
                "(SPEC.md §6.5, DECISIONS.md 016)."
            )


@dataclasses.dataclass(frozen=True)
class WeightedErrorConfig:
    """Weights for the threshold objective of ``SPEC.md`` §7.4.

    A declared policy tradeoff, not a tuned result. They live in the policy
    bundle, are versioned with it, and appear on screen during the demo — which
    is the point: the tradeoff is declared rather than solved.
    """

    w_fpr_benign: float
    w_fnr: float
    w_fpr_hard_negative: float

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if value <= 0:
                raise ConfigError(
                    f"policy.weighted_error.{f.name} must be positive, got {value}"
                )


@dataclasses.dataclass(frozen=True)
class PolicyConfig:
    """Policy engine selection and the load-time failure mode.

    ``fail_closed_on_missing_warrant`` is pinned true. Warning and continuing is
    the ordinary engineering choice and it silently reintroduces exactly the
    unbacked-claim problem the project exists to solve (``DECISIONS.md`` 012).
    """

    engine: str
    fail_closed_on_missing_warrant: bool
    weighted_error: WeightedErrorConfig

    def __post_init__(self) -> None:
        if self.engine not in {"opa", "cedar"}:
            raise ConfigError(
                f"policy.engine must be 'opa' or 'cedar', got {self.engine!r}. Do "
                "not write a DSL (SPEC.md §7.1)."
            )
        if not self.fail_closed_on_missing_warrant:
            raise ConfigError(
                "policy.fail_closed_on_missing_warrant must be true. A bundle "
                "referencing an unwarranted operating point fails to load; "
                "warning and continuing is the failure this project argues "
                "against (DECISIONS.md 012)."
            )


@dataclasses.dataclass(frozen=True)
class ProfileConfig:
    """One deployment profile: a latency budget and the bounds it demands.

    The three profiles sit at three points on **one measured curve**, not at
    three invented thresholds (``SPEC.md`` §7.3). ``min_recall`` is what a
    warrant must clear for this profile to run, so when drift widens the bounds
    below it the profile suspends itself. That is Beat 4, step 5.
    """

    inline_budget_ms: int
    min_recall: float
    max_fpr: float

    def __post_init__(self) -> None:
        if self.inline_budget_ms <= 0:
            raise ConfigError(
                f"profile.inline_budget_ms must be positive, got "
                f"{self.inline_budget_ms}"
            )
        if not 0.0 < self.min_recall <= 1.0:
            raise ConfigError(
                f"profile.min_recall must be in (0, 1], got {self.min_recall}"
            )
        if not 0.0 < self.max_fpr <= 1.0:
            raise ConfigError(f"profile.max_fpr must be in (0, 1], got {self.max_fpr}")


@dataclasses.dataclass(frozen=True)
class DetectorsConfig:
    """Third-party detectors wrapped by adapters.

    Every model string is checked against the licence denylist at load time.
    Presidio's three configurations are named here because all three are measured
    and all three are reported (``DECISIONS.md`` 008) — reporting only the
    weakest is the tilt a reviewer would rightly attack.
    """

    presidio_configs: tuple[str, ...]
    qwen3guard: str

    def __post_init__(self) -> None:
        _reject_non_permissive(self.qwen3guard, "detectors.qwen3guard")
        expected = ("stock", "enabled", "enabled_plus_custom")
        missing = sorted(set(expected) - set(self.presidio_configs))
        if missing:
            raise ConfigError(
                f"detectors.presidio_configs is missing {missing}. All three "
                "configurations are measured and reported; showing only the stock "
                "result is open to 'you crippled it' (DECISIONS.md 008)."
            )


@dataclasses.dataclass(frozen=True)
class StoreConfig:
    """Audit store location, retention and tamper-evidence.

    ``retention_days`` sits above the DPDP Rule 6 one-year minimum
    (``SPEC.md`` §1.5). ``hash_chain`` is pinned true: without the chain the
    store is a log, not evidence.
    """

    path: str
    retention_days: int
    hash_chain: bool

    def __post_init__(self) -> None:
        if self.retention_days < 365:
            raise ConfigError(
                "store.retention_days must be >= 365; DPDP Rule 6 requires at "
                f"least one year of retention, got {self.retention_days}"
            )
        if not self.hash_chain:
            raise ConfigError(
                "store.hash_chain must be true. Without the chain the store is a "
                "log rather than evidence, and tamper-evidence is a stated "
                "property of the certificate (SPEC.md §1.5)."
            )


@dataclasses.dataclass(frozen=True)
class PathsConfig:
    """Where artifacts are read and written, relative to the project root."""

    results_dir: str
    evalsets_dir: str
    policies_dir: str

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if Path(value).is_absolute():
                raise ConfigError(
                    f"paths.{f.name} must be relative to the project root, got "
                    f"{value!r}. An absolute path in a committed config is a local "
                    "path that will not exist on a judge's clean clone."
                )


@dataclasses.dataclass(frozen=True)
class Config:
    """The whole resolved configuration, frozen.

    Frozen because the config hash is taken over its rendering: a mutable config
    could be changed after the hash was recorded, and the artifact's provenance
    block would then describe settings the run did not use.
    """

    seed: int
    workload: WorkloadConfig
    model: ModelConfig
    probe: ProbeConfig
    evalsets: tuple[EvalSetSpec, ...]
    validation: ValidationConfig
    drift: DriftConfig
    sampling: SamplingConfig
    policy: PolicyConfig
    profiles: dict[str, ProfileConfig]
    detectors: DetectorsConfig
    store: StoreConfig
    paths: PathsConfig

    def __post_init__(self) -> None:
        ids = [spec.id for spec in self.evalsets]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ConfigError(
                f"evalsets contains duplicate id(s) {duplicates}. The eval set id "
                "is part of the warrant key (CLAUDE.md invariant 1); duplicates "
                "would make two different sets indistinguishable in the matrix."
            )
        if not self.profiles:
            raise ConfigError("profiles must not be empty")

    # -- serialisation ------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Return the config as plain JSON-serialisable data.

        Tuples become lists so a round-trip through JSON is stable, which matters
        because the config hash is taken over exactly this rendering.
        """
        return _jsonify(dataclasses.asdict(self))

    @property
    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON rendering, truncated to 16 hex chars.

        Recomputed on access rather than stored: the object is frozen so the
        value cannot drift, and a stored field would have to be excluded from its
        own hash.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    # -- layer resolution --------------------------------------------------- #

    def resolve_layers(self, num_hidden_layers: int) -> tuple[int, ...]:
        """Turn fractional depths into absolute hidden-state indices.

        Indices are 1-based against ``outputs.hidden_states``, where index 0 is
        the embedding output and index L is the output of transformer block L.
        Rounding is half-up rather than Python's bankers' rounding, so the
        mapping is the obvious one when a reviewer checks it by hand.

        Fractions rather than absolute indices because the probe has to be
        describable the same way across models of different depth: the thing that
        matters is a depth, not an index.

        Args:
            num_hidden_layers: ``model.config.num_hidden_layers``.

        Returns:
            Sorted, deduplicated absolute layer indices in
            ``[1, num_hidden_layers]``.

        Raises:
            ConfigError: If ``num_hidden_layers`` is not positive.
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
                "layer_fractions %s collapsed to %d distinct layers on a %d-layer "
                "model: %s",
                list(self.model.layer_fractions),
                len(unique),
                num_hidden_layers,
                list(unique),
            )
        return unique

    # -- paths -------------------------------------------------------------- #

    def results_path(self, name: str) -> Path:
        """Resolve a filename inside the configured results directory."""
        return _project_root() / self.paths.results_dir / name

    def evalset_path(self, name: str) -> Path:
        """Resolve a filename inside the configured evalsets directory."""
        return _project_root() / self.paths.evalsets_dir / name

    def policy_path(self, name: str) -> Path:
        """Resolve a filename inside the configured policies directory."""
        return _project_root() / self.paths.policies_dir / name

    def evalset_ids(self) -> tuple[str, ...]:
        """Declared eval set ids, in config order."""
        return tuple(spec.id for spec in self.evalsets)


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


def _is_optional(typ: Any) -> bool:
    """True for ``X | None`` and ``Optional[X]``."""
    return get_origin(typ) in (Union, UnionType) and type(None) in get_args(typ)


def _unwrap_optional(typ: Any) -> Any:
    """Return ``X`` from ``X | None``, rejecting genuine multi-member unions."""
    args = [a for a in get_args(typ) if a is not type(None)]
    if len(args) != 1:
        raise ConfigError(f"unsupported union type in the config schema: {typ!r}")
    return args[0]


def _coerce(value: Any, typ: Any, path: str) -> Any:
    """Coerce and type-check one config value, naming its dotted path on error."""
    if _is_optional(typ):
        if value is None:
            return None
        return _coerce(value, _unwrap_optional(typ), path)
    if is_dataclass(typ):
        return _from_mapping(typ, value, path)
    origin = get_origin(typ)
    if origin is tuple:
        item_type = get_args(typ)[0]
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        return tuple(_coerce(v, item_type, f"{path}[{i}]") for i, v in enumerate(value))
    if origin is dict:
        key_type, val_type = get_args(typ)
        if key_type is not str:
            raise ConfigError(f"{path}: config mappings must be keyed by string")
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        return {k: _coerce(v, val_type, f"{path}.{k}") for k, v in value.items()}
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
    recorded next to its results, and the artifact's provenance block is then a
    lie about a run nobody can reproduce.

    Fields carrying a default may be omitted; fields without one may not.
    """
    where = path or "<root>"
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    field_map = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(field_map))
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {unknown}; known keys are {sorted(field_map)}"
        )
    missing = sorted(
        name
        for name, f in field_map.items()
        if name not in data
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    )
    if missing:
        raise ConfigError(f"{where}: missing required key(s) {missing}")
    kwargs = {
        name: _coerce(data[name], hints[name], f"{path}.{name}" if path else name)
        for name in field_map
        if name in data
    }
    return cls(**kwargs)


def _apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply dotted-key overrides (e.g. ``validation.bootstrap_samples``).

    Overrides exist for ``--smoke``, which needs to shrink two or three values
    without a second config file drifting out of sync with the first. Overriding
    a key that does not exist is an error, and any override changes the config
    hash — correctly, because it describes a different run.
    """
    merged = json.loads(json.dumps(raw))  # deep copy through plain data
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        cursor: Any = merged
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise ConfigError(
                    f"override {dotted!r}: no such config section {part!r}"
                )
            cursor = cursor[part]
        leaf = parts[-1]
        if not isinstance(cursor, dict) or leaf not in cursor:
            raise ConfigError(f"override {dotted!r}: no such config key")
        cursor[leaf] = value
    return merged


def load_config(
    path: str | os.PathLike = "config.yaml",
    overrides: Optional[dict[str, Any]] = None,
) -> Config:
    """Load, validate and freeze the configuration.

    Args:
        path: Path to the YAML config.
        overrides: Dotted-key overrides applied before validation. Each override
            changes the config hash, which is correct: it is a different run.

    Returns:
        The frozen :class:`Config`.

    Raises:
        ConfigError: If the file is missing or malformed, contains an unknown or
            missing key, or violates one of the invariants asserted in the
            dataclasses' ``__post_init__``.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: top level must be a mapping")
    if overrides:
        raw = _apply_overrides(raw, overrides)
    return _from_mapping(Config, raw)


