"""Policy as versioned, content-hashed data. ``SPEC.md`` §7.1.

A bundle is a directory holding a manifest and a Rego module:

``` text
policies/customer_support/
  bundle.yaml     profile, version, requires_warrant, weighted_error
  policy.rego     the rules
```

**The rules are Rego and are not interpreted here.** ``SPEC.md`` §7.1 says *do
not write a DSL*, and the temptation it is guarding against is not inventing a
syntax — it is the smaller, more reasonable-looking step of parsing a few
``when``/``then`` pairs out of the YAML because the real engine is awkward to
install. That path ends with a policy language nobody has specified and whose
semantics live in whichever function last touched it. The manifest carries
declarations; ``policy.rego`` carries logic; :mod:`src.policy.engine` is the
only thing that evaluates it.

**Both files are hashed together.** Versioning the manifest alone would let the
rules change under a fixed version string, and every certificate stamps that
string as its account of what decided. The version is what a human cites; the
hash is what makes the citation checkable.

## What the manifest declares beyond SPEC §7.2

Two fields come out of what Phase 5 measured, and both make a claim the profile
was already implicitly making into one the loader can check.

``calibration.on_drift`` — a profile declaring a flag-rate budget is making a
**calibration** claim, which is separate from the **ranking** claim it makes by
declaring ``min_recall``. Phase 4 measured warrants that are ``VALID`` on
ranking and ``DRIFTED`` on calibration, and until now nothing consumed that
distinction. This field is what a profile does about it.

``calibration.sensitivity`` — the relative deviation from the budget the
profile considers worth acting on. Checked at load time against the **power the
warrant's sample actually has**, because a profile asking to detect a 10%
budget deviation on a 600-item test set is asking for something no measurement
here can supply. That is the same mechanism as the recall and FPR minima,
applied to a claim we now know the sample size cannot back.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from ..model.warrant import WarrantKey
from .errors import BundleError

__all__ = [
    "CalibrationRequirement",
    "OnCalibrationDrift",
    "PolicyBundle",
    "WarrantRequirement",
    "parse_duration",
]

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(s|m|h|d)\s*$", re.IGNORECASE)
_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}


def parse_duration(text: str) -> timedelta:
    """Parse ``"24h"``, ``"90m"``, ``"7d"``.

    A bare number is rejected rather than assumed to be seconds. ``max_age: 24``
    means one of two very different policies depending on the unit a reader
    supplies from memory, and a max age wrong by 3600× fails open.

    Args:
        text: The duration.

    Returns:
        A :class:`datetime.timedelta`.

    Raises:
        BundleError: If the text is not a number followed by ``s``/``m``/``h``/``d``.
    """
    match = _DURATION.match(str(text))
    if match is None:
        raise BundleError(
            f"cannot parse duration {text!r}; write it with an explicit unit, "
            "e.g. '24h', '90m', '7d'. A bare number is refused because the "
            "wrong assumed unit fails open by a factor of 60 or 3600."
        )
    return timedelta(**{_UNITS[match.group(2).lower()]: float(match.group(1))})


class OnCalibrationDrift:
    """What a profile does when a warrant's budget has drifted but its ranking holds.

    Not an ``Enum`` with an ordering: these are three unrelated policies, and
    inviting a comparison between them is how "refuse" quietly becomes "the
    strictest of the three" in somebody's sort.
    """

    #: The bundle fails to load. For profiles where the budget is load-bearing —
    #: a high-volume inline tier sized on a flag rate cannot absorb a budget
    #: that is 40% wrong.
    REFUSE = "REFUSE"
    #: Load, but the profile's claimed flag rate widens to the measured
    #: interval rather than the declared target. The ranking claim is untouched.
    WIDEN_BUDGET = "WIDEN_BUDGET"
    #: Load unchanged. Legitimate where the profile's economics do not depend on
    #: the flag rate — a low-volume, escalation-heavy tier pays per item either
    #: way. Recorded explicitly so that "we didn't think about it" and "we
    #: considered it and it does not apply" are different states.
    IGNORE = "IGNORE"

    ALL = (REFUSE, WIDEN_BUDGET, IGNORE)


@dataclasses.dataclass(frozen=True)
class CalibrationRequirement:
    """A profile's position on the calibration half of a warrant.

    Args:
        on_drift: One of :class:`OnCalibrationDrift`.
        sensitivity: Relative deviation from the declared budget this profile
            considers worth acting on, e.g. ``0.25`` for 25%. Checked at load
            time against the warrant's achievable power.
    """

    on_drift: str
    sensitivity: float

    def __post_init__(self) -> None:
        if self.on_drift not in OnCalibrationDrift.ALL:
            raise BundleError(
                f"calibration.on_drift must be one of "
                f"{list(OnCalibrationDrift.ALL)}, got {self.on_drift!r}"
            )
        if not 0.0 < self.sensitivity < 1.0:
            raise BundleError(
                "calibration.sensitivity is a relative deviation in (0, 1), "
                f"e.g. 0.25 for 25%; got {self.sensitivity!r}"
            )

    @classmethod
    def parse(cls, data: Optional[Mapping[str, Any]]) -> "CalibrationRequirement":
        """Build from the manifest, defaulting to the strictest useful position.

        The default is ``REFUSE`` at the tolerance the validation harness itself
        uses. A profile that has not thought about calibration should not
        silently inherit "ignore it" — the whole reason the claim was split in
        two is that one half was being carried by the other.
        """
        if data is None:
            return cls(on_drift=OnCalibrationDrift.REFUSE, sensitivity=0.25)
        if not isinstance(data, Mapping):
            raise BundleError(f"calibration must be a mapping, got {type(data).__name__}")
        unknown = set(data) - {"on_drift", "sensitivity"}
        if unknown:
            raise BundleError(
                f"unknown calibration field(s) {sorted(unknown)}; a misspelt "
                "field would otherwise be silently ignored and the profile "
                "would run under a policy nobody wrote"
            )
        return cls(
            on_drift=str(data.get("on_drift", OnCalibrationDrift.REFUSE)),
            sensitivity=float(data.get("sensitivity", 0.25)),
        )


@dataclasses.dataclass(frozen=True)
class WarrantRequirement:
    """One operating point a bundle relies on, and what it demands of it.

    Args:
        detector: Detector id.
        operating_point: Operating point id. Recall at one threshold says
            nothing about recall at another, so this is part of the key rather
            than a detail.
        envelope: ``eval_set_id``. The matrix's envelope axis is keyed by this
            and not by the envelope content hash.
        min_recall: Compared against the interval's **lower** bound.
        max_fpr_hard_negatives: Compared against the **upper** bound. ``None``
            means *this profile declares no hard-negative ceiling on this
            envelope*, which is a different statement from declaring one and
            having it go unchecked. The field is required in the manifest so
            that the difference is written down: hard-negative FPR is measured
            on ``hard-negatives-200``, and a detector holding no warrant there
            cannot have a ceiling enforced against it anywhere.
        max_age: How old the warrant may be.
        calibration: The calibration position and sensitivity.
    """

    detector: str
    operating_point: str
    envelope: str
    min_recall: float
    max_fpr_hard_negatives: Optional[float]
    max_age: timedelta
    calibration: CalibrationRequirement

    @property
    def key(self) -> WarrantKey:
        """The three-part matrix address this requirement resolves against."""
        return WarrantKey(self.detector, self.operating_point, self.envelope)

    @classmethod
    def parse(cls, data: Mapping[str, Any], index: int) -> "WarrantRequirement":
        """Build one entry of ``requires_warrant``.

        Args:
            data: The mapping.
            index: Position in the list, for error messages.

        Returns:
            A :class:`WarrantRequirement`.

        Raises:
            BundleError: On a missing or unknown field.
        """
        required = {
            "detector",
            "operating_point",
            "envelope",
            # Required, not defaulted. A defaulted ceiling of 1.0 and an
            # explicit "no ceiling declared here" are the same arithmetic and
            # different claims, and only one of them is auditable.
            "max_fpr_hard_negatives",
        }
        known = required | {
            "min_recall",
            "max_fpr_hard_negatives",
            "max_age",
            "calibration",
        }
        if not isinstance(data, Mapping):
            raise BundleError(
                f"requires_warrant[{index}] must be a mapping, got "
                f"{type(data).__name__}"
            )
        missing = sorted(required - set(data))
        if missing:
            raise BundleError(
                f"requires_warrant[{index}] is missing {missing}. All three "
                "parts of the key are required: recall at one threshold says "
                "nothing about recall at another, and an envelope is not "
                "optional context. max_fpr_hard_negatives is required too, and "
                "may be null — 'no ceiling declared on this envelope' has to be "
                "written down rather than inferred from an absent field."
            )
        unknown = sorted(set(data) - known)
        if unknown:
            raise BundleError(
                f"requires_warrant[{index}] has unknown field(s) {unknown}. A "
                "misspelt minimum would be silently dropped and the profile "
                "would load against a bar nobody set."
            )
        return cls(
            detector=str(data["detector"]),
            operating_point=str(data["operating_point"]),
            envelope=str(data["envelope"]),
            min_recall=float(data.get("min_recall", 0.0)),
            max_fpr_hard_negatives=(
                None
                if data["max_fpr_hard_negatives"] is None
                else float(data["max_fpr_hard_negatives"])
            ),
            max_age=parse_duration(data.get("max_age", "24h")),
            calibration=CalibrationRequirement.parse(data.get("calibration")),
        )


@dataclasses.dataclass(frozen=True)
class PolicyBundle:
    """A parsed, content-hashed policy bundle. Not yet resolved against a matrix.

    Parsing and resolution are separate because they fail for different reasons
    and at different times: a malformed bundle is a policy author's problem now,
    an unresolvable one is a statement about the detector fleet that can become
    true or false as warrants are issued and revoked.

    Args:
        name: Profile name. Must match a key in ``config.profiles``.
        version: Human-citable version, e.g. ``"3.1"``.
        entrypoint: Rego query the engine evaluates, e.g.
            ``"data.controlplane.decision"``.
        requires_warrant: Every operating point this bundle relies on.
        weighted_error: The objective's weights, versioned with the bundle so a
            threshold can be re-derived from the rules that chose it
            (``SPEC.md`` §7.4).
        rego_source: The Rego module text.
        content_hash: Over the manifest **and** the Rego together.
        source_dir: Where it was loaded from, for error messages.
    """

    name: str
    version: str
    entrypoint: str
    requires_warrant: tuple[WarrantRequirement, ...]
    weighted_error: Mapping[str, float]
    rego_source: str
    content_hash: str
    source_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise BundleError("a bundle must declare both profile and version")
        if not self.requires_warrant:
            raise BundleError(
                f"bundle {self.name!r} declares no requires_warrant. A policy "
                "that relies on no measured operating point is a policy whose "
                "actions rest on nothing, which is the state this loader "
                "exists to make impossible (SPEC.md §7.2)."
            )
        if not self.entrypoint.startswith("data."):
            raise BundleError(
                f"entrypoint {self.entrypoint!r} must be a Rego data reference, "
                "e.g. 'data.controlplane.decision'"
            )

    @property
    def stamp(self) -> dict[str, str]:
        """What every certificate this bundle decides carries.

        Both halves travel together. The version is what a human cites in an
        incident review; the hash is what lets them check that the rules they
        are reading are the rules that ran.
        """
        return {"policy_version": f"{self.name}/{self.version}", "policy_hash": self.content_hash}

    # -- loading ------------------------------------------------------------ #

    @classmethod
    def load(cls, directory: str | Path) -> "PolicyBundle":
        """Read and parse a bundle directory.

        Args:
            directory: Path holding ``bundle.yaml`` and the Rego module it names.

        Returns:
            A :class:`PolicyBundle`.

        Raises:
            BundleError: If either file is missing or the manifest is malformed.
        """
        path = Path(directory)
        manifest_path = path / "bundle.yaml"
        if not manifest_path.is_file():
            raise BundleError(f"no bundle.yaml in {path}")

        manifest_text = manifest_path.read_text(encoding="utf-8")
        try:
            manifest = yaml.safe_load(manifest_text)
        except yaml.YAMLError as exc:
            raise BundleError(f"{manifest_path} is not valid YAML: {exc}") from exc
        if not isinstance(manifest, Mapping):
            raise BundleError(f"{manifest_path} must contain a mapping")

        rules_name = str(manifest.get("rules", "policy.rego"))
        rego_path = path / rules_name
        if not rego_path.is_file():
            raise BundleError(
                f"bundle.yaml names rules {rules_name!r} but {rego_path} does "
                "not exist. The rules are Rego and are not written in the "
                "manifest (SPEC.md §7.1: do not write a DSL)."
            )
        rego_source = rego_path.read_text(encoding="utf-8")

        return cls.parse(manifest, rego_source, source_dir=path)

    @classmethod
    def parse(
        cls,
        manifest: Mapping[str, Any],
        rego_source: str,
        *,
        source_dir: Optional[Path] = None,
    ) -> "PolicyBundle":
        """Parse an in-memory manifest and Rego module.

        Args:
            manifest: The decoded ``bundle.yaml``.
            rego_source: The Rego module text.
            source_dir: Where it came from, if anywhere.

        Returns:
            A :class:`PolicyBundle`.

        Raises:
            BundleError: On any malformed or unknown field.
        """
        known = {
            "profile",
            "version",
            "entrypoint",
            "rules",
            "requires_warrant",
            "weighted_error",
        }
        unknown = sorted(set(manifest) - known)
        if unknown:
            raise BundleError(
                f"unknown manifest field(s) {unknown}. Bundles are refused "
                "rather than partially understood: a field the loader ignores "
                "is a rule the author believes is in force."
            )

        raw_requirements = manifest.get("requires_warrant") or []
        if not isinstance(raw_requirements, list):
            raise BundleError("requires_warrant must be a list")

        weights = manifest.get("weighted_error") or {}
        if not isinstance(weights, Mapping):
            raise BundleError("weighted_error must be a mapping")

        return cls(
            name=str(manifest.get("profile", "")),
            version=str(manifest.get("version", "")),
            entrypoint=str(manifest.get("entrypoint", "data.controlplane.decision")),
            requires_warrant=tuple(
                WarrantRequirement.parse(item, index)
                for index, item in enumerate(raw_requirements)
            ),
            weighted_error={str(k): float(v) for k, v in weights.items()},
            rego_source=rego_source,
            content_hash=content_hash_of(manifest, rego_source),
            source_dir=source_dir,
        )


def content_hash_of(manifest: Mapping[str, Any], rego_source: str) -> str:
    """Hash the manifest and the rules together.

    Together, and not separately: hashing the manifest alone would let the rules
    change under a fixed version string, and every certificate stamps that
    string as its account of what decided. The manifest is canonicalised through
    sorted-key YAML rather than hashed as raw bytes, so that reformatting a
    comment does not invalidate every certificate ever issued under it, while
    any change to a declared value does.

    Args:
        manifest: The decoded manifest.
        rego_source: The Rego text, hashed verbatim — whitespace inside a rule
            body can change what it matches.

    Returns:
        ``"sha256:"`` followed by the digest.
    """
    canonical = yaml.safe_dump(_plain(manifest), sort_keys=True, default_flow_style=False)
    digest = hashlib.sha256()
    digest.update(canonical.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(rego_source.encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _plain(value: Any) -> Any:
    """Convert a parsed manifest into plain builtins for canonical dumping."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value
