"""Issue the three operating points and decide one input under all three profiles.

The Phase 7 gate has two halves and this module is the second: *three profiles
produce three different actions on one input, at three points on one measured
curve*.

**"One curve" is checkable here, not asserted.** All three profiles name the
same detector and the same envelope, so the only thing that differs between
them is where their threshold sits on one detector's score distribution.
:func:`issue_operating_points` produces the three warrants by running the same
validation three times at three flag-rate budgets; :func:`run_profile_comparison`
loads the three bundles against the resulting matrix and evaluates one request
under each.

**The probe score is computed, not chosen.** A score written into this module
would be a number tuned to make three actions come out, which is the demo
equivalent of selecting a threshold on test. It is derived instead from the
issued thresholds: the midpoint between the two highest, which is by
construction below the most conservative profile's threshold and at or above
every other. If the operating points fail to separate, that is reported as a
failure rather than papered over.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..config import Config
from ..matrix import WarrantMatrix
from ..model.enums import Action, Severity
from ..store import Ledger
from ..validation.evalsets import EvalSet, ExtractionCache
from ..validation.runner import ValidationRun, validate
from .bundle import PolicyBundle
from .engine import build_engine
from .errors import PolicyError
from .resolution import ResolvedBundle, resolve_bundle

__all__ = [
    "ProfileComparison",
    "ProfileOutcome",
    "issue_operating_points",
    "run_profile_comparison",
]

_LOG = logging.getLogger(__name__)


def issue_operating_points(
    config: Config,
    evalset: EvalSet,
    cache: ExtractionCache,
    *,
    variant: str,
    detector_id: str,
    detector_version: str,
    ledger: Ledger,
    canary_cache: Optional[ExtractionCache] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[ValidationRun, ...]:
    """Validate the same detector at each profile's flag-rate budget.

    One detector, one envelope, three thresholds — which is what ``SPEC.md``
    §7.3's "three points on one measured curve" requires and what three
    separately-tuned detectors would not be.

    Args:
        config: Resolved config. ``config.profiles`` supplies the operating
            point ids and their budgets; nothing here is hardcoded.
        evalset: The set to validate on.
        cache: Its extraction.
        variant: Tier variant, e.g. ``"T1-last_token"``.
        detector_id: Detector identity.
        detector_version: Semver plus weights hash.
        ledger: Where the warrants are filed.
        canary_cache: Canary extraction, so the canary control really runs.
        progress: Called with progress lines.

    Returns:
        One :class:`ValidationRun` per profile, in ``config.profiles`` order.
    """
    say = progress or (lambda message: None)
    runs: list[ValidationRun] = []

    for name, profile in config.profiles.items():
        say(
            f"validating {detector_id} at {profile.operating_point} "
            f"(budget f={profile.target_flag_rate:g}) for profile {name}"
        )
        run = validate(
            config,
            evalset,
            cache,
            variant=variant,
            detector_id=detector_id,
            detector_version=detector_version,
            operating_point_id=profile.operating_point,
            target_flag_rate=profile.target_flag_rate,
            # The profile's floors are NOT passed as issuance criteria. A
            # warrant records what was measured; whether that clears a
            # profile's bar is the policy loader's question, and deciding it at
            # issuance would collapse "this detector scores X" into "this
            # profile may run", which are different facts with different
            # lifetimes.
            canary_cache=canary_cache,
            progress=progress,
        )
        ledger.append_warrant(run.warrant)
        runs.append(run)

    return tuple(runs)


@dataclasses.dataclass(frozen=True)
class ProfileOutcome:
    """What one profile decided about the shared input, and on what basis.

    Args:
        profile: Profile name.
        policy_version: ``"<profile>/<version>"``.
        policy_hash: Content hash of manifest and rules.
        operating_point: The threshold's id.
        threshold: The measured threshold, from the resolved warrant.
        recall: Point estimate at that threshold.
        recall_ci: ``(low, high)``.
        measured_flag_rate: What the operating point actually spent on test.
        target_flag_rate: What it was selected to spend, on validation.
        fired: Whether the detector fired at this profile's threshold.
        action: What the rules decided.
        rule_id: Which rule fired.
        reason: Why, in words.
        notes: Load-time caveats carried through from resolution.
    """

    profile: str
    policy_version: str
    policy_hash: str
    operating_point: str
    threshold: float
    recall: Optional[float]
    recall_ci: tuple[Optional[float], Optional[float]]
    measured_flag_rate: Optional[float]
    target_flag_rate: Optional[float]
    fired: bool
    action: Action
    rule_id: str
    reason: str
    notes: tuple[str, ...]

    def to_payload(self) -> dict:
        return {
            "profile": self.profile,
            "policy_version": self.policy_version,
            "policy_hash": self.policy_hash,
            "operating_point": self.operating_point,
            "threshold": self.threshold,
            "recall": self.recall,
            "recall_ci": list(self.recall_ci),
            "measured_flag_rate": self.measured_flag_rate,
            "target_flag_rate": self.target_flag_rate,
            "fired": self.fired,
            "action": self.action.value,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "notes": list(self.notes),
        }


@dataclasses.dataclass(frozen=True)
class ProfileComparison:
    """Three profiles, one input, and whatever they each decided.

    Args:
        request: The shared input document, exactly as handed to every engine.
        rows: One per profile, ordered as the bundles were loaded.
        distinct_actions: How many different actions came out. The gate wants
            three; fewer is reported rather than hidden.
        refused: Profiles whose bundle failed to load, with the reason. Not an
            error here — a profile that cannot be backed is a result.
    """

    request: dict
    rows: tuple[ProfileOutcome, ...]
    distinct_actions: int
    refused: tuple[tuple[str, str], ...] = ()

    def to_payload(self) -> dict:
        return {
            "request": self.request,
            "rows": [row.to_payload() for row in self.rows],
            "distinct_actions": self.distinct_actions,
            "refused": [{"profile": name, "reason": why} for name, why in self.refused],
        }


def _probe_score(thresholds: Sequence[float]) -> float:
    """A score that separates the most conservative profile from the rest.

    Derived from the issued thresholds rather than written down. The two
    highest are taken and the midpoint returned: strictly below the highest, at
    or above every other. A hardcoded score would be a number tuned until three
    actions appeared, which is selecting on the outcome.

    Args:
        thresholds: The measured thresholds, in any order.

    Returns:
        The score to evaluate.

    Raises:
        PolicyError: If the two highest thresholds coincide, in which case the
            operating points did not separate and no score can distinguish the
            profiles. That is a finding about the curve, not a bug to route
            around.
    """
    ordered = sorted(thresholds, reverse=True)
    if len(ordered) < 2:
        raise PolicyError("a comparison needs at least two profiles")
    if ordered[0] <= ordered[1]:
        raise PolicyError(
            "the two most conservative operating points have the same "
            f"threshold ({ordered[0]!r}); they are the same classifier and no "
            "input can distinguish the profiles. The flag-rate budgets need to "
            "be further apart, or the score distribution is too coarse here."
        )
    return (ordered[0] + ordered[1]) / 2.0


def run_profile_comparison(
    config: Config,
    ledger: Ledger,
    *,
    policies_dir: Path,
    now: Optional[Any] = None,
) -> ProfileComparison:
    """Load every bundle against the matrix and decide one request under each.

    Args:
        config: Resolved config, for the profile list.
        ledger: The store holding the warrants just issued.
        policies_dir: Directory of bundle directories.
        now: Clock, for the age checks.

    Returns:
        A :class:`ProfileComparison`.
    """
    matrix = WarrantMatrix.from_ledger(ledger, now=now)

    resolved: list[ResolvedBundle] = []
    refused: list[tuple[str, str]] = []
    for name in config.profiles:
        directory = Path(policies_dir) / name
        try:
            bundle = PolicyBundle.load(directory)
            resolved.append(resolve_bundle(bundle, matrix, now=now))
        except PolicyError as exc:
            _LOG.warning("profile %s did not load: %s", name, exc)
            refused.append((name, str(exc)))

    if not resolved:
        return ProfileComparison(
            request={}, rows=(), distinct_actions=0, refused=tuple(refused)
        )

    score = _probe_score(
        [r.primary.warrant.operating_point.threshold for r in resolved]
    )

    # One request, handed unchanged to every engine. The bundle's own facts --
    # its threshold, its warrant -- are injected by the engine and are the only
    # thing that differs between the three evaluations.
    request = {
        "detector": {"score": score},
        "finding": {
            "category": "HALLUCINATION",
            "severity": int(Severity.HIGH),
            "confidence_band": "UNCERTAIN",
        },
        "action": {"reversibility": 0},
    }

    rows: list[ProfileOutcome] = []
    for entry in resolved:
        engine = build_engine(entry)
        decision = engine.decide(request)
        warrant = entry.primary.warrant
        recall = warrant.metrics.recall
        flag_rate = warrant.metrics.flag_rate
        threshold = warrant.operating_point.threshold
        rows.append(
            ProfileOutcome(
                profile=entry.bundle.name,
                policy_version=decision.policy_version,
                policy_hash=decision.policy_hash,
                operating_point=warrant.operating_point.operating_point_id,
                threshold=threshold,
                recall=None if recall is None else recall.value,
                recall_ci=(
                    (None, None) if recall is None else (recall.ci_low, recall.ci_high)
                ),
                measured_flag_rate=None if flag_rate is None else flag_rate.value,
                target_flag_rate=warrant.operating_point.target_flag_rate,
                fired=score >= threshold,
                action=decision.action,
                rule_id=decision.rule_id,
                reason=decision.reason,
                notes=entry.notes(),
            )
        )

    return ProfileComparison(
        request=request,
        rows=tuple(rows),
        distinct_actions=len({row.action for row in rows}),
        refused=tuple(refused),
    )
