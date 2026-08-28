"""Load-time warrant resolution. ``SPEC.md`` §7.2, ``DECISIONS.md`` 012.

This is the load-bearing half of the policy engine and it has nothing to do
with Rego. Before a single rule may run, every operating point the bundle
declares is resolved against the warrant matrix, and **missing, expired,
refused, unvalidated, or below a declared minimum means the bundle does not
load**. Not a warning. Not a degraded mode.

The reason it has to be a hard failure is that the alternative is invisible. A
profile that warns and starts goes on serving traffic, quoting bounds in its
certificates that no valid warrant supplies, and nothing downstream can tell
the difference — the certificates look identical. Refusing at load time is the
only point at which the failure is cheap and legible: a deploy fails, a human
reads an error naming the warrant, and either validates it or changes the
profile.

Deploying a new profile therefore requires validating it first. That is the
point, not a cost.

## Seven checks, and why the last two are here

The first five are ``SPEC.md`` §7.2: presence, status, age, recall floor, FPR
ceiling. The last two come out of what Phase 4 and Phase 5 measured.

**Calibration.** A warrant makes two separable claims — how well the detector
*ranks*, and what its operating point *spends*. Phase 4 measured warrants that
are ``VALID`` on the first and ``DRIFTED`` on the second. A profile declaring
``min_recall`` is making a ranking claim; a profile whose economics are sized
on a flag rate is making a calibration claim as well, and until this check
existed nothing consumed the difference.

**Power.** A profile declaring a calibration sensitivity is asserting it can
detect a deviation of that size. Whether it can is a property of the sample the
warrant was measured on, and it is arithmetic rather than opinion: separating a
25% deviation from a 5% budget needs n ≥ 1441, and these envelopes were
measured at n = 600. A profile asking for 10% sensitivity there is asking for
something no measurement in this repo can supply, and it fails to load for
exactly the same reason a profile asking for recall 0.50 against a warrant
measured at 0.13 fails: the claim is not backed. Same mechanism, applied to a
limit we now know the sample size imposes.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any, Optional

from ..model.enums import CalibrationStatus, WarrantStatus
from ..model.serde import utc_now
from ..model.warrant import Warrant
from ..validation.calibration import n_to_detect_deviation
from .bundle import OnCalibrationDrift, PolicyBundle, WarrantRequirement
from .errors import WarrantResolutionError

__all__ = ["ResolvedBundle", "ResolvedRequirement", "resolve_bundle"]


@dataclasses.dataclass(frozen=True)
class ResolvedRequirement:
    """One requirement, the warrant that satisfied it, and any caveat attached.

    Args:
        requirement: What was asked for.
        warrant: What backs it.
        claimed_flag_rate_budget: The budget this profile may quote. The
            declared target normally; the **measured interval** when the
            profile chose ``WIDEN_BUDGET`` and calibration had drifted.
        notes: Caveats that did not prevent loading but must reach the reader.
    """

    requirement: WarrantRequirement
    warrant: Warrant
    claimed_flag_rate_budget: Any
    notes: tuple[str, ...] = ()

    def to_payload(self) -> dict:
        return {
            "key": self.requirement.key.as_string(),
            "warrant_id": self.warrant.warrant_id,
            "claimed_flag_rate_budget": self.claimed_flag_rate_budget,
            "notes": list(self.notes),
        }


@dataclasses.dataclass(frozen=True)
class ResolvedBundle:
    """A bundle whose every declared operating point is backed by a valid warrant.

    Holding one of these is the proof that the checks ran. Nothing in this
    package evaluates a rule against an unresolved bundle, so a caller cannot
    reach the engine without having passed resolution.

    Args:
        bundle: The parsed bundle.
        resolved: One entry per requirement, in declaration order.
        resolved_at: The clock the age checks used.
    """

    bundle: PolicyBundle
    resolved: tuple[ResolvedRequirement, ...]
    resolved_at: datetime

    @property
    def warrants(self) -> tuple[Warrant, ...]:
        return tuple(r.warrant for r in self.resolved)

    @property
    def weakest_status(self) -> WarrantStatus:
        """What the policy's first rule keys on.

        Always ``VALID`` for a resolved bundle — resolution refuses anything
        else. Present so the rule input has a field to read rather than a
        constant, because the same field is what a *revoked* warrant moves
        during operation, and a rule that only ever sees one value is a rule
        nobody has tested.
        """
        statuses = {w.status for w in self.warrants}
        for candidate in (
            WarrantStatus.REFUSED,
            WarrantStatus.REVOKED,
            WarrantStatus.UNVALIDATED,
            WarrantStatus.STALE,
        ):
            if candidate in statuses:
                return candidate
        return WarrantStatus.VALID

    @property
    def primary(self) -> ResolvedRequirement:
        """The operating point the profile's inline decision runs at.

        The first declared requirement. Profiles may rely on several warrants —
        a detector for correctness and another for PII — but exactly one supplies
        the threshold the inline path compares a score against, and declaration
        order is what says which. Making it positional rather than a flag keeps
        the manifest from growing a field whose only job is to disambiguate a
        list of length one.
        """
        return self.resolved[0]

    def rego_facts(self) -> dict:
        """The facts the engine injects into every rule evaluation.

        **Injected, never accepted from the caller.** The first rule in
        ``SPEC.md`` §7.2 keys on ``warrant.weakest_status``, and a request that
        could supply its own value for that field could assert its way past the
        rule the whole product rests on. These come from the resolved bundle
        and the matrix behind it.
        """
        point = self.primary
        recall = point.warrant.metrics.recall
        return {
            "profile": self.bundle.name,
            "policy_version": self.bundle.stamp["policy_version"],
            "warrant": {
                "weakest_status": self.weakest_status.value,
                "warrant_id": point.warrant.warrant_id,
                "detector_id": point.warrant.detector_id,
                "envelope": point.warrant.eval_set_id,
                "n_test": point.warrant.n_test,
                "recall": None if recall is None else recall.value,
                "recall_ci_low": None if recall is None else recall.ci_low,
            },
            "operating_point": {
                "id": point.warrant.operating_point.operating_point_id,
                "threshold": point.warrant.operating_point.threshold,
                "target_flag_rate": point.warrant.operating_point.target_flag_rate,
                "claimed_flag_rate_budget": point.claimed_flag_rate_budget,
            },
        }

    def notes(self) -> tuple[str, ...]:
        out: list[str] = []
        for entry in self.resolved:
            out.extend(entry.notes)
        return tuple(out)

    def to_payload(self) -> dict:
        return {
            "profile": self.bundle.name,
            "version": self.bundle.version,
            "policy_hash": self.bundle.content_hash,
            "resolved_at": self.resolved_at.isoformat(),
            "weakest_status": self.weakest_status.value,
            "requirements": [r.to_payload() for r in self.resolved],
        }


def _check_presence(cell, requirement: WarrantRequirement) -> Optional[str]:
    """Presence and status. ``UNVALIDATED`` and ``REFUSED`` are kept distinct."""
    key = requirement.key.as_string()
    if cell is None or cell.warrant is None:
        return (
            f"{key}: UNVALIDATED — no warrant has ever been filed here. This is "
            "the modal state in production and it is not a failure of the "
            "detector; it means nobody has measured it on this envelope."
        )
    if cell.status is WarrantStatus.REFUSED:
        return (
            f"{key}: REFUSED — measured here and failed its controls"
            + (f" ({cell.reason})" if cell.reason else "")
            + ". Out of service until a human revalidates; there is no override "
            "(CLAUDE.md invariant 3)."
        )
    if cell.status is not WarrantStatus.VALID:
        return (
            f"{key}: {cell.status.value} — not a claim a profile may rely on"
            + (f" ({cell.reason})" if cell.reason else "")
        )
    return None


def _check_age(warrant: Warrant, requirement: WarrantRequirement, now: datetime) -> Optional[str]:
    key = requirement.key.as_string()
    age = warrant.age(now)
    if age > requirement.max_age:
        return (
            f"{key}: warrant is {age} old and the profile declares max_age "
            f"{requirement.max_age}. Age is a reason to stop relying on a "
            "number independently of drift."
        )
    if warrant.is_expired(now):
        return (
            f"{key}: warrant expired at {warrant.expires_at.isoformat()}"
        )
    return None


def _check_bounds(warrant: Warrant, requirement: WarrantRequirement) -> list[str]:
    """Recall floor against the lower bound, FPR ceiling against the upper.

    Both compared against the interval rather than the point estimate. A
    profile asking for "at least 25% recall" is asking for a guarantee, and 0.26
    with a lower bound of 0.18 does not supply one.
    """
    key = requirement.key.as_string()
    failures: list[str] = []

    recall = warrant.metrics.recall
    if recall is None:
        failures.append(
            f"{key}: the profile requires recall >= {requirement.min_recall} and "
            "this envelope supports no recall claim at all (single-class)"
        )
    else:
        lower = recall.ci_low if recall.ci_low is not None else recall.value
        if lower < requirement.min_recall:
            failures.append(
                f"{key}: the profile requires recall >= {requirement.min_recall}; "
                f"the warrant's lower bound is {lower:.4f} (point estimate "
                f"{recall.value:.4f}, n={recall.n}). Compared against the lower "
                "bound because a minimum is a guarantee."
            )

    ceiling = requirement.max_fpr_hard_negatives
    fpr = warrant.metrics.fpr_hard_negatives
    if ceiling is not None:
        if fpr is None:
            # An absence is not a pass. This is the same rule as UNVALIDATED
            # never counting as VALID: the profile declared a ceiling, nothing
            # measured the quantity it constrains, and letting that through
            # would put an unbacked guarantee on every certificate the profile
            # issues. Hard-negative FPR is measured on hard-negatives-200; a
            # detector holding no warrant there cannot have a ceiling enforced.
            failures.append(
                f"{key}: the profile declares hard-negative FPR <= {ceiling}, "
                "and this warrant carries no hard-negative FPR measurement at "
                "all. An unmeasured ceiling is not a satisfied one. Either "
                "validate this detector on the hard-negative set, or declare "
                "max_fpr_hard_negatives: null and say so on the record."
            )
        else:
            upper = fpr.ci_high if fpr.ci_high is not None else fpr.value
            if upper > ceiling:
                failures.append(
                    f"{key}: the profile requires hard-negative FPR <= "
                    f"{ceiling}; the warrant's upper bound is {upper:.4f} "
                    f"(point estimate {fpr.value:.4f})."
                )
    return failures


def _check_power(warrant: Warrant, requirement: WarrantRequirement) -> Optional[str]:
    """Whether the warrant's sample could detect the deviation the profile declares.

    Carry-over from Phase 5. ``n_to_detect_deviation`` is the same function the
    validation harness uses to decide whether a calibration claim is
    underpowered, applied here to a sensitivity a *profile* declared rather than
    one the harness assumed.
    """
    target = warrant.operating_point.target_flag_rate
    if target is None:
        # Nothing to be sensitive *to*. Not a failure: a profile can decline to
        # make a budget claim, and this operating point declares no budget.
        return None

    needed = n_to_detect_deviation(target, requirement.calibration.sensitivity)
    if needed <= warrant.n_test:
        return None

    key = requirement.key.as_string()
    return (
        f"{key}: the profile declares calibration sensitivity "
        f"{requirement.calibration.sensitivity:.0%}, which needs n >= {needed} "
        f"to separate from a budget of {target:g}; this warrant was measured at "
        f"n = {warrant.n_test}. The sample cannot support the claim the profile "
        "is making, and no rule can fix that — it needs more test items."
    )


def _check_calibration(
    warrant: Warrant, requirement: WarrantRequirement
) -> tuple[Optional[str], Any, tuple[str, ...]]:
    """Apply the profile's position on a drifted budget.

    Returns:
        ``(failure, claimed_budget, notes)``. ``failure`` is None when the
        bundle may load.
    """
    key = requirement.key.as_string()
    declared = warrant.operating_point.target_flag_rate
    claim = warrant.calibration
    policy = requirement.calibration.on_drift

    if claim is None or claim.status is CalibrationStatus.UNKNOWN:
        return (
            None,
            declared,
            (
                f"{key}: no calibration verdict — the budget claim is untested "
                "here, which is not the same as tested and passed.",
            ),
        )

    if claim.status is CalibrationStatus.DRIFTED:
        if policy == OnCalibrationDrift.REFUSE:
            return (
                f"{key}: calibration has DRIFTED and the profile declares "
                f"on_drift: REFUSE. The ranking claim still holds — this is not "
                f"a broken detector — but the operating point no longer spends "
                f"the budget it declared ({claim.detail}). A tier sized on a "
                "flag rate cannot absorb that.",
                None,
                (),
            )
        if policy == OnCalibrationDrift.WIDEN_BUDGET:
            widened = (
                None
                if claim.realised is None
                else {
                    "value": claim.realised.value,
                    "ci_low": claim.realised.ci_low,
                    "ci_high": claim.realised.ci_high,
                    "n": claim.realised.n,
                    "declared_target": declared,
                    "note": "measured, not declared: calibration drifted and the "
                    "profile widened rather than refusing",
                }
            )
            return (
                None,
                widened,
                (
                    f"{key}: calibration DRIFTED; the profile quotes the measured "
                    f"flag-rate interval instead of its declared budget "
                    f"{declared!r}. The ranking claim is unchanged.",
                ),
            )
        return (
            None,
            declared,
            (
                f"{key}: calibration DRIFTED and the profile declares on_drift: "
                "IGNORE. Recorded so that 'considered and does not apply' is "
                "distinguishable from 'never thought about'.",
            ),
        )

    # CALIBRATED.
    if claim.underpowered:
        return (
            None,
            declared,
            (
                f"{key}: calibration reads CALIBRATED but the interval straddles "
                f"the ±{claim.tolerance:.0%} band at n={warrant.n_test} — "
                f"unresolved rather than passed. Needs n >= {claim.n_to_detect}.",
            ),
        )
    return None, declared, ()


def resolve_bundle(
    bundle: PolicyBundle,
    matrix,
    *,
    now: Optional[datetime] = None,
) -> ResolvedBundle:
    """Resolve every operating point the bundle declares, or refuse to load it.

    Args:
        bundle: The parsed bundle.
        matrix: A :class:`~src.matrix.matrix.WarrantMatrix`.
        now: Clock for the age checks.

    Returns:
        A :class:`ResolvedBundle`.

    Raises:
        WarrantResolutionError: If **any** requirement is unmet. Every failure
            is collected before raising rather than raising on the first: a
            policy author fixing one missing warrant at a time, one deploy at a
            time, is the workflow that makes people disable the check.
    """
    clock = now or utc_now()
    failures: list[str] = []
    resolved: list[ResolvedRequirement] = []

    for requirement in bundle.requires_warrant:
        cell = matrix.cell(requirement.key)

        problem = _check_presence(cell, requirement)
        if problem is not None:
            failures.append(problem)
            continue

        warrant = cell.warrant
        entry_failures: list[str] = []

        entry_notes: list[str] = []
        if requirement.max_fpr_hard_negatives is None:
            entry_notes.append(
                f"{requirement.key.as_string()}: no hard-negative FPR ceiling "
                "declared on this envelope. Declared absent rather than left "
                "off, so the gap is on the record."
            )

        aged = _check_age(warrant, requirement, clock)
        if aged is not None:
            entry_failures.append(aged)
        entry_failures.extend(_check_bounds(warrant, requirement))

        underpowered = _check_power(warrant, requirement)
        if underpowered is not None:
            entry_failures.append(underpowered)

        calibration_failure, budget, notes = _check_calibration(warrant, requirement)
        if calibration_failure is not None:
            entry_failures.append(calibration_failure)

        if entry_failures:
            failures.extend(entry_failures)
            continue

        resolved.append(
            ResolvedRequirement(
                requirement=requirement,
                warrant=warrant,
                claimed_flag_rate_budget=budget,
                notes=tuple(entry_notes) + notes,
            )
        )

    if failures:
        raise WarrantResolutionError(
            "policy bundle %s/%s failed to load: %d of %d declared operating "
            "point(s) are not backed by a valid warrant.\n  - %s"
            % (
                bundle.name,
                bundle.version,
                len(bundle.requires_warrant) - len(resolved),
                len(bundle.requires_warrant),
                "\n  - ".join(failures),
            ),
            failures=tuple(failures),
        )

    return ResolvedBundle(bundle=bundle, resolved=tuple(resolved), resolved_at=clock)
