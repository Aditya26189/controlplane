"""The policy engine. ``SPEC.md`` §7.

Three groups, and only the middle one is about Rego.

**Bundles** are versioned, content-hashed data. A field the loader does not
understand is refused rather than ignored, because a field the loader ignores
is a rule its author believes is in force.

**Resolution** is the load-bearing half, and it does not evaluate a single
rule. Every operating point a bundle declares is checked against the matrix,
and anything missing, expired, refused, unvalidated or below a declared
minimum stops the bundle loading (``DECISIONS.md`` 012). ``test_the_phase_7_gate``
is the first half of the gate sentence.

**The engine** evaluates the rules, and its tests are mostly about the two
silent traps in the binding.
"""

from __future__ import annotations

import dataclasses
from datetime import timedelta
from pathlib import Path

import pytest

from controlplane.model import (
    Action,
    Metric,
    MetricKind,
    Severity,
    WarrantStatus,
    utc_now,
)
from controlplane.model.enums import CalibrationStatus
from controlplane.model.calibration import CalibrationClaim
from controlplane.matrix import WarrantMatrix
from controlplane.policy import (
    BundleError,
    OnCalibrationDrift,
    PolicyBundle,
    WarrantResolutionError,
    build_engine,
    choose_threshold,
    error_rates_at,
    parse_duration,
    resolve_bundle,
    weighted_error,
)
from controlplane.policy.engine import RESERVED_KEYS
from controlplane.validation.calibration import n_to_detect_deviation

from .factories import failing_controls, make_operating_point, make_warrant

DETECTOR = "probe-qwen2.5-7b-L23"
ENVELOPE = "triviaqa-600"
WEIGHTS = {"w_fpr_benign": 50.0, "w_fnr": 5.0, "w_fpr_hard_negative": 2.0}

RULES = """package controlplane
import rego.v1

default decision := {"action": "ALLOW", "rule_id": "R-default-allow", "reason": "no rule fired"}

fired if input.detector.score >= input.operating_point.threshold

candidates contains c if {
	input.warrant.weakest_status != "VALID"
	input.action.reversibility >= 2
	c := {"priority": 100, "action": "BLOCK", "rule_id": "R-no-warrant-irreversible",
	      "reason": "no valid warrant backs an irreversible action"}
}

candidates contains c if {
	fired
	input.finding.severity >= 3
	c := {"priority": 50, "action": "ESCALATE", "rule_id": "R-escalate-on-flag",
	      "reason": "detector fired at the validated operating point"}
}

priorities := [c.priority | some c in candidates]

decision := {"action": c.action, "rule_id": c.rule_id, "reason": c.reason} if {
	count(priorities) > 0
	some c in candidates
	c.priority == max(priorities)
}
"""


def manifest(**overrides) -> dict:
    """A well-formed manifest, with one requirement, ready to be broken."""
    requirement = {
        "detector": DETECTOR,
        "operating_point": "P-conservative",
        "envelope": ENVELOPE,
        "min_recall": 0.10,
        "max_fpr_hard_negatives": None,
        "max_age": "24h",
        "calibration": {"on_drift": "REFUSE", "sensitivity": 0.25},
    }
    requirement.update(overrides.pop("requirement", {}))
    body = {
        "profile": "customer_support",
        "version": "1.0",
        "entrypoint": "data.controlplane.decision",
        "rules": "policy.rego",
        "requires_warrant": [requirement],
        "weighted_error": WEIGHTS,
    }
    body.update(overrides)
    return body


def bundle(**overrides) -> PolicyBundle:
    return PolicyBundle.parse(manifest(**overrides), RULES)


def warrant(
    *,
    operating_point: str = "P-conservative",
    threshold: float = 0.9,
    target_flag_rate: float = 0.05,
    recall: float = 0.30,
    n_test: int = 2000,
    calibration_status: CalibrationStatus = CalibrationStatus.CALIBRATED,
    **kwargs,
):
    """A warrant sized so the calibration power check passes by default.

    ``n_test`` defaults well above what a 25% sensitivity on a 5% budget needs,
    so tests that are not about power do not accidentally depend on it.
    """
    base = make_warrant(detector_id=DETECTOR, eval_set_id=ENVELOPE, **kwargs)
    point = dataclasses.replace(
        make_operating_point(operating_point, DETECTOR, threshold=threshold),
        target_flag_rate=target_flag_rate,
    )
    metrics = dataclasses.replace(
        base.metrics,
        recall=Metric(
            "recall", recall, MetricKind.ESTIMATED, n_test,
            recall - 0.04, recall + 0.05, 0.95, "rate", "bootstrap-percentile-1000",
            convention="two_sided_95",
        ),
    )
    realised = Metric(
        "flag_rate",
        target_flag_rate if calibration_status is CalibrationStatus.CALIBRATED else target_flag_rate * 2,
        MetricKind.ESTIMATED, n_test,
        target_flag_rate * 0.99 if calibration_status is CalibrationStatus.CALIBRATED else target_flag_rate * 1.9,
        target_flag_rate * 1.01 if calibration_status is CalibrationStatus.CALIBRATED else target_flag_rate * 2.1,
        0.95, "rate", "bootstrap-percentile-1000",
            convention="two_sided_95",
        )
    return dataclasses.replace(
        base,
        operating_point=point,
        metrics=metrics,
        n_test=n_test,
        calibration=CalibrationClaim(
            status=calibration_status,
            target_flag_rate=target_flag_rate,
            realised=realised,
            n_to_detect=n_to_detect_deviation(target_flag_rate, 0.25),
            detail="fixture",
        ),
    )


def matrix_of(*warrants, envelopes=None) -> WarrantMatrix:
    now = utc_now()
    cells = [WarrantMatrix._cell_for(w, now) for w in warrants]
    return WarrantMatrix(
        cells,
        detectors=sorted({w.detector_id for w in warrants}) or [DETECTOR],
        envelopes=envelopes or sorted({w.eval_set_id for w in warrants}) or [ENVELOPE],
        now=now,
    )


# --------------------------------------------------------------------------- #
# Bundles are data, and refused rather than partially understood
# --------------------------------------------------------------------------- #


def test_an_unknown_manifest_field_is_refused() -> None:
    """A field the loader ignores is a rule its author believes is in force."""
    with pytest.raises(BundleError, match="unknown manifest field"):
        PolicyBundle.parse(manifest(fail_open=True), RULES)


def test_a_misspelt_minimum_is_refused_rather_than_dropped() -> None:
    with pytest.raises(BundleError, match="unknown field"):
        bundle(requirement={"min_recal": 0.9})


def test_all_three_parts_of_the_key_are_required() -> None:
    """Recall at one threshold says nothing about recall at another."""
    body = manifest()
    del body["requires_warrant"][0]["operating_point"]
    with pytest.raises(BundleError, match="operating_point"):
        PolicyBundle.parse(body, RULES)


def test_the_fpr_ceiling_must_be_declared_even_when_it_is_absent() -> None:
    """"No ceiling here" is a claim and has to be written down.

    A defaulted 1.0 and an explicit null are the same arithmetic and different
    statements; only one of them is auditable.
    """
    body = manifest()
    del body["requires_warrant"][0]["max_fpr_hard_negatives"]
    with pytest.raises(BundleError, match="max_fpr_hard_negatives"):
        PolicyBundle.parse(body, RULES)


def test_a_bundle_relying_on_no_warrant_is_refused() -> None:
    """A policy resting on nothing is the state this loader exists to prevent."""
    with pytest.raises(BundleError, match="declares no requires_warrant"):
        PolicyBundle.parse(manifest(requires_warrant=[]), RULES)


@pytest.mark.parametrize(
    "text,expected",
    [("24h", timedelta(hours=24)), ("90m", timedelta(minutes=90)), ("7d", timedelta(days=7))],
)
def test_durations_parse_with_their_unit(text, expected) -> None:
    assert parse_duration(text) == expected


def test_a_bare_number_is_not_a_duration() -> None:
    """The wrong assumed unit fails open by a factor of 60 or 3600."""
    with pytest.raises(BundleError, match="explicit unit"):
        parse_duration("24")


def test_the_content_hash_covers_the_rules_and_not_only_the_manifest() -> None:
    """Otherwise the rules change under a fixed version every certificate cites."""
    same = PolicyBundle.parse(manifest(), RULES)
    changed_rules = PolicyBundle.parse(manifest(), RULES + "\n# a changed rule\n")
    changed_manifest = PolicyBundle.parse(manifest(version="1.1"), RULES)
    assert same.content_hash == PolicyBundle.parse(manifest(), RULES).content_hash
    assert changed_rules.content_hash != same.content_hash
    assert changed_manifest.content_hash != same.content_hash


def test_reformatting_the_manifest_does_not_move_the_hash() -> None:
    """Key order is canonicalised, so a tidy-up does not invalidate history."""
    body = manifest()
    reordered = dict(reversed(list(body.items())))
    assert (
        PolicyBundle.parse(body, RULES).content_hash
        == PolicyBundle.parse(reordered, RULES).content_hash
    )


# --------------------------------------------------------------------------- #
# Load-time warrant resolution — SPEC.md §7.2
# --------------------------------------------------------------------------- #


def test_a_bundle_naming_an_unwarranted_point_fails_to_load() -> None:
    """The Phase 7 gate, first half. The error must name the missing warrant."""
    empty = WarrantMatrix([], detectors=[DETECTOR], envelopes=[ENVELOPE])
    with pytest.raises(WarrantResolutionError) as caught:
        resolve_bundle(bundle(), empty)
    message = str(caught.value)
    assert f"{DETECTOR}|P-conservative|{ENVELOPE}" in message
    assert "UNVALIDATED" in message


def test_refused_and_unvalidated_do_not_share_a_message() -> None:
    """``CLAUDE.md`` invariant 2. Different facts, different operational owners."""
    refused = warrant(
        controls=failing_controls("canary"),
        status=WarrantStatus.REFUSED,
        status_reason="canary failed",
    )
    with pytest.raises(WarrantResolutionError) as caught:
        resolve_bundle(bundle(), matrix_of(refused))
    assert "REFUSED" in str(caught.value)
    assert "no warrant has ever been filed" not in str(caught.value)


def test_the_recall_floor_is_checked_against_the_lower_bound() -> None:
    """A minimum is a guarantee; 0.26 with a lower bound of 0.18 is not one."""
    marginal = warrant(recall=0.12)  # ci_low 0.08, below the profile's 0.10
    with pytest.raises(WarrantResolutionError, match="lower bound"):
        resolve_bundle(bundle(), matrix_of(marginal))
    assert resolve_bundle(bundle(), matrix_of(warrant(recall=0.30))).resolved


def test_an_expired_warrant_does_not_load() -> None:
    old = warrant(issued_at=utc_now() - timedelta(hours=48), ttl_hours=24)
    with pytest.raises(WarrantResolutionError, match="max_age|expired"):
        resolve_bundle(bundle(), matrix_of(old))


def test_a_declared_fpr_ceiling_with_no_measurement_behind_it_is_refused() -> None:
    """An absence is not a pass — the same rule as UNVALIDATED never being VALID.

    The fixture warrants carry no ``fpr_hard_negatives``, because it is measured
    on ``hard-negatives-200`` and not on this envelope. A profile declaring a
    ceiling anyway would put an unbacked guarantee on every certificate it
    issues.
    """
    declared = bundle(requirement={"max_fpr_hard_negatives": 0.02})
    with pytest.raises(WarrantResolutionError, match="no hard-negative FPR measurement"):
        resolve_bundle(declared, matrix_of(warrant()))


def test_declaring_no_ceiling_loads_and_says_so_on_the_record() -> None:
    resolved = resolve_bundle(bundle(), matrix_of(warrant()))
    assert any("no hard-negative FPR ceiling declared" in n for n in resolved.notes())


# -- carry-over 1: the calibration claim gets a policy consequence ----------- #


def test_a_profile_that_refuses_drift_does_not_load_against_a_drifted_budget() -> None:
    """The ranking claim holds and the bundle still fails. That is the point.

    A tier sized on a flag rate cannot absorb a budget that is demonstrably not
    the budget any more, even though the detector ranks exactly as well as it
    did.
    """
    drifted = warrant(calibration_status=CalibrationStatus.DRIFTED)
    assert drifted.status is WarrantStatus.VALID, "ranking is unaffected"
    with pytest.raises(WarrantResolutionError, match="DRIFTED"):
        resolve_bundle(bundle(), matrix_of(drifted))


def test_a_profile_that_widens_quotes_the_measured_rate_not_the_declared_one() -> None:
    widening = bundle(
        requirement={"calibration": {"on_drift": OnCalibrationDrift.WIDEN_BUDGET, "sensitivity": 0.25}}
    )
    drifted = warrant(calibration_status=CalibrationStatus.DRIFTED)
    resolved = resolve_bundle(widening, matrix_of(drifted))
    budget = resolved.primary.claimed_flag_rate_budget
    assert isinstance(budget, dict)
    assert budget["value"] != budget["declared_target"]
    assert any("measured" in n or "DRIFTED" in n for n in resolved.notes())


def test_a_profile_that_ignores_drift_records_that_it_considered_it() -> None:
    """"Considered and does not apply" must be distinguishable from "never asked"."""
    ignoring = bundle(
        requirement={"calibration": {"on_drift": OnCalibrationDrift.IGNORE, "sensitivity": 0.25}}
    )
    resolved = resolve_bundle(ignoring, matrix_of(warrant(calibration_status=CalibrationStatus.DRIFTED)))
    assert any("IGNORE" in n for n in resolved.notes())


def test_calibration_defaults_to_refuse_rather_than_to_ignore() -> None:
    """The whole reason the claim was split in two is that one half was being
    carried by the other; inheriting "ignore it" would put it straight back."""
    body = manifest()
    del body["requires_warrant"][0]["calibration"]
    parsed = PolicyBundle.parse(body, RULES)
    assert parsed.requires_warrant[0].calibration.on_drift == OnCalibrationDrift.REFUSE


# -- carry-over 2: declared sensitivity against achievable power ------------- #


def test_a_sensitivity_the_sample_cannot_support_fails_to_load() -> None:
    """The n>=1441 boundary, as a load-time refusal.

    Separating a 25% deviation from a 5% budget needs n >= 1441. A profile
    declaring 10% sensitivity there needs far more. This is the same mechanism
    as the recall floor, applied to a claim the sample size cannot back — and
    no rule can fix it, only more test items.
    """
    assert n_to_detect_deviation(0.05, 0.25) == 1441
    strict = bundle(
        requirement={"calibration": {"on_drift": "REFUSE", "sensitivity": 0.10}}
    )
    with pytest.raises(WarrantResolutionError) as caught:
        resolve_bundle(strict, matrix_of(warrant(n_test=600, target_flag_rate=0.05)))
    message = str(caught.value)
    assert "n >= " in message and "n = 600" in message
    assert "more test items" in message


def test_the_same_sensitivity_loads_once_the_sample_is_large_enough() -> None:
    """The refusal is about n, not about the profile being unreasonable."""
    needed = n_to_detect_deviation(0.05, 0.25)
    ok = warrant(n_test=needed, target_flag_rate=0.05)
    assert resolve_bundle(bundle(), matrix_of(ok)).resolved


def test_an_operating_point_declaring_no_budget_has_nothing_to_be_sensitive_to() -> None:
    """A profile may decline to make a budget claim; that is not a failure."""
    no_budget = warrant(n_test=10)
    no_budget = dataclasses.replace(
        no_budget,
        operating_point=dataclasses.replace(no_budget.operating_point, target_flag_rate=None),
        calibration=None,
    )
    assert resolve_bundle(bundle(), matrix_of(no_budget)).resolved


def test_every_failure_is_collected_before_raising() -> None:
    """Fixing one missing warrant per deploy is the workflow that gets the
    check disabled."""
    strict = bundle(
        requirement={
            "min_recall": 0.99,
            "max_fpr_hard_negatives": 0.01,
            "calibration": {"on_drift": "REFUSE", "sensitivity": 0.05},
        }
    )
    with pytest.raises(WarrantResolutionError) as caught:
        resolve_bundle(strict, matrix_of(warrant(n_test=600, recall=0.30)))
    assert len(caught.value.failures) >= 3


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


def engine_for(**kwargs):
    return build_engine(resolve_bundle(bundle(**kwargs), matrix_of(warrant())))


def test_the_default_branch_fires_when_no_rule_matches() -> None:
    decision = engine_for().decide(
        {
            "detector": {"score": 0.0},
            "finding": {"severity": int(Severity.INFO), "category": "HALLUCINATION"},
            "action": {"reversibility": 0},
        }
    )
    assert decision.action is Action.ALLOW
    assert decision.rule_id == "R-default-allow"


def test_a_fired_detector_escalates_and_names_its_rule() -> None:
    decision = engine_for().decide(
        {
            "detector": {"score": 1.0},
            "finding": {"severity": int(Severity.HIGH), "category": "HALLUCINATION"},
            "action": {"reversibility": 0},
        }
    )
    assert decision.action is Action.ESCALATE
    assert decision.rule_id == "R-escalate-on-flag"
    assert decision.reason


@pytest.mark.parametrize("key", RESERVED_KEYS)
def test_a_request_cannot_supply_the_bundles_own_facts(key) -> None:
    """A request able to set warrant.weakest_status can assert its way past the
    one rule every other guarantee rests on."""
    with pytest.raises(BundleError, match="injects from the resolved bundle"):
        engine_for().decide({key: {"weakest_status": "VALID"}, "detector": {"score": 1.0}})


def test_a_json_string_payload_is_refused() -> None:
    """The binding accepts one, sets input to a string, and every rule then
    falls through to its default — a silent permissive decision."""
    with pytest.raises(BundleError, match="mapping, not a JSON string"):
        engine_for().decide('{"detector": {"score": 1.0}}')


def test_an_entrypoint_the_module_does_not_define_fails_at_construction() -> None:
    """A deploy error, not a crash under traffic."""
    with pytest.raises(BundleError, match="entrypoint"):
        build_engine(
            resolve_bundle(bundle(entrypoint="data.controlplane.nonexistent"), matrix_of(warrant()))
        )


def test_a_module_that_does_not_parse_fails_at_construction() -> None:
    resolved = resolve_bundle(bundle(), matrix_of(warrant()))
    broken = dataclasses.replace(resolved.bundle, rego_source="package p\nthis is not rego {{{")
    with pytest.raises(BundleError, match="does not parse"):
        build_engine(dataclasses.replace(resolved, bundle=broken))


def test_an_action_outside_the_enum_is_refused() -> None:
    """A policy is refused rather than partially understood."""
    resolved = resolve_bundle(bundle(), matrix_of(warrant()))
    rogue = dataclasses.replace(
        resolved.bundle,
        rego_source=RULES.replace('"action": "ALLOW"', '"action": "SHRUG"'),
    )
    with pytest.raises(BundleError, match="not one of"):
        build_engine(dataclasses.replace(resolved, bundle=rogue))


# --------------------------------------------------------------------------- #
# The weighted-error objective — SPEC.md §7.4
# --------------------------------------------------------------------------- #


def test_the_threshold_may_not_be_selected_on_test() -> None:
    """``CLAUDE.md`` invariant 9, and the first thing a reviewer checks."""
    with pytest.raises(BundleError, match="invariant 9"):
        choose_threshold(
            benign=[0.1, 0.2], positive=[0.8, 0.9], weights=WEIGHTS, selected_on="test"
        )


def test_the_weights_move_the_chosen_threshold() -> None:
    """If they did not, declaring them would be decoration."""
    benign = [0.1, 0.2, 0.3, 0.4, 0.5]
    positive = [0.45, 0.55, 0.65, 0.75, 0.85]
    fp_averse = choose_threshold(
        benign=benign, positive=positive, selected_on="validation",
        weights={"w_fpr_benign": 100.0, "w_fnr": 1.0, "w_fpr_hard_negative": 2.0},
    )
    fn_averse = choose_threshold(
        benign=benign, positive=positive, selected_on="validation",
        weights={"w_fpr_benign": 1.0, "w_fnr": 100.0, "w_fpr_hard_negative": 2.0},
    )
    assert fp_averse.threshold > fn_averse.threshold


def test_an_unmeasured_hard_negative_term_is_excluded_not_scored_as_zero() -> None:
    """Counting an unmeasured rate as zero error flatters every threshold
    equally and quietly reweights the two terms that were measured."""
    rates = error_rates_at(0.5, benign=[0.1, 0.9], positive=[0.4, 0.6])
    assert rates.n_hard_negative == 0
    # With the third term dropped the denominator is 55, not 57.
    expected = (50 * rates.fpr_benign + 5 * rates.fnr) / 55.0
    assert weighted_error(rates, WEIGHTS) == pytest.approx(expected)


def test_every_weight_must_be_declared_including_a_zero_one() -> None:
    """An absent weight and a zero weight are the same arithmetic and different
    intentions."""
    with pytest.raises(BundleError, match="missing"):
        weighted_error(
            error_rates_at(0.5, benign=[0.1], positive=[0.9]),
            {"w_fpr_benign": 50.0, "w_fnr": 5.0},
        )


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def _profile_bundle(name: str, operating_point: str, min_recall: float, on_drift: str, rules: str):
    body = manifest(
        profile=name,
        requirement={
            "operating_point": operating_point,
            "min_recall": min_recall,
            "calibration": {"on_drift": on_drift, "sensitivity": 0.25},
        },
    )
    return PolicyBundle.parse(body, rules)


def test_the_phase_7_gate() -> None:
    """``TASKS.md`` Phase 7.

    A bundle referencing an unwarranted operating point fails to load with an
    error naming the missing warrant; and three profiles produce three different
    actions on one input, at three points on **one measured curve** — same
    detector, same envelope, three thresholds.

    The score is derived from the thresholds rather than written down: the
    midpoint of the two highest is by construction below the most conservative
    profile's threshold and at or above the others. A hardcoded score would be a
    number tuned until three actions appeared.
    """
    # -- half one: the refusal names the warrant --------------------------- #
    with pytest.raises(WarrantResolutionError) as caught:
        resolve_bundle(bundle(), WarrantMatrix([], detectors=[DETECTOR], envelopes=[ENVELOPE]))
    assert f"{DETECTOR}|P-conservative|{ENVELOPE}" in str(caught.value)

    # -- half two: three points on one curve -------------------------------- #
    points = [
        ("customer_support", "P-cs", 0.9045, 0.10, 0.17),
        ("internal_knowledge", "P-ik", 0.8169, 0.20, 0.33),
        ("decision_support", "P-ds", 0.4673, 0.50, 0.73),
    ]
    warrants = [
        warrant(operating_point=op, threshold=tau, target_flag_rate=f, recall=r)
        for _, op, tau, f, r in points
    ]
    assert len({w.detector_id for w in warrants}) == 1, "one detector"
    assert len({w.eval_set_id for w in warrants}) == 1, "one envelope"
    matrix = matrix_of(*warrants)

    postures = {
        "customer_support": RULES,  # escalates only above its own threshold
        "internal_knowledge": RULES.replace("ESCALATE", "REDACT").replace(
            "R-escalate-on-flag", "R-verify-before-answer"
        ),
        "decision_support": RULES,
    }
    resolved = [
        resolve_bundle(
            _profile_bundle(name, op, min_recall=0.05, on_drift="IGNORE", rules=postures[name]),
            matrix,
        )
        for name, op, _, _, _ in points
    ]

    thresholds = sorted((r.primary.warrant.operating_point.threshold for r in resolved), reverse=True)
    score = (thresholds[0] + thresholds[1]) / 2.0

    request = {
        "detector": {"score": score},
        "finding": {"severity": int(Severity.HIGH), "category": "HALLUCINATION"},
        "action": {"reversibility": 0},
    }
    decisions = {r.bundle.name: build_engine(r).decide(request) for r in resolved}

    assert decisions["customer_support"].action is Action.ALLOW
    assert decisions["internal_knowledge"].action is Action.REDACT
    assert decisions["decision_support"].action is Action.ESCALATE
    assert len({d.action for d in decisions.values()}) == 3

    # Every decision is traceable and stamped with the rules that made it.
    for name, decision in decisions.items():
        assert decision.rule_id and decision.reason
        assert decision.policy_version.startswith(name)
        assert decision.policy_hash.startswith("sha256:")
    assert len({d.policy_hash for d in decisions.values()}) == 3


def test_the_shipped_bundles_load_and_agree_with_config(tmp_path) -> None:
    """The repo's own ``policies/`` parse, and their minima match ``config.yaml``.

    Two sources for one number is how they drift. The bundles are authoritative
    for policy and ``config.profiles`` is what routing sees; this asserts they
    have not come apart.
    """
    from controlplane.config import load_config

    config = load_config(str(Path(__file__).resolve().parents[1] / "config.yaml"))
    policies = Path(__file__).resolve().parents[1] / "policies"

    for name, profile in config.profiles.items():
        loaded = PolicyBundle.load(policies / name)
        requirement = loaded.requires_warrant[0]
        assert loaded.name == name
        assert requirement.operating_point == profile.operating_point
        assert requirement.min_recall == pytest.approx(profile.min_recall)
        assert requirement.calibration.on_drift == profile.on_calibration_drift
        assert requirement.calibration.sensitivity == pytest.approx(
            profile.calibration_sensitivity
        )
        assert loaded.weighted_error["w_fnr"] == pytest.approx(
            config.policy.weighted_error.w_fnr
        )


def test_all_three_shipped_bundles_name_one_detector_and_one_envelope() -> None:
    """"Three points on one measured curve" is checkable, not asserted."""
    policies = Path(__file__).resolve().parents[1] / "policies"
    loaded = [PolicyBundle.load(p) for p in sorted(policies.iterdir()) if p.is_dir()]
    assert len(loaded) == 3
    assert len({b.requires_warrant[0].detector for b in loaded}) == 1
    assert len({b.requires_warrant[0].envelope for b in loaded}) == 1
    assert len({b.requires_warrant[0].operating_point for b in loaded}) == 3
