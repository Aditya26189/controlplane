"""Composing two warranted detectors. ``DECISIONS.md`` 088.

One test per case in 088, named for the case, so a reader can check the code
against the rule rather than inferring the rule from the code. The rules were
committed at `543899e`, before `controlplane/policy/compose.py` existed.
"""

from __future__ import annotations

import dataclasses

import pytest

from controlplane.model import Action, Category, WarrantStatus
from controlplane.policy.compose import RESTRICTIVENESS, ComposedDecision, DetectorVerdict, compose

from .factories import failing_controls, make_finding, make_warrant

PROBE = "probe-qwen2.5-7b-instruct-T1-last_token"
PII = "pii-reference"


def verdict(
    detector_id: str,
    *,
    status: WarrantStatus = WarrantStatus.VALID,
    fired: bool = False,
    action: Action = Action.ESCALATE,
    category: Category = Category.HALLUCINATION,
) -> DetectorVerdict:
    if status is WarrantStatus.VALID:
        warrant = make_warrant(detector_id=detector_id, eval_set_id="envelope-a")
    elif status is WarrantStatus.REFUSED:
        warrant = make_warrant(
            detector_id=detector_id, eval_set_id="envelope-a",
            controls=failing_controls("canary"), status=WarrantStatus.REFUSED,
            status_reason="canary failed",
        )
    else:
        warrant = None
    return DetectorVerdict(
        detector_id=detector_id,
        status=status,
        fired=fired,
        action=action,
        warrant=warrant,
        finding=(
            make_finding(f"F-{detector_id}", detector_id=detector_id, category=category)
            if fired
            else None
        ),
    )


# --------------------------------------------------------------------------- #
# The design position: actions compose, bounds do not
# --------------------------------------------------------------------------- #


def test_bounds_are_keyed_by_detector_and_never_merged() -> None:
    """There is no arithmetic turning two detectors' bounds into a joint bound.

    Their errors are not independent — both read the same text — so multiplying
    them would manufacture a number nobody measured. A reader must be able to
    see what each detector was worth and must not be able to read off what the
    pair was worth.
    """
    decision = compose(
        [
            verdict(PROBE, fired=True, action=Action.ESCALATE),
            verdict(PII, fired=True, action=Action.REDACT, category=Category.PII),
        ]
    )
    assert set(decision.claimed_bounds) == {PROBE, PII}
    for detector_id, bounds in decision.claimed_bounds.items():
        assert bounds["detector_id"] == detector_id
        assert "recall" in bounds and "eval_set_id" in bounds
    # No joint key of any kind.
    assert not any(
        key in decision.claimed_bounds for key in ("joint", "combined", "overall")
    )


def test_the_restrictiveness_ladder_covers_every_action() -> None:
    """A new Action with no place on the ladder would make case 1 arbitrary."""
    assert set(RESTRICTIVENESS) == set(Action)
    assert len(set(RESTRICTIVENESS.values())) == len(Action)


# --------------------------------------------------------------------------- #
# Case 1 — both VALID, both flag
# --------------------------------------------------------------------------- #


def test_case_1_both_flag_takes_the_more_restrictive_action() -> None:
    decision = compose(
        [
            verdict(PROBE, fired=True, action=Action.ESCALATE),
            verdict(PII, fired=True, action=Action.REDACT, category=Category.PII),
        ]
    )
    assert decision.action is Action.ESCALATE
    assert decision.rule == "all-valid-agree"
    assert len(decision.warrants_relied_upon) == 2
    assert len(decision.findings) == 2


def test_case_1_is_not_a_vote() -> None:
    """Two detectors looking for different things agreeing that something is
    wrong is two true statements, not two opinions on one question. The
    composed action must not be softened because they chose differently."""
    decision = compose(
        [
            verdict(PROBE, fired=True, action=Action.BLOCK),
            verdict(PII, fired=True, action=Action.ALLOW, category=Category.PII),
        ]
    )
    assert decision.action is Action.BLOCK


# --------------------------------------------------------------------------- #
# Case 2 — both VALID, they disagree
# --------------------------------------------------------------------------- #


def test_case_2_disagreement_takes_the_flagging_detectors_action() -> None:
    """A PII detector not firing on a hallucination is it working correctly.

    Treating that silence as a dissenting vote would let a correct silence
    cancel a correct finding.
    """
    decision = compose(
        [
            verdict(PROBE, fired=True, action=Action.ESCALATE),
            verdict(PII, fired=False, category=Category.PII),
        ]
    )
    assert decision.action is Action.ESCALATE
    assert decision.rule == "valid-disagree"
    assert PII in decision.reason
    # Both bounds are still quoted: the silent detector was validated here and
    # its claim is part of what this certificate can say.
    assert set(decision.claimed_bounds) == {PROBE, PII}


def test_case_2_with_nothing_firing_allows_and_still_cites_both() -> None:
    decision = compose([verdict(PROBE), verdict(PII, category=Category.PII)])
    assert decision.action is Action.ALLOW
    assert decision.rule == "none-fired"
    assert len(decision.warrants_relied_upon) == 2


# --------------------------------------------------------------------------- #
# Case 3 — one VALID, one REFUSED
# --------------------------------------------------------------------------- #


def test_case_3_refusal_is_not_inherited() -> None:
    """A refused detector is out of service; it is not a veto.

    Letting it block would take a working detector out of service because an
    unrelated one failed its controls.
    """
    decision = compose(
        [
            verdict(PROBE, fired=True, action=Action.ESCALATE),
            verdict(PII, status=WarrantStatus.REFUSED, category=Category.PII),
        ]
    )
    assert decision.action is Action.ESCALATE
    assert decision.weakest_status is WarrantStatus.VALID
    assert decision.warrants_relied_upon == (
        make_warrant(detector_id=PROBE, eval_set_id="envelope-a").warrant_id,
    )
    assert set(decision.claimed_bounds) == {PROBE}


def test_case_3_records_what_was_not_checked() -> None:
    """'We checked for PII' and 'our PII detector is out of service' produce
    identical-looking ALLOWs otherwise."""
    decision = compose(
        [
            verdict(PROBE),
            verdict(PII, status=WarrantStatus.REFUSED, category=Category.PII),
        ]
    )
    assert any(PII in note and "REFUSED" in note for note in decision.unchecked)


def test_case_3_does_not_enqueue_a_refused_detector() -> None:
    """Enqueuing is how UNVALIDATED cells get measured. A REFUSED cell was
    already measured and failed; it needs a human, not a queue."""
    decision = compose(
        [
            verdict(PROBE, fired=True),
            verdict(PII, status=WarrantStatus.REFUSED, fired=True, category=Category.PII),
        ]
    )
    assert PII not in decision.enqueue_for_validation


# --------------------------------------------------------------------------- #
# Case 4 — one VALID, one UNVALIDATED
# --------------------------------------------------------------------------- #


def test_case_4_an_unvalidated_detector_that_fires_triggers_the_default() -> None:
    """``CLAUDE.md`` invariant 2, at the composition layer.

    UNVALIDATED output is of unknown quality, which is neither 'known wrong'
    (REFUSED, ignore it) nor 'known good' (VALID, trust it).
    """
    decision = compose(
        [
            verdict(PROBE, fired=False),
            verdict(PII, status=WarrantStatus.UNVALIDATED, fired=True, category=Category.PII),
        ],
        conservative_default=Action.ESCALATE,
    )
    assert decision.action is Action.ESCALATE
    assert decision.rule == "valid-plus-unvalidated-fired"
    assert PII in decision.enqueue_for_validation
    # It contributed a finding but no bound.
    assert set(decision.claimed_bounds) == {PROBE}
    assert len(decision.findings) == 1


def test_case_4_is_distinct_from_case_3() -> None:
    """The distinction the whole product argues for, asserted as behaviour.

    Same input, same firing detector, differing only in whether it was measured
    here and failed or never measured at all. The two must not produce the same
    decision.
    """
    common = dict(fired=True, category=Category.PII)
    refused = compose(
        [verdict(PROBE), verdict(PII, status=WarrantStatus.REFUSED, **common)],
        conservative_default=Action.ESCALATE,
    )
    unvalidated = compose(
        [verdict(PROBE), verdict(PII, status=WarrantStatus.UNVALIDATED, **common)],
        conservative_default=Action.ESCALATE,
    )
    assert refused.action is Action.ALLOW
    assert unvalidated.action is Action.ESCALATE
    assert refused.rule != unvalidated.rule


def test_case_4_silent_unvalidated_detector_does_not_trigger_the_default() -> None:
    """Only a *firing* unvalidated detector is information. A silent one is
    nothing, and escalating on it would escalate every request."""
    decision = compose(
        [
            verdict(PROBE, fired=True, action=Action.REDACT),
            verdict(PII, status=WarrantStatus.UNVALIDATED, fired=False, category=Category.PII),
        ]
    )
    assert decision.action is Action.REDACT
    # Not "valid-disagree": the silent detector holds no valid warrant here, so
    # it is not one of the detectors being agreed or disagreed with. Among those
    # actually relied upon, all of them fired.
    assert decision.rule == "all-valid-agree"
    assert PII not in decision.enqueue_for_validation
    assert any(PII in note and "UNVALIDATED" in note for note in decision.unchecked)


# --------------------------------------------------------------------------- #
# The overriding rule
# --------------------------------------------------------------------------- #


def test_two_unvalidated_detectors_do_not_add_up_to_one_validated_one() -> None:
    decision = compose(
        [
            verdict(PROBE, status=WarrantStatus.UNVALIDATED, fired=True),
            verdict(PII, status=WarrantStatus.UNVALIDATED, fired=True, category=Category.PII),
        ],
        conservative_default=Action.ESCALATE,
    )
    assert decision.action is Action.ESCALATE
    assert decision.rule == "no-valid-warrant"
    assert decision.claimed_bounds == {}
    assert decision.warrants_relied_upon == ()
    assert decision.weakest_status is WarrantStatus.UNVALIDATED
    assert set(decision.enqueue_for_validation) == {PROBE, PII}


def test_findings_survive_even_when_no_warrant_backs_them() -> None:
    """An unwarranted finding is still information; it just licenses no claim
    about how often it is right."""
    decision = compose(
        [verdict(PROBE, status=WarrantStatus.UNVALIDATED, fired=True)],
        conservative_default=Action.ESCALATE,
    )
    assert len(decision.findings) == 1
    assert decision.claimed_bounds == {}


def test_composing_nothing_is_a_caller_error_not_a_conservative_outcome() -> None:
    with pytest.raises(ValueError, match="at least one detector"):
        compose([])
