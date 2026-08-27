"""Model-version invalidation. ``SPEC.md`` §5.4, ``DECISIONS.md`` 074.

The second way a warrant stops being true, and the one with no measurement
behind it. These tests pin the asymmetry that makes it different from drift: a
model change is *told*, not observed, and the response is unconditional.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.drift.model_version import (
    ModelVersionInvalidation,
    invalidate_for_model_change,
    pins_to_model,
)
from src.model import AccessTier, WarrantStatus

from .factories import failing_controls, make_warrant

OLD = "Qwen/Qwen2.5-7B-Instruct"
NEW = "Qwen/Qwen3-8B-Instruct"


def _at_tier(tier: AccessTier, **kwargs):
    return dataclasses.replace(make_warrant(**kwargs), access_tier=tier)


def _pin(mapping: dict):
    return lambda w: mapping.get(w.warrant_id)


def _only(results: tuple[ModelVersionInvalidation, ...]) -> ModelVersionInvalidation:
    assert len(results) == 1
    return results[0]


def test_an_activation_warrant_on_a_changed_model_is_revoked() -> None:
    """A probe's weights against a different model are an unrelated function."""
    warrant = _at_tier(AccessTier.T1_ACTIVATIONS)
    result = _only(
        invalidate_for_model_change(
            [warrant], live_model=NEW, pinned_model_of=_pin({warrant.warrant_id: OLD})
        )
    )
    assert result.invalidated
    assert result.to_status is WarrantStatus.REVOKED
    assert OLD in result.reason and NEW in result.reason


def test_an_activation_warrant_on_the_same_model_survives() -> None:
    warrant = _at_tier(AccessTier.T1_ACTIVATIONS)
    result = _only(
        invalidate_for_model_change(
            [warrant], live_model=OLD, pinned_model_of=_pin({warrant.warrant_id: OLD})
        )
    )
    assert not result.invalidated
    assert result.to_status is warrant.status


@pytest.mark.parametrize("tier", [AccessTier.T2_LOGPROBS, AccessTier.T3_TEXT])
def test_model_agnostic_tiers_survive_but_are_not_endorsed(tier) -> None:
    """``DECISIONS.md`` 074: the detector still runs; its numbers are unmeasured.

    Survival must be visible in the output — returning only the invalidations
    would make "untouched" indistinguishable from "still correct".
    """
    warrant = _at_tier(tier)
    result = _only(
        invalidate_for_model_change(
            [warrant], live_model=NEW, pinned_model_of=_pin({warrant.warrant_id: OLD})
        )
    )
    assert not result.invalidated
    assert "not claimed to transfer" in result.reason


def test_an_unrecorded_pin_is_not_a_matching_pin() -> None:
    """Unknown must not default to "keep serving".

    Otherwise the one warrant class this module exists to protect is the one it
    silently skips.
    """
    warrant = _at_tier(AccessTier.T1_ACTIVATIONS)
    result = _only(
        invalidate_for_model_change([warrant], live_model=NEW, pinned_model_of=_pin({}))
    )
    assert result.invalidated
    assert result.pinned_model is None
    assert "no recorded model pin" in result.reason


def test_a_refused_warrant_stays_refused_and_is_not_recorded_as_invalidated() -> None:
    """Otherwise a later revalidation reads as though the refusal was a model problem."""
    warrant = _at_tier(
        AccessTier.T1_ACTIVATIONS,
        controls=failing_controls("canary"),
        status=WarrantStatus.REFUSED,
        status_reason="canary failed",
    )
    result = _only(
        invalidate_for_model_change(
            [warrant], live_model=NEW, pinned_model_of=_pin({warrant.warrant_id: OLD})
        )
    )
    assert result.to_status is WarrantStatus.REFUSED
    assert not result.invalidated


def test_the_tier_decides_and_not_the_detector_name() -> None:
    """A rename that silently switched off invalidation is a failure nobody sees."""
    renamed = _at_tier(AccessTier.T1_ACTIVATIONS, detector_id="entirely-unrelated-name")
    assert pins_to_model(renamed)
    assert not pins_to_model(_at_tier(AccessTier.T2_LOGPROBS))


def test_a_model_change_suspends_the_whole_activation_row() -> None:
    """The operational cost SPEC.md §5.4 asks to be stated, measured here."""
    t1 = [
        _at_tier(AccessTier.T1_ACTIVATIONS, detector_id="probe-a", eval_set_id="env-1"),
        _at_tier(AccessTier.T1_ACTIVATIONS, detector_id="probe-b", eval_set_id="env-2"),
    ]
    others = [
        _at_tier(AccessTier.T2_LOGPROBS, detector_id="logprob", eval_set_id="env-1"),
        _at_tier(AccessTier.T3_TEXT, detector_id="judge", eval_set_id="env-1"),
    ]
    pins = {w.warrant_id: OLD for w in t1 + others}

    results = invalidate_for_model_change(
        t1 + others, live_model=NEW, pinned_model_of=_pin(pins)
    )

    assert len(results) == 4, "every warrant is reported, touched or not"
    invalidated = [r for r in results if r.invalidated]
    assert len(invalidated) == 2
    assert {r.access_tier for r in invalidated} == {AccessTier.T1_ACTIVATIONS}
