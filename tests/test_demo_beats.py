"""The six beats assemble from committed artifacts, or say why they cannot.

A demo that silently skips a beat shows only what happened to work. These
tests exist so that a missing artifact on submission day is a failing test
rather than a gap discovered on camera.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from controlplane.report.beats import assemble_beats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEATS = assemble_beats(PROJECT_ROOT)


def test_all_six_beats_assemble() -> None:
    """Every beat has its artifact. A missing one names itself."""
    missing = [f"beat {b.number}: {b.missing}" for b in BEATS if b.missing]
    assert not missing, "beats could not be assembled:\n  " + "\n  ".join(missing)
    assert len(BEATS) == 6


@pytest.mark.parametrize("beat", BEATS, ids=lambda b: f"beat{b.number}")
def test_every_beat_names_what_it_answers_and_where_to_check(beat) -> None:
    """A beat without an artifact path is a claim without a receipt."""
    assert beat.answers, f"beat {beat.number} does not say what it answers"
    assert beat.rows, f"beat {beat.number} renders no rows"
    assert beat.artifacts, f"beat {beat.number} cites no artifact"


def test_the_refusal_beat_carries_the_uninflated_numbers() -> None:
    """199 and 271, cluster-uncorrected, with that stated.

    DECISIONS 113: the inflated 868 was built on a design effect imported from
    a planning document. The beat must not quietly reacquire one.
    """
    from controlplane.config import load_config
    from controlplane.validation.warrant_stats import min_n_for

    config = load_config(str(PROJECT_ROOT / "config.yaml"))
    budget = float(config.profiles["customer_support"].max_fpr_hard_negatives)
    n_profiles = len(config.profiles)

    beat = next(b for b in BEATS if b.number == 1)
    values = {label: value for label, value in beat.rows}
    # Recomputed from config, not pinned to a literal. A literal here is how
    # the beat came to claim 199/271 off a planning document's 0.015 when this
    # repository declares 0.02 and the answer is 149/203 (DECISIONS 117).
    assert values["clean negatives for ONE profile"] == str(min_n_for(budget, 0.05))
    assert values[f"across {n_profiles} profiles (Bonferroni)"] == str(
        min_n_for(budget, 0.05 / n_profiles)
    )
    assert str(budget) in values[f"customer_support hard-negative FPR ceiling"]
    assert "cluster-uncorrected" in (beat.note or "")
    assert "868" not in str(beat.rows)


def test_the_refusal_beat_does_not_reintroduce_an_imported_budget() -> None:
    """0.015 is not this repository's number, and 199/271 follow from it."""
    beat = next(b for b in BEATS if b.number == 1)
    text = str(beat.rows)
    assert "0.015" not in text, (
        "the refusal beat carries 0.015, which is a planning document's figure; "
        "config.yaml declares 0.02"
    )
    for imported in ("199", "271"):
        assert imported not in text, (
            f"{imported} follows from the imported 0.015 budget, not from config"
        )


def test_the_profile_beat_shows_one_score_and_three_actions() -> None:
    """The whole point: the detector did not change, the policy did."""
    beat = next(b for b in BEATS if b.number == 2)
    text = str(beat.rows)
    for action in ("ALLOW", "REDACT", "ESCALATE"):
        assert action in text, f"{action} missing from the profile beat"
    assert "one input, one score" in text


def test_the_matrix_beat_shows_unvalidated_as_the_modal_state() -> None:
    """Invariant 2. UNVALIDATED must never collapse into VALID or REFUSED, and
    a demo is exactly where the temptation to hide it lives."""
    beat = next(b for b in BEATS if b.number == 5)
    values = {label: value for label, value in beat.rows}
    assert "cells UNVALIDATED" in values
    unvalidated = int(values["cells UNVALIDATED"].split()[0])
    valid = int(values["cells VALID"].split()[0])
    assert unvalidated > valid, (
        "UNVALIDATED is no longer the modal state; either the matrix grew "
        "coverage or the beat stopped reporting it honestly"
    )


def test_every_beat_with_a_known_gap_states_it() -> None:
    """The beats that rest on a declared limitation must say so on screen."""
    notes = {b.number: (b.note or "") for b in BEATS}
    assert "DECISIONS 104" in notes[4], "the pilot's prompt-side privacy scope is unstated"
    assert "session correlation" in notes[5], "the i.i.d. limitation is unstated"
