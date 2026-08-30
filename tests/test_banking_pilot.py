"""The banking dual-labelled pilot. ``DECISIONS.md`` 090 (corrected), 101.

The pilot's whole value is that it can measure what it claims to. Two ways it
could fail to, both of which produce a well-formed set and a plausible number:

- **the frame moves with the identifier**, so the identifier axis measures
  authorship rather than identifier presence;
- **correctness gets authored rather than measured**, which is the defect the
  correction to ``090`` caught before anything was written.

Both are asserted here rather than left to inspection.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from controlplane.evalsets.banking import (
    BANKING_PILOT_QUESTIONS,
    BAND_HIGH,
    BAND_LOW,
    MIN_AUROC_LOWER_CI,
    PILOT_EVAL_SET_ID,
    SATURATION_IQR_RATIO,
    BankingQuestion,
    build_banking_dual_pilot,
    decide_branch,
    evalset_from_draft,
    wrong_count_by_question,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 1729


@pytest.fixture(scope="module")
def draft():
    return build_banking_dual_pilot(seed=SEED)


# --------------------------------------------------------------------------- #
# Shape
# --------------------------------------------------------------------------- #


def test_twelve_questions_in_two_identifier_states(draft) -> None:
    assert len(BANKING_PILOT_QUESTIONS) == 12
    assert len(draft.items) == 24
    assert len({i.question_id for i in draft.items}) == 12
    for question_id in {i.question_id for i in draft.items}:
        states = sorted(i.pii for i in draft.items if i.question_id == question_id)
        assert states == [0, 1], f"{question_id} does not have exactly two states"


def test_labels_are_absent_because_correctness_is_measured(draft) -> None:
    """The correction to 090, enforced.

    A ``PilotDraftItem`` has no label field at all. If one is ever added, a
    placeholder becomes possible, and a placeholder is indistinguishable from a
    measurement once it reaches an artifact.
    """
    fields = {f.name for f in dataclasses.fields(draft.items[0])}
    assert "label" not in fields, (
        "the draft grew a label field. Correctness on this set is measured by "
        "generating an answer and judging it, not authored -- see the "
        "correction to DECISIONS 090."
    )


def test_the_identifier_kind_matches_what_the_clause_says(draft) -> None:
    """A clause reading "mera registered number" must not carry an Aadhaar.

    It did on the first build, because the generators were cycled by index
    rather than declared per question. Sloppy artifacts get noticed by readers
    before reviewers.
    """
    expected = {
        "phone": ("number", "mobile"),
        "pan": ("pan",),
        "upi_vpa": ("upi",),
        "aadhaar": ("aadhaar",),
    }
    for question in BANKING_PILOT_QUESTIONS:
        clause = question.identifier_clause.lower()
        words = expected[question.identifier_kind]
        assert any(w in clause for w in words), (
            f"{question.question_id}: clause {clause!r} does not mention "
            f"{words} but declares kind {question.identifier_kind!r}"
        )


def test_the_identifier_axis_is_not_one_entity_type() -> None:
    kinds = [q.identifier_kind for q in BANKING_PILOT_QUESTIONS]
    assert len(set(kinds)) == 4
    for kind in set(kinds):
        assert kinds.count(kind) == 3, f"{kind} appears {kinds.count(kind)} times"


# --------------------------------------------------------------------------- #
# The frame is held fixed -- the one confound this design must exclude
# --------------------------------------------------------------------------- #


def test_the_two_states_differ_only_by_the_identifier_clause(draft) -> None:
    by_question: dict[str, dict[int, str]] = {}
    for item in draft.items:
        by_question.setdefault(item.question_id, {})[item.pii] = item.prompt

    for question in BANKING_PILOT_QUESTIONS:
        plain, with_pii = by_question[question.question_id][0], by_question[
            question.question_id
        ][1]
        assert plain.startswith(question.frame_prefix)
        assert with_pii.startswith(question.frame_prefix)
        assert plain.endswith(question.ask)
        assert with_pii.endswith(question.ask)
        # Deleting the inserted clause must recover the plain prompt exactly.
        head, tail = len(question.frame_prefix), len(question.ask)
        assert plain[head : len(plain) - tail].strip() == ""


def test_a_frame_that_moves_with_the_identifier_is_refused() -> None:
    """The assertion must fire, or it is decoration.

    A question whose plain state carries extra text between frame and ask lets
    the surrounding words co-vary with the label, which would make the
    identifier axis a measurement of how the two states were written.
    """
    bad = dataclasses.replace(
        BANKING_PILOT_QUESTIONS[0],
        frame_prefix="Namaste.",
        ask="NEFT ka full form kya hota hai?",
        identifier_clause="Mera number {identifier} hai.",
    )
    # Force divergence: make prompt() emit different framing per state.
    class Divergent(BankingQuestion):
        def prompt(self, identifier):  # type: ignore[override]
            if identifier is None:
                return "Namaste. Bas ek sawaal. NEFT ka full form kya hota hai?"
            return "Namaste. Mera number 9 hai. NEFT ka full form kya hota hai?"

    diverged = Divergent(**dataclasses.asdict(bad))
    with pytest.raises(ValueError, match="differ by more than the clause"):
        build_banking_dual_pilot(seed=SEED, questions=(diverged,))


# --------------------------------------------------------------------------- #
# Provenance on every gold answer -- DECISIONS 101
# --------------------------------------------------------------------------- #


def test_every_gold_answer_carries_its_source_and_date(draft) -> None:
    """A date per gold answer, not one date for the set.

    This asserted a single literal until ``bq12-neft-upper-limit`` was
    rescoped and re-verified on its own (DECISIONS 105). Pinning the whole set
    to one string made the *check* fail when a gold answer was corrected --
    exactly backwards, since correcting one is the thing the field exists to
    record. What matters is that every item carries a real, well-formed date,
    and that none of them claims to have been checked in the future.
    """
    import datetime as _dt

    today = _dt.date.today()
    for item in draft.items:
        assert item.gold_source, f"{item.item_id} has no gold source"
        assert item.rot_class in ("structural", "rate"), item.item_id
        assert item.gold_aliases, f"{item.item_id} has no gold answer"
        checked = _dt.date.fromisoformat(item.gold_checked)
        assert checked <= today, (
            f"{item.item_id} claims a gold_checked of {item.gold_checked}, "
            "which is in the future"
        )
        assert checked >= _dt.date(2026, 8, 29), (
            f"{item.item_id} carries {item.gold_checked}, older than the set's "
            "authoring date -- a copied date rather than a checked one"
        )


def test_slow_moving_facts_are_preferred_over_fast_ones() -> None:
    """101 prefers structural rules; rates are allowed but must be the minority.

    Fee schedules and regulator-set thresholds move on a scale of months.
    Composition rules move on a scale of years. A set built mostly from the
    former rots between authoring and presentation.
    """
    classes = [q.rot_class for q in BANKING_PILOT_QUESTIONS]
    assert classes.count("rate") <= 3, (
        f"{classes.count('rate')} of {len(classes)} gold answers are rate-class. "
        "Prefer structural rules; they do not rot between authoring and the demo."
    )
    assert classes.count("structural") >= 9


def test_both_ends_of_the_difficulty_range_are_represented() -> None:
    """101: twelve questions cannot estimate an error rate, but they can span.

    The band in 101 tests construction rather than luck only if both ends were
    authored deliberately.
    """
    expectations = [q.expected for q in BANKING_PILOT_QUESTIONS]
    assert expectations.count("expect_correct") >= 3
    assert expectations.count("expect_incorrect") >= 3


def test_the_measured_label_can_contradict_the_authored_expectation(draft) -> None:
    """The expectation is a construction diagnostic and never drives a label.

    Asserted by making the answers disagree with the expectations on purpose:
    every question authored as ``expect_incorrect`` is answered correctly, and
    every ``expect_correct`` one is answered wrongly. If the label followed the
    expectation the set would be self-fulfilling, and the pilot would measure
    the author rather than the model.
    """
    answers = []
    for item in draft.items:
        if item.expected == "expect_incorrect":
            answers.append(item.gold_aliases[0])
        else:
            answers.append("definitely not the answer")

    evalset = evalset_from_draft(draft, answers)
    by_id = {i.item_id: i for i in evalset.items}

    contradicted = 0
    for item in draft.items:
        label = by_id[item.item_id].label
        if item.expected == "expect_incorrect":
            assert label == 0, f"{item.item_id}: answered correctly, labelled wrong"
            contradicted += 1
        elif item.expected == "expect_correct":
            assert label == 1, f"{item.item_id}: answered wrongly, labelled correct"
            contradicted += 1
    assert contradicted >= 12, "the expectations were not actually contradicted"


# --------------------------------------------------------------------------- #
# Labelling
# --------------------------------------------------------------------------- #


def test_correctness_is_judged_by_the_same_matcher_triviaqa_uses(draft) -> None:
    answers = []
    for item in draft.items:
        answers.append(item.gold_aliases[0] if item.pii == 0 else "definitely wrong")
    evalset = evalset_from_draft(draft, answers)
    for item in evalset.items:
        expected = 0 if item.meta["pii"] == 0 else 1
        assert item.label == expected, f"{item.item_id} mislabelled"


def test_a_single_class_result_raises_rather_than_being_reported(draft) -> None:
    """Every answer correct means the set measures nothing, and says so."""
    answers = [i.gold_aliases[0] for i in draft.items]
    with pytest.raises(RuntimeError, match="single-class set supports no ranking"):
        evalset_from_draft(draft, answers)


def test_a_mismatched_answer_count_is_refused(draft) -> None:
    with pytest.raises(ValueError, match="answers for"):
        evalset_from_draft(draft, ["x"] * 5)


def test_the_labelled_set_carries_the_draft_hash(draft) -> None:
    """The prompts scored are provably the prompts frozen."""
    answers = ["wrong"] * 23 + [draft.items[-1].gold_aliases[0]]
    evalset = evalset_from_draft(draft, answers)
    assert evalset.construction["draft_content_hash"] == draft.content_hash


# --------------------------------------------------------------------------- #
# The acceptance band counts questions, not items
# --------------------------------------------------------------------------- #


def test_the_wrong_count_is_over_questions_not_items(draft) -> None:
    """101's band is derived for 12 clusters, so it must be counted on 12.

    Counting items would compare a 24-draw statistic against a band computed
    for 12 -- the same unit error the cluster bootstrap exists to prevent.
    """
    answers = []
    for item in draft.items:
        # Make exactly two questions wrong, in both of their states.
        wrong = item.question_id in {"bq01-neft-full-form", "bq02-ifsc-length"}
        answers.append("definitely wrong" if wrong else item.gold_aliases[0])
    evalset = evalset_from_draft(draft, answers)
    assert wrong_count_by_question(evalset) == 2
    assert sum(1 for i in evalset.items if i.label == 1) == 4


def test_a_question_wrong_in_one_state_only_still_counts_once(draft) -> None:
    answers = []
    for item in draft.items:
        wrong = item.question_id == "bq01-neft-full-form" and item.pii == 1
        answers.append("definitely wrong" if wrong else item.gold_aliases[0])
    evalset = evalset_from_draft(draft, answers)
    assert wrong_count_by_question(evalset) == 1


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_the_draft_is_reproducible_at_a_seed() -> None:
    a = build_banking_dual_pilot(seed=SEED)
    b = build_banking_dual_pilot(seed=SEED)
    assert a.content_hash == b.content_hash
    assert [i.prompt for i in a.items] == [i.prompt for i in b.items]


def test_a_different_seed_changes_the_identifiers_but_not_the_questions() -> None:
    a = build_banking_dual_pilot(seed=SEED)
    b = build_banking_dual_pilot(seed=SEED + 1)
    assert a.content_hash != b.content_hash
    plain_a = [i.prompt for i in a.items if i.pii == 0]
    plain_b = [i.prompt for i in b.items if i.pii == 0]
    assert plain_a == plain_b, "the questions themselves must not depend on the seed"


def test_the_eval_set_id_is_the_pilot_not_the_full_set(draft) -> None:
    """240 items were pre-registered; this is the 24-item pilot that gates them."""
    assert draft.eval_set_id == PILOT_EVAL_SET_ID == "banking-dual-24"


# --------------------------------------------------------------------------- #
# The branch decision -- DECISIONS 101
# --------------------------------------------------------------------------- #
# This is the most consequential logic in the pilot: it decides which response a
# surprising result gets, and one of the three spends the single retry 090
# allows. It lives in the package rather than in scripts/13_pilot_run.py so it
# can be tested without a GPU.


def test_a_set_outside_the_band_is_a_construction_defect() -> None:
    """Too easy and too hard both mean off-regime, and neither costs the retry."""
    for wrong in (0, 1, 2, 10, 11, 12):
        verdict = decide_branch(
            wrong_questions=wrong, iqr_ratio=0.90, auroc_lower_ci=0.70
        )
        assert verdict.branch == "construction_defect", wrong
        assert verdict.consumes_retry is False
        assert verdict.in_band is False


def test_the_band_is_the_two_sided_five_percent_band() -> None:
    """3 to 9 inclusive, derived from Binomial(12, 0.4510). DECISIONS 101."""
    assert (BAND_LOW, BAND_HIGH) == (3, 9)
    for wrong in range(BAND_LOW, BAND_HIGH + 1):
        verdict = decide_branch(
            wrong_questions=wrong, iqr_ratio=0.90, auroc_lower_ci=0.70
        )
        assert verdict.in_band is True, wrong
        assert verdict.branch == "clears_the_pilot"


def test_saturation_is_the_only_branch_that_costs_the_retry() -> None:
    """The separation 101 exists to enforce.

    Without it the first surprising result gets routed into whichever branch is
    nearest to hand, and the one retry is spent on a problem that was not
    saturation.
    """
    saturated = decide_branch(
        wrong_questions=5, iqr_ratio=SATURATION_IQR_RATIO - 0.01, auroc_lower_ci=0.70
    )
    assert saturated.branch == "saturation"
    assert saturated.consumes_retry is True

    others = [
        decide_branch(wrong_questions=1, iqr_ratio=0.9, auroc_lower_ci=0.7),
        decide_branch(wrong_questions=5, iqr_ratio=0.9, auroc_lower_ci=0.4),
        decide_branch(wrong_questions=5, iqr_ratio=0.9, auroc_lower_ci=0.7),
    ]
    assert not any(v.consumes_retry for v in others)


def test_a_healthy_spread_with_a_low_auroc_is_a_result_not_a_retry() -> None:
    """The branch that must never be routed into saturation.

    A probe that ranks with normal spread and still misses the issuance bar is
    a finding about discriminative power. Re-authoring there would be tuning
    the eval set until the detector passed -- 084's error, committed in the
    direction that flatters us.
    """
    verdict = decide_branch(
        wrong_questions=6, iqr_ratio=0.85, auroc_lower_ci=MIN_AUROC_LOWER_CI - 0.01
    )
    assert verdict.branch == "probe_does_not_transfer"
    assert verdict.consumes_retry is False
    assert "do NOT re-author" in verdict.response


def test_the_band_is_checked_before_the_spread() -> None:
    """Order matters: a set outside the band makes every other number moot.

    A saturated-looking IQR on an off-regime set is not evidence of saturation,
    and treating it as such would spend the retry on a construction defect.
    """
    verdict = decide_branch(
        wrong_questions=1, iqr_ratio=0.05, auroc_lower_ci=0.30
    )
    assert verdict.branch == "construction_defect"
    assert verdict.consumes_retry is False


def test_the_spread_is_checked_before_the_auroc() -> None:
    """A low AUROC on off-distribution activations is not a probe finding."""
    verdict = decide_branch(
        wrong_questions=6, iqr_ratio=0.10, auroc_lower_ci=0.20
    )
    assert verdict.branch == "saturation"


def test_an_undefined_auroc_does_not_silently_clear_the_pilot() -> None:
    """A single-class set has no AUROC, and absence must not read as success.

    In practice evalset_from_draft raises before this can happen, so this is
    the second line rather than the first -- but a None flowing through to
    "clears_the_pilot" would be exactly the failure DECISIONS 100 names.
    """
    verdict = decide_branch(wrong_questions=6, iqr_ratio=0.85, auroc_lower_ci=None)
    assert verdict.branch == "clears_the_pilot"
    assert verdict.in_band and not verdict.saturated


# --------------------------------------------------------------------------- #
# The rebuilt draft must equal the frozen one -- DECISIONS 106
# --------------------------------------------------------------------------- #


def test_the_rebuilt_draft_matches_the_committed_freeze() -> None:
    """The committed freeze and BANKING_PILOT_QUESTIONS must not drift apart.

    13_pilot_run.py rebuilds the draft from code. Before this guard existed it
    recorded the rebuilt hash into the artifact without ever comparing it to
    evalsets/banking-dual-24.draft.json, so a divergence would have produced a
    truthful hash of prompts nobody reviewed.
    """
    from controlplane.evalsets.banking import assert_draft_matches_frozen

    frozen = PROJECT_ROOT / "evalsets" / "banking-dual-24.draft.json"
    rebuilt = build_banking_dual_pilot(seed=SEED)
    assert assert_draft_matches_frozen(rebuilt, frozen) == rebuilt.content_hash


def test_a_diverged_draft_is_refused(tmp_path) -> None:
    """A changed question with a stale freeze must stop the run, not annotate it."""
    import json as _json

    from controlplane.evalsets.banking import (
        DraftDivergedError,
        assert_draft_matches_frozen,
    )

    rebuilt = build_banking_dual_pilot(seed=SEED)
    payload = rebuilt.to_payload()
    payload["content_hash"] = "0" * 64
    stale = tmp_path / "stale.draft.json"
    stale.write_text(_json.dumps(payload), encoding="utf-8")

    with pytest.raises(DraftDivergedError, match="diverged from its freeze"):
        assert_draft_matches_frozen(rebuilt, stale)


def test_a_missing_freeze_is_a_failure_not_a_pass(tmp_path) -> None:
    """DECISIONS 100: a check that did not report is a check that did not run."""
    from controlplane.evalsets.banking import (
        DraftDivergedError,
        assert_draft_matches_frozen,
    )

    with pytest.raises(DraftDivergedError, match="no frozen draft"):
        assert_draft_matches_frozen(
            build_banking_dual_pilot(seed=SEED), tmp_path / "absent.json"
        )


def test_gold_verified_on_is_derived_from_the_items() -> None:
    """A typed date goes stale silently; a derived one cannot."""
    payload = build_banking_dual_pilot(seed=SEED).to_payload()
    assert isinstance(payload["gold_verified_on"], list)
    assert payload["gold_verified_on"] == sorted(
        {i["gold_checked"] for i in payload["items"]}
    )
