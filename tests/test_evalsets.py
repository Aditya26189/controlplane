"""Evaluation sets: construction, freezing, and the claims they can support.

The gate for Phase 3 is that every set is content-hashed and registered, that
``/validate`` runs against each, and that FPR on ``hard-negatives-200`` is
measured. These tests assert all three, plus the properties that make the sets
worth having: no LLM generated them, the identifiers are synthetic and
checksum-correct, and a set edited after freezing is caught.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import numpy as np
import pytest

from src.config import Config
from src.detectors.pii_reference import ReferencePiiDetector
from src.evalsets import (
    build_canary_pii,
    build_hard_negatives,
    build_hinglish_pii,
    build_longctx,
    load_evalset,
    save_evalset,
    verhoeff_check_digit,
    verhoeff_is_valid,
    verify_manifest,
    write_manifest,
)
from src.evalsets.hard_negatives import FRAMINGS, HARD_NEGATIVES
from src.evalsets.hinglish import NEAR_MISS_NEGATIVES, PII_SCENARIOS
from src.evalsets.identifiers import DISCLOSURE_FORMS, GENERATORS, aadhaar
from src.model import MetricError, WarrantStatus
from src.validation.evalsets import EvalSetError
from src.validation.stats import clopper_pearson
from src.validation.text_runner import validate_text_detector


@pytest.fixture(scope="module")
def hinglish():
    return build_hinglish_pii(seed=1729)


@pytest.fixture(scope="module")
def hard_negatives():
    return build_hard_negatives(seed=1729)


@pytest.fixture(scope="module")
def canary():
    return build_canary_pii(seed=1729)


# --------------------------------------------------------------------------- #
# Identifiers
# --------------------------------------------------------------------------- #


def test_verhoeff_accepts_valid_and_rejects_perturbations() -> None:
    """Aadhaar uses Verhoeff, not Luhn, and a wrong checksum must not pass."""
    rng = random.Random(0)
    for _ in range(50):
        body = "9999" + "".join(str(rng.randint(0, 9)) for _ in range(7))
        number = body + verhoeff_check_digit(body)
        assert verhoeff_is_valid(number)
        # Any single-digit change must break it. This is the property Luhn does
        # not have for all transpositions, and the reason Aadhaar uses Verhoeff.
        position = rng.randrange(len(number))
        digit = number[position]
        replacement = str((int(digit) + rng.randint(1, 9)) % 10)
        perturbed = number[:position] + replacement + number[position + 1 :]
        assert not verhoeff_is_valid(perturbed), perturbed


def test_verhoeff_catches_adjacent_transpositions() -> None:
    """The failure mode Verhoeff exists for and a simple mod-10 sum misses."""
    rng = random.Random(1)
    caught = 0
    trials = 0
    for _ in range(50):
        body = "9999" + "".join(str(rng.randint(0, 9)) for _ in range(7))
        number = body + verhoeff_check_digit(body)
        for position in range(len(number) - 1):
            if number[position] == number[position + 1]:
                continue
            swapped = (
                number[:position]
                + number[position + 1]
                + number[position]
                + number[position + 2 :]
            )
            trials += 1
            caught += not verhoeff_is_valid(swapped)
    assert trials > 100
    assert caught == trials, f"{trials - caught} transpositions slipped through"


def test_all_generated_identifiers_are_synthetic() -> None:
    """Nothing here can collide with a real person's identifier."""
    rng = random.Random(7)
    for form in DISCLOSURE_FORMS:
        assert aadhaar(rng, form).canonical.startswith("9999")


def test_deliberately_invalid_aadhaar_fails_its_checksum() -> None:
    """So the checksum-validating configuration can be shown to reject it."""
    rng = random.Random(3)
    bad = aadhaar(rng, "verbatim", valid=False)
    assert not verhoeff_is_valid(bad.canonical)
    assert bad.checksum_valid is False


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_no_set_was_generated_by_a_model(hinglish, hard_negatives, canary) -> None:
    """``DECISIONS.md`` 007: a model's judgment cannot be the ground truth."""
    for evalset in (hinglish, hard_negatives, canary):
        assert evalset.construction["llm_generated"] is False


def test_hinglish_is_balanced_and_covers_every_disclosure_form(hinglish) -> None:
    assert len(hinglish) == 200
    assert 0.4 <= hinglish.base_rate <= 0.6, hinglish.base_rate
    counts = hinglish.construction["per_form_counts"]
    assert set(counts) == set(DISCLOSURE_FORMS)
    assert min(counts.values()) >= 30, counts


def test_hinglish_covers_code_switching(hinglish) -> None:
    """An English-only recogniser is blind to exactly this axis."""
    scripts = {i.meta.get("script") for i in hinglish.items if i.label == 1}
    assert {"latin", "mixed", "devanagari"} <= scripts
    devanagari = [
        i for i in hinglish.items
        if i.label == 1 and i.meta.get("script") in ("mixed", "devanagari")
    ]
    assert len(devanagari) >= 10
    assert any(any("ऀ" <= c <= "ॿ" for c in i.prompt) for i in devanagari)


def test_hinglish_positives_carry_a_span(hinglish) -> None:
    """Evidence spans drive explainability, so they must actually locate the text."""
    for item in hinglish.items:
        if item.label != 1:
            continue
        start, end = item.meta["span"]
        assert start >= 0 and end > start
        assert item.prompt[start:end]


def test_hinglish_near_misses_are_not_identifiers(hinglish) -> None:
    """The negatives are numeric decoys, and none is tagged as PII."""
    negatives = [i for i in hinglish.items if i.label == 0]
    assert len(negatives) >= 90
    for item in negatives:
        assert item.meta["synthetic_pii"] is False
        assert "decoy_kind" in item.meta


def test_hard_negatives_are_single_class_and_cover_five_domains(hard_negatives) -> None:
    assert len(hard_negatives) == 200
    assert hard_negatives.labels.sum() == 0
    domains = hard_negatives.construction["domains"]
    assert set(domains) == {"security", "clinical", "hr", "grievance", "legal"}
    assert min(domains.values()) >= 20, domains


def test_hard_negatives_use_every_framing(hard_negatives) -> None:
    """The same content in four wrappers is what a real queue looks like."""
    framings = {i.meta["framing"] for i in hard_negatives.items}
    assert framings == {name for name, _ in FRAMINGS}


def test_hard_negative_scenarios_are_distinct() -> None:
    """Hand-written means distinct, not paraphrased."""
    contents = [n.content for n in HARD_NEGATIVES]
    assert len(set(contents)) == len(contents)
    assert len(HARD_NEGATIVES) >= 40


def test_hinglish_scenarios_are_distinct() -> None:
    templates = [s.template for s in PII_SCENARIOS]
    assert len(set(templates)) == len(templates)
    assert len(PII_SCENARIOS) >= 40
    assert len(NEAR_MISS_NEGATIVES) >= 20


def test_canary_is_deliberately_easy(canary) -> None:
    """A canary a detector can plausibly miss is a tripwire that fires on noise."""
    assert len(canary) == 20
    assert canary.labels.sum() == len(canary)
    assert all(i.meta["disclosure_form"] == "verbatim" for i in canary.items)
    assert all(i.meta["checksum_valid"] for i in canary.items)


# --------------------------------------------------------------------------- #
# Long context
# --------------------------------------------------------------------------- #


def test_longctx_differs_only_in_length(hinglish) -> None:
    """Same questions, same labels, different envelope."""
    long_set = build_longctx(hinglish, seed=1729, pad_tokens=(4000, 16000))
    assert len(long_set) == len(hinglish)
    assert (long_set.labels == hinglish.labels).all()
    assert list(long_set.question_ids) == list(hinglish.question_ids)
    assert long_set.envelope_id != hinglish.envelope_id

    mean_base = np.mean([len(i.prompt) for i in hinglish.items])
    mean_long = np.mean([len(i.prompt) for i in long_set.items])
    assert mean_long > 100 * mean_base

    # The question goes last: a detector reading early positions must not
    # succeed for a reason unrelated to long context.
    for base_item, long_item in zip(hinglish.items, long_set.items):
        assert long_item.prompt.endswith(base_item.prompt)


def test_longctx_inherits_no_warrant(hinglish) -> None:
    """Invariant 1: long-context traffic is a different input distribution."""
    long_set = build_longctx(hinglish, seed=1729, pad_tokens=(4000, 16000))
    assert long_set.content_hash != hinglish.content_hash
    assert long_set.construction["base_content_hash"] == hinglish.content_hash


# --------------------------------------------------------------------------- #
# Freezing and the registry
# --------------------------------------------------------------------------- #


def test_sets_are_reproducible(hinglish, hard_negatives, canary) -> None:
    """Rebuild and re-hash rather than trust."""
    assert build_hinglish_pii(seed=1729).content_hash == hinglish.content_hash
    assert build_hard_negatives(seed=1729).content_hash == hard_negatives.content_hash
    assert build_canary_pii(seed=1729).content_hash == canary.content_hash


def test_round_trip_through_disk(hinglish, tmp_path: Path) -> None:
    path = save_evalset(hinglish, tmp_path)
    restored = load_evalset(path)
    assert restored.content_hash == hinglish.content_hash
    assert restored.items == hinglish.items


def test_an_edited_set_is_caught_on_read(hinglish, tmp_path: Path) -> None:
    """Invariant 9: a changed set is a different set and inherits nothing."""
    import json

    path = save_evalset(hinglish, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["items"][0]["prompt"] = data["items"][0]["prompt"] + " (edited)"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(EvalSetError, match="edited after it was frozen"):
        load_evalset(path)


def test_manifest_registers_and_verifies(
    hinglish, hard_negatives, canary, tmp_path: Path
) -> None:
    for evalset in (hinglish, hard_negatives, canary):
        save_evalset(evalset, tmp_path)
    write_manifest([hinglish, hard_negatives, canary], tmp_path)
    assert verify_manifest(tmp_path) == []

    # Drift is reported, not tolerated.
    (tmp_path / f"{canary.eval_set_id}.json").unlink()
    problems = verify_manifest(tmp_path)
    assert any("missing" in p for p in problems)


# --------------------------------------------------------------------------- #
# Measurement — the Phase 3 gate
# --------------------------------------------------------------------------- #


def test_validate_runs_against_every_set(
    hinglish, hard_negatives, canary, config: Config
) -> None:
    """The gate: ``/validate`` runs against each set and issues a warrant."""
    detector = ReferencePiiDetector()
    for evalset, is_hard in ((hinglish, False), (hard_negatives, True)):
        run = validate_text_detector(
            config, evalset, detector,
            threshold=detector.min_confidence,
            canary_evalset=canary,
            max_fpr_hard_negatives=0.02 if is_hard else None,
            is_hard_negative_set=is_hard,
        )
        assert run.warrant.status is WarrantStatus.VALID, run.warrant.status_reason
        assert run.warrant.envelope.envelope_id == evalset.envelope_id
        assert run.test_scored == 1
        assert len(run.controls) == 5


def test_fpr_on_hard_negatives_is_measured_with_an_interval(
    hard_negatives, canary, config: Config
) -> None:
    """The number a skeptic accepts, and it carries bounds."""
    detector = ReferencePiiDetector()
    run = validate_text_detector(
        config, hard_negatives, detector,
        threshold=detector.min_confidence,
        canary_evalset=canary,
        max_fpr_hard_negatives=0.02,
        is_hard_negative_set=True,
    )
    fpr = run.metrics.fpr_hard_negatives
    assert fpr is not None
    assert fpr.has_interval
    assert fpr.ci_high > fpr.ci_low or fpr.value == 0.0
    # Zero events must not produce a zero-width interval (DECISIONS.md 035).
    if fpr.value == 0.0:
        assert fpr.ci_high > 0.0
        assert "Clopper-Pearson" in fpr.estimator


def test_single_class_envelope_claims_fpr_only(
    hard_negatives, canary, config: Config
) -> None:
    """``DECISIONS.md`` 032."""
    detector = ReferencePiiDetector()
    run = validate_text_detector(
        config, hard_negatives, detector,
        threshold=detector.min_confidence,
        canary_evalset=canary,
        max_fpr_hard_negatives=0.02,
        is_hard_negative_set=True,
    )
    assert run.metrics.auroc is None
    assert run.metrics.recall is None
    assert run.metrics.precision is None
    with pytest.raises(MetricError, match="recall is undefined"):
        _ = run.metrics.lift


def test_ranking_metrics_are_all_or_nothing(hinglish, canary, config: Config) -> None:
    """Invariant 5 survives making them optional: they go together."""
    detector = ReferencePiiDetector()
    run = validate_text_detector(
        config, hinglish, detector,
        threshold=detector.min_confidence, canary_evalset=canary,
    )
    with pytest.raises(MetricError, match="present together or absent together"):
        dataclasses.replace(run.metrics, recall=None)


def test_single_class_envelope_without_a_declared_maximum_is_refused(
    hard_negatives, canary, config: Config
) -> None:
    """Without the FPR bar there is no bar at all, which is worse than refusing."""
    detector = ReferencePiiDetector()
    run = validate_text_detector(
        config, hard_negatives, detector,
        threshold=detector.min_confidence,
        canary_evalset=canary,
        is_hard_negative_set=True,
    )
    assert run.warrant.status is WarrantStatus.REFUSED
    assert "single_class_envelope" in run.warrant.status_reason


def test_recall_degrades_on_non_verbatim_disclosure(hinglish) -> None:
    """The finding the set exists to produce, on our own detector.

    Even a purpose-built recogniser loses substantial recall once identifiers
    stop being pasted cleanly. That is the structure behind Presidio's published
    number, and reproducing it on our own floor is what makes the Phase 8
    comparison fair rather than flattering.
    """
    detector = ReferencePiiDetector()
    scores = detector.score([i.prompt for i in hinglish.items])
    by_form: dict[str, float] = {}
    for form in DISCLOSURE_FORMS:
        rows = [
            k for k, i in enumerate(hinglish.items)
            if i.label == 1 and i.meta.get("disclosure_form") == form
        ]
        by_form[form] = float(np.mean(scores[rows] >= detector.min_confidence))
    assert by_form["verbatim"] > 0.95
    assert by_form["spaced"] < by_form["verbatim"]
    assert by_form["obfuscated"] < by_form["verbatim"]


def test_reference_detector_is_length_invariant(hinglish) -> None:
    """``DECISIONS.md`` 037: the control case for Beat 4.

    A stateless pattern matcher has no pooling step to dilute, so context length
    is invisible to it. That makes a probe's long-context collapse a property of
    *pooling* rather than of long inputs in general.
    """
    detector = ReferencePiiDetector()
    long_set = build_longctx(hinglish, seed=1729, pad_tokens=(4000, 16000))
    short_scores = detector.score([i.prompt for i in hinglish.items])
    long_scores = detector.score([i.prompt for i in long_set.items])
    assert np.array_equal(short_scores, long_scores)


# --------------------------------------------------------------------------- #
# Boundary statistics
# --------------------------------------------------------------------------- #


def test_clopper_pearson_matches_the_rule_of_three() -> None:
    """Zero events in n trials gives an upper bound near 3/n."""
    for n in (100, 200, 1000):
        low, high = clopper_pearson(0, n, 0.95)
        assert low == 0.0
        assert 2.5 / n < high < 3.9 / n, (n, high)
    low, high = clopper_pearson(200, 200, 0.95)
    assert high == 1.0 and low < 1.0


def test_clopper_pearson_brackets_the_point_estimate() -> None:
    for successes, n in ((1, 200), (17, 200), (100, 200)):
        low, high = clopper_pearson(successes, n, 0.95)
        assert low <= successes / n <= high


# --------------------------------------------------------------------------- #
# Applicability fencing
# --------------------------------------------------------------------------- #


def test_inapplicable_controls_are_fenced(hinglish, canary, config: Config) -> None:
    """``DECISIONS.md`` 034: the one escape hatch, fenced three ways."""
    from src.model import ControlResult, WarrantError

    detector = ReferencePiiDetector()
    run = validate_text_detector(
        config, hinglish, detector,
        threshold=detector.min_confidence, canary_evalset=canary,
    )
    inapplicable = [c for c in run.controls if not c.applicable]
    assert {c.control for c in inapplicable} == {
        "padding_fault", "label_shuffle", "null_feature"
    }
    # Each states why the mechanism cannot exist.
    for control in inapplicable:
        assert len(control.detail) > 60
        assert control.margin == 0.0
    assert run.warrant.claimed_bounds()["controls_run"] == 2

    # An inapplicable control cannot carry a verdict or a margin.
    with pytest.raises(WarrantError, match="carries no verdict and no margin"):
        ControlResult("canary", False, 0.0, "recall == 1.0", -1.0, "x", applicable=False)
    with pytest.raises(WarrantError, match="carries no verdict and no margin"):
        ControlResult("canary", True, 1.0, "recall == 1.0", 0.5, "x", applicable=False)
    # Nor exist without a reason.
    with pytest.raises(WarrantError, match="must state why"):
        ControlResult("canary", True, 0.0, "recall == 1.0", 0.0, "", applicable=False)


def test_an_applicable_failed_control_still_refuses(
    hinglish, config: Config
) -> None:
    """Narrowing invariant 3 must not have holed it: no canary set, no warrant."""
    detector = ReferencePiiDetector()
    run = validate_text_detector(
        config, hinglish, detector,
        threshold=detector.min_confidence, canary_evalset=None,
    )
    assert run.warrant.status is WarrantStatus.REFUSED
    assert "canary" in run.warrant.status_reason
