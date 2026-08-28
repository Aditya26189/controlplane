"""The paired comparison and the ROC geometry. ``DECISIONS.md`` 081, 082.

These exist because ``080`` published a claim that no comparison had tested. The
tests are therefore weighted towards the two ways this analysis goes wrong while
producing plausible output: comparing two independent intervals instead of the
paired difference, and reading a CI that contains zero as "no difference".
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from controlplane.validation.evalsets import TEST, TRAIN, VALIDATION, EvalItem, EvalSet
from controlplane.validation.paired import (
    PairedDifference,
    paired_bootstrap,
    split_relationship,
)
from controlplane.validation.roc import roc_curve


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from controlplane.validation.paired import _auroc as impl

    return impl(scores, labels)


def make_set(assignment, eval_set_id="s") -> EvalSet:
    """A set whose split for question ``q`` is ``assignment(q)``."""
    items = tuple(
        EvalItem(
            item_id=f"i-{q:04d}",
            question_id=f"q-{q:04d}",
            prompt=f"p{q}",
            response=f"r{q}",
            label=q % 2,
            split=assignment(q),
        )
        for q in range(1000)
    )
    return EvalSet(eval_set_id=eval_set_id, items=items, construction={"method": "fixture"})


# --------------------------------------------------------------------------- #
# B.1 — the split relationship is computed, never assumed
# --------------------------------------------------------------------------- #


def test_a_promotion_is_recognised_and_pairs_the_whole_old_test_split() -> None:
    """Items moved only into test, nothing reshuffled."""
    old = make_set(lambda q: TRAIN if q < 500 else VALIDATION if q < 750 else TEST)
    new = make_set(
        lambda q: TRAIN if q < 400 else VALIDATION if q < 600 else TEST, "s2"
    )
    rel = split_relationship(old, new)
    assert rel.is_promotion
    assert rel.new_train_within_old_train and rel.new_test_contains_old_test
    assert rel.test_intersection == 250
    assert len(rel.paired_item_ids) == 250
    assert rel.usable


def test_a_reshuffle_is_recognised_and_pairs_only_the_intersection() -> None:
    """The case the instruction warns about: the paired set shrinks, and the
    containment checks are what reveal it."""
    old = make_set(lambda q: TRAIN if q < 500 else VALIDATION if q < 750 else TEST)
    # Reversed assignment: the new train set is drawn from the old test end.
    new = make_set(
        lambda q: TEST if q < 400 else VALIDATION if q < 600 else TRAIN, "s2"
    )
    rel = split_relationship(old, new)
    assert not rel.is_promotion
    assert not rel.new_train_within_old_train
    assert len(rel.paired_item_ids) < rel.test_intersection or rel.test_intersection == 0


def test_the_paired_set_never_contains_a_training_item() -> None:
    """Both leakage fields must be empty, and they are reported rather than
    asserted so a reader can see the check ran."""
    old = make_set(lambda q: TRAIN if q < 500 else VALIDATION if q < 750 else TEST)
    new = make_set(
        lambda q: TRAIN if q < 400 else VALIDATION if q < 600 else TEST, "s2"
    )
    rel = split_relationship(old, new)
    assert rel.leaked_from_old_train == ()
    assert rel.leaked_from_new_train == ()
    old_train = {i.item_id for i in old.items if i.split == TRAIN}
    new_train = {i.item_id for i in new.items if i.split == TRAIN}
    assert not (set(rel.paired_item_ids) & (old_train | new_train))


def test_a_paired_set_under_200_items_is_not_usable() -> None:
    """Below this the MDD exceeds anything worth acting on, and the honest move
    is to withdraw rather than publish an underpowered non-result."""
    old = make_set(lambda q: TRAIN if q < 900 else VALIDATION if q < 950 else TEST)
    new = make_set(
        lambda q: TRAIN if q < 880 else VALIDATION if q < 940 else TEST, "s2"
    )
    rel = split_relationship(old, new)
    assert len(rel.paired_item_ids) < 200
    assert not rel.usable


# --------------------------------------------------------------------------- #
# B.3 — the pairing is the point
# --------------------------------------------------------------------------- #


def _correlated_pair(n=600, seed=0, shift=0.05):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.46).astype(int)
    shared = rng.normal(labels * 1.2, 1.0)
    baseline = shared + rng.normal(0, 0.05, n)
    variant = shared + rng.normal(0, 0.05, n) + shift * labels
    return baseline, variant, labels


def test_the_paired_interval_is_much_tighter_than_an_unpaired_one() -> None:
    """The entire reason for pairing, measured rather than asserted.

    Both models score the same items, so item-level difficulty is shared and
    cancels in the difference. Resampling the two models independently does not
    cancel it, and the resulting interval is wide enough to hide a real effect —
    which is the specific failure that looks like evidence of no difference.
    """
    baseline, variant, labels = _correlated_pair()
    paired = paired_bootstrap(
        baseline, variant, labels,
        quantities={"auroc": (_auroc, _auroc)}, n_bootstrap=400, seed=1729,
    )[0]

    # The unpaired alternative: resample each model independently.
    rng = np.random.default_rng(1729)
    n = len(labels)
    unpaired = []
    for _ in range(400):
        a, b = rng.integers(0, n, n), rng.integers(0, n, n)
        unpaired.append(_auroc(variant[b], labels[b]) - _auroc(baseline[a], labels[a]))
    unpaired_se = float(np.std(unpaired, ddof=1))

    assert paired.standard_error < unpaired_se / 2, (
        f"paired SE {paired.standard_error:.5f} should be far below the unpaired "
        f"{unpaired_se:.5f}; if it is not, the pairing is not being used"
    )


def test_a_real_difference_is_detected_and_the_interval_excludes_zero() -> None:
    baseline, variant, labels = _correlated_pair(shift=1.5)
    result = paired_bootstrap(
        baseline, variant, labels,
        quantities={"auroc": (_auroc, _auroc)}, n_bootstrap=400, seed=1729,
    )[0]
    assert result.difference > 0
    assert result.excludes_zero
    assert "excludes zero" in result.verdict()


def test_no_difference_is_never_reported_without_the_minimum_detectable() -> None:
    """A CI containing zero has two readings and the interval does not
    distinguish them."""
    baseline, variant, labels = _correlated_pair(shift=0.0)
    result = paired_bootstrap(
        baseline, variant, labels,
        quantities={"auroc": (_auroc, _auroc)}, n_bootstrap=400, seed=1729,
    )[0]
    assert not result.excludes_zero
    assert result.minimum_detectable > 0
    # With no floor supplied there is nothing to judge the MDD against, so the
    # only permitted reading is "underpowered".
    assert "UNDERPOWERED" in result.verdict()
    # Given a floor the MDD is comfortably inside, the bounded reading is allowed.
    assert "bounded" in result.verdict(floor=result.minimum_detectable * 2)


def test_the_two_metrics_are_both_recomputed_inside_each_resample() -> None:
    """The own-threshold regime scores each model at its own operating point.

    An earlier version bootstrapped both legs at one threshold and corrected the
    baseline afterwards, producing a difference its own interval did not
    describe. This pins the difference to the two observed values.
    """
    baseline, variant, labels = _correlated_pair()
    low = lambda s, y: float((s[y == 1] >= -0.5).mean())
    high = lambda s, y: float((s[y == 1] >= 0.5).mean())
    result = paired_bootstrap(
        baseline, variant, labels,
        quantities={"recall": (low, high)}, n_bootstrap=200, seed=1729,
    )[0]
    assert result.baseline == pytest.approx(low(baseline, labels))
    assert result.variant == pytest.approx(high(variant, labels))
    assert result.difference == pytest.approx(result.variant - result.baseline)
    assert result.ci_low <= result.difference <= result.ci_high


def test_misaligned_arrays_raise_rather_than_comparing_different_items() -> None:
    baseline, variant, labels = _correlated_pair()
    with pytest.raises(ValueError, match="different items"):
        paired_bootstrap(
            baseline, variant[:-1], labels,
            quantities={"auroc": (_auroc, _auroc)}, n_bootstrap=10, seed=1,
        )


# --------------------------------------------------------------------------- #
# C.2 — ROC geometry
# --------------------------------------------------------------------------- #


def test_a_perfect_separator_has_auroc_one_and_the_positive_class_is_incorrect() -> None:
    """Polarity check. Inverting the label meaning yields 1 - AUROC, which reads
    as a strong negative result and misdirects debugging for hours."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert roc_curve(scores, labels).auroc == pytest.approx(1.0)
    assert roc_curve(scores, 1 - labels).auroc == pytest.approx(0.0)


def test_a_single_class_split_supports_no_curve() -> None:
    with pytest.raises(ValueError, match="both classes"):
        roc_curve(np.array([0.1, 0.2]), np.array([1, 1]))


def test_the_local_slope_is_steep_where_the_curve_is_steep() -> None:
    """The C.2 hypothesis, on data built to have a known shape.

    Positives concentrated at the top of the score range make the curve steep
    near FPR 0 and flat later, so a low-flag-rate point must report a larger
    slope than a high-flag-rate one.
    """
    rng = np.random.default_rng(1729)
    n = 4000
    labels = (rng.random(n) < 0.45).astype(int)
    scores = rng.normal(labels * 1.6, 1.0)
    low = float(np.quantile(scores, 0.90))
    high = float(np.quantile(scores, 0.50))
    curve = roc_curve(scores, labels, operating_points={"low-f": low, "high-f": high})
    by_id = {p.operating_point_id: p for p in curve.points}
    assert by_id["low-f"].slope > by_id["high-f"].slope
    assert by_id["low-f"].flag_rate < by_id["high-f"].flag_rate


def test_the_slope_window_is_reported_so_the_estimate_is_reproducible() -> None:
    """A slope quoted without the interval it was fitted over cannot be checked."""
    rng = np.random.default_rng(7)
    labels = (rng.random(800) < 0.5).astype(int)
    scores = rng.normal(labels, 1.0)
    curve = roc_curve(
        scores, labels, operating_points={"p": float(np.quantile(scores, 0.8))},
        slope_window=0.03,
    )
    point = curve.points[0]
    assert point.window_fpr > 0
    assert point.window_fpr <= 0.06 + 1e-9


def test_the_curve_is_monotone_and_spans_the_unit_square() -> None:
    rng = np.random.default_rng(3)
    labels = (rng.random(500) < 0.4).astype(int)
    scores = rng.normal(labels * 0.8, 1.0)
    curve = roc_curve(scores, labels)
    assert np.all(np.diff(curve.fpr) >= -1e-12)
    assert np.all(np.diff(curve.tpr) >= -1e-12)
    assert curve.fpr[0] == pytest.approx(0.0) and curve.tpr[0] == pytest.approx(0.0)
    assert curve.fpr[-1] == pytest.approx(1.0) and curve.tpr[-1] == pytest.approx(1.0)


def test_the_curve_auroc_agrees_with_the_rank_statistic() -> None:
    """Two independent computations of the same quantity, so a bug in either is
    visible rather than self-consistent."""
    rng = np.random.default_rng(11)
    labels = (rng.random(700) < 0.45).astype(int)
    scores = rng.normal(labels * 1.1, 1.0)
    assert roc_curve(scores, labels).auroc == pytest.approx(_auroc(scores, labels), abs=1e-9)


# --------------------------------------------------------------------------- #
# Threshold-selection uncertainty — DECISIONS.md 083
# --------------------------------------------------------------------------- #


def _split_pair(n_val=480, n_test=960, seed=5, separation=1.2):
    rng = np.random.default_rng(seed)
    def draw(n):
        labels = (rng.random(n) < 0.46).astype(int)
        return rng.normal(labels * separation, 1.0), labels
    return draw(n_val), draw(n_test)


def test_selection_noise_widens_the_interval_it_is_added_to() -> None:
    """The whole point: a threshold chosen from a sample is not a fixed number.

    The conditional interval resamples test only, which is what every warrant in
    this repo has reported. The selection-aware one also resamples validation
    and reselects, and must be wider — if it is not, the reselection is not
    reaching the threshold.
    """
    from controlplane.validation.selection import selection_aware_recall

    (vs, vy), (ts, ty) = _split_pair()
    threshold = float(np.quantile(vs, 0.90))
    bound = selection_aware_recall(
        vs, vy, ts, ty,
        operating_point_id="P-test", target_flag_rate=0.10,
        threshold=threshold, n_bootstrap=400, seed=1729,
    )
    assert bound.selection_aware_width > bound.conditional_width
    assert bound.widening > 1.0
    assert bound.threshold_ci[0] < threshold < bound.threshold_ci[1]


def test_the_widening_is_larger_where_fewer_negatives_position_the_threshold() -> None:
    """``DECISIONS.md`` 082's geometry showing up in interval width.

    A low-flag-rate point is positioned by a handful of validation negatives and
    sits on a steep segment; a high-flag-rate point is positioned by many and
    sits on a flat one. The first must widen more.
    """
    from controlplane.validation.selection import selection_aware_recall

    (vs, vy), (ts, ty) = _split_pair()
    bounds = {}
    for point, budget in (("low", 0.10), ("high", 0.50)):
        bounds[point] = selection_aware_recall(
            vs, vy, ts, ty,
            operating_point_id=point, target_flag_rate=budget,
            threshold=float(np.quantile(vs, 1.0 - budget)),
            n_bootstrap=400, seed=1729,
        )
    assert (
        bounds["low"].n_negatives_above_validation
        < bounds["high"].n_negatives_above_validation
    )
    assert bounds["low"].widening > bounds["high"].widening


def test_the_realised_rates_are_measured_on_test_not_taken_from_the_budget() -> None:
    """Invariant 6. The target is what was aimed at; only the measured rate may
    feed a downstream calculation."""
    from controlplane.validation.selection import selection_aware_recall

    (vs, vy), (ts, ty) = _split_pair()
    bound = selection_aware_recall(
        vs, vy, ts, ty,
        operating_point_id="P-test", target_flag_rate=0.10,
        threshold=float(np.quantile(vs, 0.90)), n_bootstrap=200, seed=1729,
    )
    assert bound.realised_flag_rate == pytest.approx(
        float((ts >= bound.threshold).mean())
    )
    assert bound.realised_fpr == pytest.approx(
        float((ts[ty == 0] >= bound.threshold).mean())
    )
    # The realised rate is free to differ from the budget; that it does is the
    # reason invariant 6 exists.
    assert bound.target_flag_rate == 0.10


def test_a_split_without_positives_supports_no_recall_bound() -> None:
    from controlplane.validation.selection import selection_aware_recall

    (vs, vy), (ts, _) = _split_pair()
    with pytest.raises(ValueError, match="needs positives"):
        selection_aware_recall(
            vs, vy, ts, np.zeros(len(ts), dtype=int),
            operating_point_id="P-test", target_flag_rate=0.10,
            threshold=0.5, n_bootstrap=50, seed=1,
        )


# --------------------------------------------------------------------------- #
# The nesting check is at the point of the decision
# --------------------------------------------------------------------------- #


def test_a_reused_seed_nests_and_a_fresh_one_does_not(caplog) -> None:
    """``DECISIONS.md`` 081's finding, enforced in code rather than discovered.

    Nesting is not a property of re-splitting. It holds only when the call
    reuses the source's seed and ordering, and the difference decides whether a
    paired comparison is possible at all.
    """
    import logging
    from controlplane.evalsets.resplit import resplit_by_question

    source = make_set(lambda q: TRAIN if q < 500 else VALIDATION if q < 750 else TEST)
    # The source's own splits here are positional rather than permuted, so no
    # seed reproduces them; what the test pins is that the check runs and that a
    # non-nesting result is announced rather than silent.
    with caplog.at_level(logging.WARNING):
        resplit_by_question(
            source, eval_set_id="derived", fractions=(0.4, 0.2, 0.4),
            seed=4242, rationale="fixture",
        )
    assert any("does not nest" in r.message for r in caplog.records)


def test_require_nested_refuses_rather_than_warning() -> None:
    """For callers whose next step is a paired comparison, a warning is not
    enough — the comparison would run on whatever intersection survived."""
    from controlplane.evalsets.resplit import resplit_by_question
    from controlplane.validation.evalsets import EvalSetError

    source = make_set(lambda q: TRAIN if q < 500 else VALIDATION if q < 750 else TEST)
    with pytest.raises(EvalSetError, match="does not nest"):
        resplit_by_question(
            source, eval_set_id="derived", fractions=(0.4, 0.2, 0.4),
            seed=4242, rationale="fixture", require_nested=True,
        )


def test_the_nesting_check_does_not_change_the_frozen_identity() -> None:
    """Construction notes are inside the content hash.

    Recording the relationship there would change the identity of every set
    already built, and ``triviaqa-2400-t960`` would stop hashing to the value
    its warrants are keyed on.
    """
    from controlplane.evalsets.resplit import resplit_by_question

    source = make_set(lambda q: TRAIN if q < 500 else VALIDATION if q < 750 else TEST)
    derived = resplit_by_question(
        source, eval_set_id="derived", fractions=(0.4, 0.2, 0.4),
        seed=4242, rationale="fixture",
    )
    assert "nests_within_source" not in derived.construction
    assert "paired_with_source_n" not in derived.construction
