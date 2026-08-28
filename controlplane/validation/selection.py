"""Threshold-selection uncertainty in a warranted recall. ``DECISIONS.md`` 083.

Every recall interval this repo has published is **conditional on the threshold
being correct**. The threshold is chosen on validation to hit a flag-rate
budget, then frozen and applied to test, and the bootstrap that produces the
interval resamples test only. Selection noise is not in it.

That omission is not uniform across operating points, and ``DECISIONS.md`` 082
is what makes it quantifiable. The threshold's position on the ROC is set by the
handful of validation items sitting above it — at ``P-customer-support``,
**five negatives out of 255**. Resampling validation moves that position, and
the local slope decides what the move costs: at slope 6.70 a small shift in
false-positive rate is a large shift in recall, while at 0.85 it is not.

So the interval that is most conditional on a threshold nobody can place
precisely belongs to the profile with the highest traffic and the tightest
budget. This module propagates that noise into the bound.

## What is resampled and what is not

Validation is resampled and the threshold **reselected on each draw**; test is
resampled independently; recall is recomputed at the reselected threshold. The
probe itself is held fixed.

Holding the fit fixed is deliberate and it bounds the claim: what comes out is
recall uncertainty given *this* probe, including where its threshold lands. A
fully nested version would refit on a resampled training split and would be a
larger statement — and a much more expensive one. The narrower quantity is the
one the warrant needs, because the warrant is issued for a specific fitted
probe.

## Why this widens rather than shifts

Reselecting the threshold is unbiased for the budget, so the point estimate
moves little. What changes is the interval, and the widening factor is the
number worth reading: it says how much of the previously-reported precision was
an artefact of treating a selected quantity as fixed.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import numpy as np

from .stats import threshold_for_flag_rate

__all__ = ["SelectionAwareBound", "selection_aware_recall"]

_LOG = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SelectionAwareBound:
    """A recall bound with and without threshold-selection noise in it.

    Args:
        operating_point_id: Which point.
        target_flag_rate: The budget the threshold was selected to hit.
        threshold: The threshold actually issued, from the run artifact.
        recall: Point estimate on test at that threshold.
        conditional_ci: Interval from resampling **test only** — what the
            warrant has been reporting. Conditional on the threshold.
        selection_aware_ci: Interval from resampling validation, reselecting the
            threshold, and resampling test. Carries selection noise.
        threshold_ci: Where the reselected threshold landed across draws. A wide
            interval here is the mechanism behind a wide recall interval.
        realised_flag_rate: Measured on test at the issued threshold. Invariant
            6: the measured rate, never the target, is what downstream
            calculations use.
        realised_fpr: Measured false-positive rate on test. **Not a declared
            budget** — the budget is a flag rate — but it is what fixes the
            point's position on the ROC, and therefore what the slope acts on.
        n_negatives_above_validation: How many validation negatives sat above
            the issued threshold. The count that positions the point on the FPR
            axis, and the reason the widening is uneven.
        n_validation: Validation items.
        n_test: Test items.
        n_bootstrap: Draws.
    """

    operating_point_id: str
    target_flag_rate: float
    threshold: float
    recall: float
    conditional_ci: tuple[float, float]
    selection_aware_ci: tuple[float, float]
    threshold_ci: tuple[float, float]
    realised_flag_rate: float
    realised_fpr: float
    n_negatives_above_validation: int
    n_validation: int
    n_test: int
    n_bootstrap: int

    @property
    def conditional_width(self) -> float:
        return self.conditional_ci[1] - self.conditional_ci[0]

    @property
    def selection_aware_width(self) -> float:
        return self.selection_aware_ci[1] - self.selection_aware_ci[0]

    @property
    def widening(self) -> float:
        """How much wider the honest interval is.

        The headline number: how much of the reported precision was an artefact
        of treating a selected threshold as a fixed one.
        """
        if self.conditional_width <= 0:
            return float("nan")
        return self.selection_aware_width / self.conditional_width

    def to_payload(self) -> dict:
        return {
            "operating_point_id": self.operating_point_id,
            "target_flag_rate": self.target_flag_rate,
            "threshold": self.threshold,
            "recall": self.recall,
            "conditional_ci": list(self.conditional_ci),
            "selection_aware_ci": list(self.selection_aware_ci),
            "conditional_width": self.conditional_width,
            "selection_aware_width": self.selection_aware_width,
            "widening": self.widening,
            "threshold_ci": list(self.threshold_ci),
            "realised_flag_rate": self.realised_flag_rate,
            "realised_fpr": self.realised_fpr,
            "n_negatives_above_validation": self.n_negatives_above_validation,
            "n_validation": self.n_validation,
            "n_test": self.n_test,
            "n_bootstrap": self.n_bootstrap,
        }


def selection_aware_recall(
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    *,
    operating_point_id: str,
    target_flag_rate: float,
    threshold: float,
    n_bootstrap: int,
    seed: int,
    ci_level: float = 0.95,
) -> SelectionAwareBound:
    """Propagate threshold-selection noise into a recall bound.

    Args:
        validation_scores: Scores on the split the threshold was chosen on.
        validation_labels: 0/1, 1 meaning *incorrect*.
        test_scores: Scores on the held-out split.
        test_labels: Its labels.
        operating_point_id: For the record.
        target_flag_rate: The budget the threshold aims at.
        threshold: The threshold that was actually issued, read from the run
            artifact rather than reselected here — the bound must describe the
            warrant that exists, not one this function would have chosen.
        n_bootstrap: Draws.
        seed: Reproducibility.
        ci_level: Interval level.

    Returns:
        A :class:`SelectionAwareBound`.

    Raises:
        ValueError: If either split lacks positives, in which case no recall is
            defined and no interval can be built.
    """
    validation_scores = np.asarray(validation_scores, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)
    validation_labels = np.asarray(validation_labels)
    test_labels = np.asarray(test_labels)

    if (test_labels == 1).sum() == 0 or (validation_labels == 1).sum() == 0:
        raise ValueError(
            f"{operating_point_id}: recall needs positives in both splits; got "
            f"{(validation_labels == 1).sum()} on validation and "
            f"{(test_labels == 1).sum()} on test"
        )

    rng = np.random.default_rng(seed)
    n_val, n_test = len(validation_scores), len(test_scores)

    conditional = np.empty(n_bootstrap)
    selection_aware = np.empty(n_bootstrap)
    thresholds = np.empty(n_bootstrap)

    for draw in range(n_bootstrap):
        # Test-only resample: the interval the warrant has been reporting.
        test_index = rng.integers(0, n_test, n_test)
        y = test_labels[test_index]
        s = test_scores[test_index]
        positives = y == 1
        conditional[draw] = (
            float((s[positives] >= threshold).mean()) if positives.any() else np.nan
        )

        # Reselect the threshold on a resampled validation split, then apply it
        # to an independently resampled test split. Independently, because
        # validation and test are disjoint samples and coupling their draws
        # would understate the combined variance.
        validation_index = rng.integers(0, n_val, n_val)
        reselected = threshold_for_flag_rate(
            validation_scores[validation_index], target_flag_rate
        )
        thresholds[draw] = reselected
        selection_aware[draw] = (
            float((s[positives] >= reselected).mean()) if positives.any() else np.nan
        )

    tail = (1.0 - ci_level) / 2.0

    def interval(values: np.ndarray) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        return float(np.quantile(finite, tail)), float(np.quantile(finite, 1.0 - tail))

    fires_test = test_scores >= threshold
    fires_validation = validation_scores >= threshold

    bound = SelectionAwareBound(
        operating_point_id=operating_point_id,
        target_flag_rate=float(target_flag_rate),
        threshold=float(threshold),
        recall=float(fires_test[test_labels == 1].mean()),
        conditional_ci=interval(conditional),
        selection_aware_ci=interval(selection_aware),
        threshold_ci=interval(thresholds),
        realised_flag_rate=float(fires_test.mean()),
        realised_fpr=float(fires_test[test_labels == 0].mean()),
        n_negatives_above_validation=int(fires_validation[validation_labels == 0].sum()),
        n_validation=n_val,
        n_test=n_test,
        n_bootstrap=n_bootstrap,
    )
    _LOG.info(
        "%s: recall %.4f, conditional %.4f wide, selection-aware %.4f wide (%.2fx); "
        "threshold positioned by %d validation negatives",
        bound.operating_point_id, bound.recall, bound.conditional_width,
        bound.selection_aware_width, bound.widening,
        bound.n_negatives_above_validation,
    )
    return bound
