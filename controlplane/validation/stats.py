"""Measurement: AUROC, recall, precision, flag rate, and bootstrap intervals.

Every function here that produces a reportable number returns a
:class:`~controlplane.model.metrics.Metric`, never a bare float. That is invariant 4
enforced at the point of production rather than at the point of rendering: a
rate that never exists as a naked float cannot be quoted as one.

**Polarity.** The positive class is *incorrect* — label 1 means the model's
answer was wrong and the probe should fire. Inverting it yields ``1 - AUROC``,
which reads as a strong negative result and misdirects debugging for hours
(``CLAUDE.md``, "Silent failures"). :func:`assert_polarity` is called on every
labelled array that enters this module.

**Resampling unit.** The bootstrap resamples *groups*, not rows, whenever a
grouping is supplied. TriviaQA ships several examples per question; resampling
rows would treat correlated items as independent and produce an interval
narrower than the data supports — an error that makes the result look better
and raises nothing.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np

from ..model import Metric, MetricKind

__all__ = [
    "MeasurementError",
    "clopper_pearson",
    "assert_polarity",
    "auroc",
    "bootstrap_interval",
    "confusion_at",
    "estimated",
    "exact_count",
    "false_positive_rate_at",
    "flag_rate_at",
    "precision_at",
    "recall_at",
    "threshold_for_flag_rate",
    "weighted_error",
]

_LOG = logging.getLogger(__name__)

#: Positive class means "the model's answer was incorrect". Stated as a constant
#: so the assertion and the docstrings cannot drift apart.
POSITIVE_CLASS_MEANING = "incorrect"


class MeasurementError(ValueError):
    """Raised when a measurement cannot be made honestly from the data given."""


def assert_polarity(labels: np.ndarray) -> None:
    """Check the label array is binary with 1 meaning *incorrect*.

    Cannot detect an inversion on its own — 0/1 arrays look identical either way
    — so this asserts the representation and ``test_polarity`` asserts the
    *meaning* end to end, by constructing a case where a probe that fires on
    wrong answers must score above 0.5.

    Args:
        labels: Array of 0/1 labels.

    Raises:
        MeasurementError: If the array is not binary, or is degenerate.
    """
    unique = np.unique(labels)
    if not np.all(np.isin(unique, (0, 1))):
        raise MeasurementError(
            f"labels must be binary 0/1 with 1 meaning {POSITIVE_CLASS_MEANING!r}, "
            f"got values {unique.tolist()}"
        )
    if unique.size < 2:
        raise MeasurementError(
            f"labels contain only class {unique.tolist()}; AUROC and recall are "
            "undefined on a single-class sample. Report the base rate and stop "
            "rather than emitting a number that looks like a measurement."
        )


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, computed from ranks.

    Rank-based (Mann–Whitney U) rather than by integrating a stepped curve, so
    ties are handled by mid-ranks instead of by an arbitrary tie-break. A probe
    that emits many identical scores — which a degenerate one does — otherwise
    scores differently depending on sort order.

    Args:
        labels: 0/1 labels, 1 meaning incorrect.
        scores: Detector scores, higher meaning more likely incorrect.

    Returns:
        AUROC in [0, 1].
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    assert_polarity(labels)
    if labels.shape != scores.shape:
        raise MeasurementError(
            f"labels {labels.shape} and scores {scores.shape} have different shapes"
        )
    order = np.argsort(scores, kind="mergesort")
    ranked = np.empty_like(order, dtype=float)
    sorted_scores = scores[order]
    # Mid-ranks for ties.
    i = 0
    rank = np.arange(1, scores.size + 1, dtype=float)
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranked[i : j + 1] = rank[i : j + 1].mean()
        i = j + 1
    ranks = np.empty_like(ranked)
    ranks[order] = ranked
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    rank_sum_pos = float(ranks[labels == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def confusion_at(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, tn, fn)`` at a threshold.

    Fires at ``score >= threshold``, stated because ``>`` and ``>=`` differ by a
    whole flag on a probe with tied scores, and the difference propagates into
    every downstream economic figure.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    flagged = scores >= threshold
    tp = int(np.sum(flagged & (labels == 1)))
    fp = int(np.sum(flagged & (labels == 0)))
    tn = int(np.sum(~flagged & (labels == 0)))
    fn = int(np.sum(~flagged & (labels == 1)))
    return tp, fp, tn, fn


def recall_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """Fraction of incorrect answers the detector fires on."""
    tp, _, _, fn = confusion_at(labels, scores, threshold)
    if tp + fn == 0:
        raise MeasurementError(
            "recall is undefined with no positive examples in the sample"
        )
    return tp / (tp + fn)


def precision_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """Fraction of flags that are real errors.

    Returns 0.0 when nothing was flagged. Not undefined: "we flagged nothing, so
    none of our flags were right" is the operationally correct reading, and
    raising here would make a conservative operating point unreportable.
    """
    tp, fp, _, _ = confusion_at(labels, scores, threshold)
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def flag_rate_at(scores: np.ndarray, threshold: float) -> float:
    """Fraction of all items flagged — the *measured* rate.

    Never the target rate. Every downstream calculation uses this one
    (``CLAUDE.md`` invariant 6); the target is what you aimed at on validation
    and the measured rate is what you got on test, and they differ.
    """
    scores = np.asarray(scores, dtype=float)
    return float(np.mean(scores >= threshold))


def false_positive_rate_at(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> float:
    """Fraction of correct answers wrongly flagged."""
    _, fp, tn, _ = confusion_at(labels, scores, threshold)
    if fp + tn == 0:
        raise MeasurementError(
            "FPR is undefined with no negative examples in the sample"
        )
    return fp / (fp + tn)


def threshold_for_flag_rate(scores: np.ndarray, target_flag_rate: float) -> float:
    """Smallest threshold whose measured flag rate does not exceed the target.

    Selected on **validation** scores only. Choosing it on test is one line and
    inflates every number downstream (``CLAUDE.md``, "Silent failures"); the
    caller is responsible for passing validation scores, and
    :class:`~controlplane.model.findings.OperatingPoint` records which split was used so
    the choice is visible in the warrant.

    Ties matter here. With many identical scores the achievable flag rates are
    discrete, so the returned threshold is the conservative side of the target
    and the *measured* rate is what gets reported.

    Args:
        scores: Validation scores.
        target_flag_rate: Budget in (0, 1).

    Returns:
        A threshold value drawn from the observed scores.
    """
    if not 0.0 < target_flag_rate < 1.0:
        raise MeasurementError(
            f"target_flag_rate must be in (0, 1), got {target_flag_rate}"
        )
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(scores)[::-1]
    for threshold in candidates:
        if flag_rate_at(scores, float(threshold)) > target_flag_rate:
            break
        chosen = float(threshold)
    else:
        return float(candidates[-1])
    return chosen


def weighted_error(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    w_fpr_benign: float,
    w_fnr: float,
    w_fpr_hard_negative: float,
    hard_negative_mask: Optional[np.ndarray] = None,
) -> float:
    """The declared threshold objective of ``SPEC.md`` §7.4.

    Weights come from the policy bundle and are versioned with it. This function
    does not choose them; it applies them, so the tradeoff stays a declared
    policy input rather than something the optimiser invented.

    Args:
        labels: 0/1 labels, 1 meaning incorrect.
        scores: Detector scores.
        threshold: Where the detector fires.
        w_fpr_benign: Weight on false positives outside the hard-negative set.
        w_fnr: Weight on false negatives.
        w_fpr_hard_negative: Weight on false positives inside the hard-negative
            set, which are the ones a skeptic cares about.
        hard_negative_mask: Which items are hard negatives, if any.

    Returns:
        Weighted error, lower being better.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    flagged = scores >= threshold
    negatives = labels == 0
    if hard_negative_mask is None:
        hard = np.zeros_like(negatives)
    else:
        hard = np.asarray(hard_negative_mask, dtype=bool)

    fn = float(np.sum(~flagged & (labels == 1)))
    fp_benign = float(np.sum(flagged & negatives & ~hard))
    fp_hard = float(np.sum(flagged & negatives & hard))

    n_pos = float(np.sum(labels == 1))
    n_benign = float(np.sum(negatives & ~hard))
    n_hard = float(np.sum(negatives & hard))

    numerator = 0.0
    denominator = 0.0
    if n_pos:
        numerator += w_fnr * (fn / n_pos)
        denominator += w_fnr
    if n_benign:
        numerator += w_fpr_benign * (fp_benign / n_benign)
        denominator += w_fpr_benign
    if n_hard:
        numerator += w_fpr_hard_negative * (fp_hard / n_hard)
        denominator += w_fpr_hard_negative
    if denominator == 0:
        raise MeasurementError("weighted error is undefined with no scorable items")
    return numerator / denominator


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def bootstrap_interval(
    statistic: Callable[[np.ndarray, np.ndarray], float],
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    n_resamples: int,
    ci: float,
    seed: int,
    groups: Optional[np.ndarray] = None,
    max_degenerate_fraction: float = 0.10,
) -> tuple[float, float, int]:
    """Percentile bootstrap interval for a statistic of (labels, scores).

    **Resamples groups, not rows**, when ``groups`` is given. TriviaQA ships
    several examples per question; resampling rows treats correlated items as
    independent and yields an interval narrower than the data supports. That
    error flatters the result and raises nothing, which is the combination this
    module exists to avoid.

    Resamples in which the statistic is undefined — typically a resample that
    drew only one class — are dropped rather than counted as zero. Dropping them
    biases the interval slightly toward the achievable range, so the fraction
    dropped is capped and a breach is an error rather than a warning.

    Args:
        statistic: Callable taking ``(labels, scores)`` and returning a float.
        labels: 0/1 labels.
        scores: Detector scores.
        n_resamples: Number of bootstrap resamples.
        ci: Coverage, e.g. 0.95.
        seed: Seed for the resampling RNG. Two runs at one seed give identical
            bounds, which is what ``test_determinism`` checks.
        groups: Group id per row. Resampling unit when supplied.
        max_degenerate_fraction: Largest tolerable fraction of dropped resamples.

    Returns:
        ``(ci_low, ci_high, n_effective_resamples)``.

    Raises:
        MeasurementError: If too many resamples were degenerate to trust the
            interval.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)

    if groups is None:
        index_pool = [np.array([i]) for i in range(labels.size)]
    else:
        groups = np.asarray(groups)
        index_pool = [np.flatnonzero(groups == g) for g in np.unique(groups)]

    n_units = len(index_pool)
    values: list[float] = []
    degenerate = 0
    for _ in range(n_resamples):
        picks = rng.integers(0, n_units, size=n_units)
        idx = np.concatenate([index_pool[p] for p in picks])
        try:
            values.append(float(statistic(labels[idx], scores[idx])))
        except MeasurementError:
            degenerate += 1

    if not values:
        raise MeasurementError(
            "every bootstrap resample was degenerate; the sample is too small or "
            "too imbalanced to support an interval"
        )
    dropped = degenerate / n_resamples
    if dropped > max_degenerate_fraction:
        raise MeasurementError(
            f"{degenerate}/{n_resamples} bootstrap resamples were degenerate "
            f"({dropped:.1%}, limit {max_degenerate_fraction:.0%}). The interval "
            "would be biased toward the achievable range. Report the base rate "
            "and the sample size instead of an interval this data cannot support."
        )
    if degenerate:
        _LOG.warning(
            "%d/%d bootstrap resamples degenerate (%.1f%%)",
            degenerate,
            n_resamples,
            100 * dropped,
        )
    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(values, alpha))
    high = float(np.quantile(values, 1.0 - alpha))
    return low, high, len(values)


def clopper_pearson(successes: int, n: int, ci: float) -> tuple[float, float]:
    """Exact binomial interval for a proportion. Correct at the boundary.

    Used where the bootstrap collapses. If a detector fires on **zero** of 200
    hard negatives, every bootstrap resample also contains zero events, so the
    percentile interval is ``[0, 0]`` — a claim of *perfect certainty* from 200
    observations. That is the loudest possible false claim in a project whose
    entire thesis is that unbacked claims are the problem.

    The exact interval for 0/200 at 95% is ``[0, 0.0149]``, which is the
    familiar rule of three: with no events in ``n`` trials the upper bound is
    about ``3/n``. Saying "at most 1.5%, from 200 observations" is both true and
    useful; saying "0.0%" is neither.

    Quantity: a binomial proportion. Propagation: Clopper-Pearson, inverting the
    binomial CDF via the Beta distribution, which has guaranteed coverage at the
    boundary where normal approximations and resampling both fail.

    Args:
        successes: Events observed.
        n: Trials.
        ci: Coverage, e.g. 0.95.

    Returns:
        ``(low, high)``.
    """
    from scipy import stats as scipy_stats

    if n <= 0:
        raise MeasurementError("a proportion interval needs at least one trial")
    if not 0 <= successes <= n:
        raise MeasurementError(f"{successes} successes in {n} trials is impossible")
    alpha = 1.0 - ci
    low = 0.0 if successes == 0 else float(
        scipy_stats.beta.ppf(alpha / 2, successes, n - successes + 1)
    )
    high = 1.0 if successes == n else float(
        scipy_stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes)
    )
    return low, high


def estimated(
    name: str,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    n_resamples: int,
    ci: float,
    seed: int,
    groups: Optional[np.ndarray] = None,
    unit: str = "rate",
    binomial_events: Optional[int] = None,
    binomial_trials: Optional[int] = None,
) -> Metric:
    """Compute a statistic and its bootstrap interval as one :class:`Metric`.

    The only way this module produces a rate. Point estimate and interval are
    computed from the same data in the same call, which is what stops the two
    from being computed on different splits — a mismatch that shows up as a
    value outside its own interval and is otherwise easy to miss.

    Args:
        name: Metric name, e.g. ``"recall"``.
        statistic: Callable taking ``(labels, scores)``.
        labels: 0/1 labels.
        scores: Detector scores.
        n_resamples: Bootstrap resamples.
        ci: Coverage.
        seed: Resampling seed.
        groups: Resampling unit; see :func:`bootstrap_interval`.
        unit: ``"rate"`` or ``"ratio"``.
        binomial_events: Numerator, when the statistic is a proportion. Supplied
            so the boundary case can fall back to an exact interval — see
            :func:`clopper_pearson` for why a zero-width bootstrap interval is
            the worst possible output here.
        binomial_trials: Denominator for the same fallback.

    Returns:
        An ``ESTIMATED`` metric carrying bounds, ``n`` and its estimator.
    """
    value = float(statistic(labels, scores))
    low, high, n_effective = bootstrap_interval(
        statistic,
        labels,
        scores,
        n_resamples=n_resamples,
        ci=ci,
        seed=seed,
        groups=groups,
    )
    # A percentile interval can exclude the point estimate on a skewed
    # distribution. Widening to include it is honest; silently reporting a value
    # outside its own interval is not, and Metric would reject it anyway.
    low = min(low, value)
    high = max(high, value)
    unit_of_resampling = "questions" if groups is not None else "items"
    estimator = f"bootstrap-percentile-{n_effective} over {unit_of_resampling}, seed={seed}"

    # The bootstrap collapses at the boundary. Zero events in n trials means
    # every resample also has zero, so the percentile interval is [0, 0] -- a
    # claim of perfect certainty from a finite sample. Fall back to the exact
    # binomial interval, which is correct precisely where resampling is not.
    if low == high and unit == "rate" and binomial_events is not None:
        n_trials = binomial_trials if binomial_trials is not None else int(np.asarray(labels).size)
        if n_trials == 0:
            # The bootstrap collapsed AND the exact fallback is unavailable,
            # because a proportion over zero trials is not a small number --
            # it is undefined. Emitting the [0, 0] the bootstrap produced
            # would report perfect certainty about a quantity that does not
            # exist, which is strictly worse than any interval. The caller
            # knows what the statistic means and must declare it absent.
            raise MeasurementError(
                f"{name} is undefined at this threshold: the bootstrap "
                f"degenerated to [{low}, {high}] and the exact fallback needs "
                f"trials > 0, but {binomial_events}/0 events were supplied. "
                "Report the metric as absent with a reason rather than as a "
                "zero-width interval."
            )
        if n_trials > 0:
            low, high = clopper_pearson(int(binomial_events), int(n_trials), ci)
            low, high = min(low, value), max(high, value)
            estimator = (
                f"Clopper-Pearson exact ({int(binomial_events)}/{n_trials} events); "
                "the bootstrap degenerated to a zero-width interval at the boundary"
            )

    return Metric(
        name=name,
        value=value,
        kind=MetricKind.ESTIMATED,
        n=int(np.asarray(labels).size),
        ci_low=low,
        ci_high=high,
        ci_level=ci,
        unit=unit,
        estimator=estimator,
    )


def exact_count(name: str, value: int, n: int) -> Metric:
    """A count of reviewed, confirmed items — the free claim.

    Separate constructor from :func:`estimated` so the two cannot be reached by
    changing one argument. Yield and rate are different kinds of claim, and the
    call sites should not look alike (``CLAUDE.md``, yield versus rate).
    """
    return Metric(name=name, value=float(value), kind=MetricKind.EXACT, n=n, unit="count")
