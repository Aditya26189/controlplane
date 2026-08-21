"""Metrics, bootstrap confidence intervals, and the abstention correlation.

Precision and recall are always reported separately and there is no F1 anywhere
in this repo (CLAUDE.md invariant 5). The two failure modes differ in cost by
orders of magnitude -- a false positive wastes one judge call, a false negative
lets a user act on a wrong answer -- and a blended score hides which one you
have (DECISIONS.md 005).

Every headline number is bootstrapped. A point estimate from ~600 test examples
is not defensible on its own, and "how confident are you in 14x?" is the first
question a reviewer asks (SPEC.md §6).
"""

import logging
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import Config
from src.economics import lift

LOGGER = logging.getLogger(__name__)


def confusion_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, int]:
    """Counts at the operating point. Flagged means ``score >= threshold``."""
    labels = np.asarray(labels).astype(int)
    flagged = np.asarray(scores) >= threshold
    positive = labels == 1  # positive class is "incorrect" (DECISIONS.md 004)
    return {
        "tp": int(np.sum(flagged & positive)),
        "fp": int(np.sum(flagged & ~positive)),
        "fn": int(np.sum(~flagged & positive)),
        "tn": int(np.sum(~flagged & ~positive)),
    }


def evaluate_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Score one split at the frozen threshold.

    ``flag_rate`` here is the **measured** rate -- the fraction of this split
    the probe actually flags -- and it is what every downstream calculation
    uses, never the target that was aimed at on validation (CLAUDE.md
    invariant 6).

    Args:
        labels: ``(n,)`` labels, 1 == incorrect.
        scores: ``(n,)`` probe scores.
        threshold: The frozen threshold.

    Returns:
        AUROC, base rate, measured flag rate, recall, precision, lift and the
        confusion counts. Precision is ``None`` when nothing was flagged, since
        it is undefined rather than zero.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = labels.shape[0]
    counts = confusion_at_threshold(labels, scores, threshold)

    n_positive = counts["tp"] + counts["fn"]
    n_flagged = counts["tp"] + counts["fp"]
    recall = counts["tp"] / n_positive if n_positive else None
    precision = counts["tp"] / n_flagged if n_flagged else None
    flag_rate = n_flagged / n if n else None

    auroc: Optional[float]
    if len(np.unique(labels)) < 2:
        auroc = None
        LOGGER.warning("AUROC undefined: labels contain a single class")
    else:
        auroc = float(roc_auc_score(labels, scores))

    return {
        "n": int(n),
        "auroc": auroc,
        "base_rate": float(np.mean(labels)) if n else None,
        "flag_rate": flag_rate,
        "recall": recall,
        "precision": precision,
        "lift": lift(recall, flag_rate) if (recall is not None and flag_rate) else None,
        # The largest lift attainable at this split's base rate. Resampled with
        # everything else so the interval on lift and the interval on its own
        # ceiling come from the same resamples -- otherwise the lift CI appears
        # to exceed the ceiling, which is impossible within any single resample
        # (lift / ceiling == precision <= 1). See DECISIONS.md 015.
        "ceiling": (
            1.0 / float(np.mean(labels)) if n and float(np.mean(labels)) > 0 else None
        ),
        "n_flagged": int(n_flagged),
        "n_incorrect": int(n_positive),
        "threshold": float(threshold),
        **counts,
    }


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #

BOOTSTRAP_METRICS = (
    "auroc",
    "flag_rate",
    "recall",
    "precision",
    "lift",
    "ceiling",
)


def bootstrap_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_samples: int,
    ci: float,
    seed: int,
) -> dict[str, Any]:
    """Percentile confidence intervals by resampling the test set.

    Resamples rows with replacement, recomputing every metric jointly on each
    resample -- jointly, because recall and flag rate are correlated and lift is
    their ratio, so bootstrapping them independently would understate the
    interval on the number that matters.

    The ceiling ``1/base_rate`` is resampled too. A resample that happens to draw
    fewer incorrect answers has a higher ceiling, so the interval on lift can
    reach above the ceiling computed from the *point* base rate. That is not a
    contradiction: within every individual resample ``lift / ceiling`` equals
    precision and is therefore at most 1.

    Resamples where a metric is undefined (a single label class, no positives,
    nothing flagged) are dropped for that metric and counted, rather than
    silently coerced to zero.

    Args:
        labels: ``(n,)`` labels, 1 == incorrect.
        scores: ``(n,)`` probe scores.
        threshold: The frozen threshold.
        n_samples: Number of bootstrap resamples.
        ci: Coverage, e.g. 0.95.
        seed: Seed for the resampling generator.

    Returns:
        Metric name -> point estimate, CI bounds, and the number of valid
        resamples behind that interval.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = labels.shape[0]
    rng = np.random.RandomState(seed)

    collected: dict[str, list[float]] = {name: [] for name in BOOTSTRAP_METRICS}
    for _ in range(n_samples):
        idx = rng.randint(0, n, size=n)
        result = evaluate_at_threshold(labels[idx], scores[idx], threshold)
        for name in BOOTSTRAP_METRICS:
            value = result[name]
            if value is not None:
                collected[name].append(float(value))

    alpha = (1.0 - ci) / 2.0
    point = evaluate_at_threshold(labels, scores, threshold)
    out: dict[str, Any] = {}
    for name in BOOTSTRAP_METRICS:
        values = np.array(collected[name], dtype=float)
        if values.size == 0:
            out[name] = {
                "point": point[name],
                "ci_low": None,
                "ci_high": None,
                "n_valid_resamples": 0,
            }
            continue
        out[name] = {
            "point": point[name],
            "ci_low": float(np.quantile(values, alpha)),
            "ci_high": float(np.quantile(values, 1.0 - alpha)),
            "bootstrap_mean": float(np.mean(values)),
            "n_valid_resamples": int(values.size),
        }
    out["ci"] = ci
    out["n_samples"] = int(n_samples)
    out["seed"] = int(seed)
    return out


# --------------------------------------------------------------------------- #
# The AUROC floor (TASKS.md Stage 4)
# --------------------------------------------------------------------------- #


def auroc_floor_status(auroc: Optional[float], config: Config) -> dict[str, Any]:
    """Flag a weak result loudly instead of quietly tuning it away.

    TASKS.md Stage 4 is explicit: at or below the configured floor, stop and
    report rather than tune, and work the checklist in order. A weak result
    honestly reported is a valid outcome of this repo; one manufactured by
    selecting on test is worse than nothing. This returns a status rather than
    raising, so the report still gets written and says so.

    Args:
        auroc: Measured test AUROC, or None.
        config: Resolved experiment config.

    Returns:
        A status dict recorded in ``results/probe_test.json``.
    """
    floor = config.evaluation.min_auroc_to_proceed
    below = auroc is None or auroc <= floor
    status = {
        "auroc": auroc,
        "floor": floor,
        "below_floor": bool(below),
        "checklist": [
            "confirm the probe's positive class is 'incorrect'",
            "confirm the Stage 3 left-padding equivalence check still passes",
            "widen the layer range to include earlier and later layers",
            "increase data.n_examples",
        ],
    }
    if below:
        LOGGER.warning(
            "test AUROC %s is at or below the floor %.2f. Do not tune on test. "
            "Work the checklist in order and report the result as measured "
            "(TASKS.md Stage 4).",
            "undefined" if auroc is None else f"{auroc:.4f}",
            floor,
        )
    return status


# --------------------------------------------------------------------------- #
# Secondary validation: abstention (SPEC.md §9)
# --------------------------------------------------------------------------- #


def abstention_analysis(
    scores: np.ndarray, abstained: Sequence[bool], config: Config
) -> dict[str, Any]:
    """Compare probe scores for abstaining and non-abstaining generations.

    Independent evidence that the probe reads something real rather than a
    dataset artifact: if the direction also tracks the model's own expressed
    uncertainty, that is a second, unrelated correlate.

    Below ``abstention.min_rate_to_report`` the comparison is marked
    underpowered rather than reported as a number, because a mean over a
    handful of examples is noise (SPEC.md §9).

    Args:
        scores: Test probe scores.
        abstained: Boolean abstention flags, same order.
        config: Resolved experiment config.

    Returns:
        Rates, group means, the abstention-prediction AUROC, and an
        ``underpowered`` flag.
    """
    scores = np.asarray(scores, dtype=float)
    flags = np.asarray(list(abstained)).astype(bool)
    rate = float(np.mean(flags)) if flags.size else 0.0
    underpowered = rate < config.abstention.min_rate_to_report

    result: dict[str, Any] = {
        "abstention_rate": rate,
        "n_abstained": int(np.sum(flags)),
        "n": int(flags.size),
        "min_rate_to_report": config.abstention.min_rate_to_report,
        "underpowered": bool(underpowered),
        "mean_score_abstained": float(np.mean(scores[flags])) if flags.any() else None,
        "mean_score_not_abstained": (
            float(np.mean(scores[~flags])) if (~flags).any() else None
        ),
    }
    if flags.any() and (~flags).any():
        result["auroc_predicting_abstention"] = float(roc_auc_score(flags, scores))
    else:
        result["auroc_predicting_abstention"] = None
    if underpowered:
        LOGGER.warning(
            "abstention rate %.4f is below the %.2f floor: comparison is "
            "underpowered and is reported as such",
            rate,
            config.abstention.min_rate_to_report,
        )
    return result


# --------------------------------------------------------------------------- #
# Curve data for the plots
# --------------------------------------------------------------------------- #


def roc_points(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    """ROC curve data plus the operating point, for ``report.py``.

    Returned as data rather than drawn here so the plotting code stays in one
    module and the numbers stay checkable.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return {"fpr": [], "tpr": [], "operating_point": None}
    fpr, tpr, _ = roc_curve(labels, scores)
    point = evaluate_at_threshold(labels, scores, threshold)
    fp_rate = (
        point["fp"] / (point["fp"] + point["tn"]) if (point["fp"] + point["tn"]) else None
    )
    return {
        "fpr": [float(v) for v in fpr],
        "tpr": [float(v) for v in tpr],
        "operating_point": {"fpr": fp_rate, "tpr": point["recall"]},
    }
