"""The single place a :class:`WarrantMetrics` is built from scores and labels.

**Why this module exists.** The same quantity had two implementations — one on
the activation path, one on the text path — and the `fpr_hard_negatives`
conflation shipped in one, was fixed in one, and survived in the other through
193 passing tests (``DECISIONS.md`` 036, 040). The failure class is *"same
quantity, two implementations, one drifts"*, and a documented pitfall does not
prevent it. One implementation does.

Both runners call :func:`build_warrant_metrics` and neither computes a metric
itself. ``test_metric_paths_agree`` asserts the two paths produce byte-identical
metrics from identical input, so a future divergence fails a test rather than
waiting to be noticed.

Two rules live here rather than in either caller, because both callers got one
of them wrong at some point:

* **Within-set FPR is not hard-negative FPR.** ``fpr_hard_negatives`` is the
  field a profile declares a maximum against, and it is populated only when the
  set under test *is* the hard-negative set. Otherwise the within-set FPR is
  reported as ``fpr``. Passed explicitly rather than inferred from a set's name,
  because names get changed.
* **Proportions get an exact interval at the boundary.** Zero events in `n`
  trials collapses the bootstrap to zero width, which claims perfect certainty
  from a finite sample (``DECISIONS.md`` 035).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import Config
from ..model import Metric, MetricKind, WarrantMetrics
from .stats import (
    auroc,
    estimated,
    exact_count,
    false_positive_rate_at,
    flag_rate_at,
    precision_at,
    recall_at,
)

__all__ = ["assert_metric_shape_compatible", "build_warrant_metrics"]

_LOG = logging.getLogger(__name__)


def build_warrant_metrics(
    config: Config,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    groups: Optional[np.ndarray] = None,
    is_hard_negative_set: bool = False,
) -> WarrantMetrics:
    """Build every metric a warrant carries, from one scoring.

    Args:
        config: Supplies the bootstrap count, the coverage level and the seed,
            so two paths cannot disagree about how wide an interval is.
        labels: 0/1 labels, 1 meaning the detector should fire.
        scores: Detector scores, higher meaning more likely positive.
        threshold: The operating point.
        groups: Resampling unit for the bootstrap — question ids where several
            items share a question. Resampling rows instead treats correlated
            items as independent and narrows the interval below what the data
            supports.
        is_hard_negative_set: Whether the set under test *is* the hard-negative
            set. Only then does its FPR populate ``fpr_hard_negatives`` and face
            a profile's declared maximum.

    Returns:
        A :class:`WarrantMetrics`. On a single-class set the ranking metrics are
        all ``None`` together, which is what invariant 5 requires and what
        ``DECISIONS.md`` 032 explains.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    boot = config.validation.bootstrap_samples
    ci = config.validation.ci
    seed = config.seed

    single_class = len(set(labels.tolist())) < 2
    n_flagged = int(np.sum(scores >= threshold))
    n_items = int(labels.size)

    if single_class:
        _LOG.info(
            "single-class set (%d items, all class %d): AUROC, recall and "
            "precision are undefined, so this warrant claims FPR only",
            n_items,
            int(labels[0]) if n_items else -1,
        )
        ranking: dict[str, object] = {"auroc": None, "recall": None, "precision": None}
        ranking_absent_reason = (
            f"single-class envelope: all {n_items} items carry the same label, "
            "so there is no ranking to score"
        )
    else:
        ranking_absent_reason = None
        ranking = {
            "auroc": estimated(
                "auroc", auroc, labels, scores,
                n_resamples=boot, ci=ci, seed=seed, groups=groups, unit="ratio",
            ),
            "recall": estimated(
                "recall", lambda y, s: recall_at(y, s, threshold), labels, scores,
                n_resamples=boot, ci=ci, seed=seed, groups=groups,
                binomial_events=int(np.sum((scores >= threshold) & (labels == 1))),
                binomial_trials=int(np.sum(labels == 1)),
            ),
            # Precision is TP/(TP+FP). With NOTHING flagged the denominator
            # is zero, so the quantity is undefined -- and the bootstrap does
            # not know that: it resamples all-zero data and returns [0, 0], a
            # claim of PERFECT CERTAINTY about a ratio with no denominator.
            # That shipped in results/transfer-T1-mean_pool.json.
            #
            # The interval is [0, 1], not absent. Absent would take AUROC and
            # recall with it (invariant 5, enforced in the model layer), and on
            # that same artifact AUROC is 0.5015 -- the mean-pool long-context
            # collapse, a README claim and the documented failure mode. Dropping
            # a measured result to avoid an undefined one is the wrong trade.
            # Zero flagged items tell us NOTHING about precision, and [0, 1] is
            # exactly that statement. DECISIONS 108.
            "precision": (
                Metric(
                    name="precision",
                    value=float(precision_at(labels, scores, threshold)),
                    kind=MetricKind.ESTIMATED,
                    n=n_items,
                    ci_low=0.0,
                    ci_high=1.0,
                    ci_level=ci,
                    unit="rate",
                    convention="two_sided_95",
                    estimator=(
                        f"undefined: 0 of {n_items} items flagged at this "
                        "threshold, so precision is 0/0. The interval is "
                        "vacuous [0, 1] rather than the zero-width interval a "
                        "bootstrap returns on all-zero data"
                    ),
                )
                if n_flagged == 0
                else estimated(
                    "precision", lambda y, s: precision_at(y, s, threshold),
                    labels, scores,
                    n_resamples=boot, ci=ci, seed=seed, groups=groups,
                    binomial_events=int(np.sum((scores >= threshold) & (labels == 1))),
                    binomial_trials=n_flagged,
                )
            ),
        }
        if n_flagged == 0:
            _LOG.info(
                "nothing flagged at threshold %.6f on %d items: precision is "
                "undefined (0/0) and carries a vacuous [0, 1] interval; AUROC "
                "and recall are unaffected and still measured",
                threshold, n_items,
            )

    within_set_fpr = None
    if (labels == 0).any():
        within_set_fpr = estimated(
            "fpr_hard_negatives" if is_hard_negative_set else "fpr",
            lambda y, s: false_positive_rate_at(y, s, threshold),
            labels, scores,
            n_resamples=boot, ci=ci, seed=seed, groups=groups,
            binomial_events=int(np.sum((scores >= threshold) & (labels == 0))),
            binomial_trials=int(np.sum(labels == 0)),
        )

    return WarrantMetrics(
        auroc=ranking["auroc"],
        recall=ranking["recall"],
        precision=ranking["precision"],
        ranking_absent_reason=ranking_absent_reason,
        flag_rate=estimated(
            "flag_rate", lambda y, s: flag_rate_at(s, threshold), labels, scores,
            n_resamples=boot, ci=ci, seed=seed, groups=groups,
            binomial_events=n_flagged, binomial_trials=n_items,
        ),
        confirmed_errors=exact_count(
            "confirmed_errors",
            int(np.sum((scores >= threshold) & (labels == 1))),
            n=n_flagged,
        ),
        fpr_hard_negatives=within_set_fpr if is_hard_negative_set else None,
        base_rate=float(labels.mean()) if n_items else None,
        extra=()
        if (is_hard_negative_set or within_set_fpr is None)
        else (within_set_fpr,),
    )


def assert_metric_shape_compatible(
    first: WarrantMetrics, second: WarrantMetrics, *, first_name: str, second_name: str
) -> None:
    """Assert two metric sets have the same *structure*, not the same values.

    The remaining dual-path risk after the duplication was removed is not two
    implementations of one quantity — it is the fixture path and the real
    extraction path producing metrics that *should* be comparable and silently
    are not, because they differ in normalisation, split derivation or label
    polarity. Values must differ; shape must not.

    Run this when the real extraction lands, on the same eval set id through
    both paths. A mismatch means the two are not measuring the same thing, and
    every comparison between fixture and measured results is void.

    Checks, in the order they would fail:

    * the same metrics are present and the same ones absent — a path that
      silently drops recall on a set the other measures it on is the failure;
    * each shared metric has the same ``kind`` and ``unit`` — an ``EXACT`` count
      on one side and an ``ESTIMATED`` rate on the other is the yield/rate
      confusion arriving through the back door;
    * every estimated metric carries an interval on both sides.

    Args:
        first: Metrics from one path.
        second: Metrics from the other.
        first_name: Human-readable name for error messages, e.g. ``"fixture"``.
        second_name: Likewise, e.g. ``"measured"``.

    Raises:
        AssertionError: Naming the specific divergence.
    """
    def shape(metrics: WarrantMetrics) -> dict[str, tuple[str, str, bool]]:
        return {
            m.name: (m.kind.value, m.unit, m.has_interval)
            for m in metrics.all_metrics()
        }

    left, right = shape(first), shape(second)

    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    assert not only_left and not only_right, (
        f"metric sets differ in which metrics exist: {first_name} has "
        f"{only_left or 'nothing extra'}, {second_name} has "
        f"{only_right or 'nothing extra'}. The two paths are not measuring the "
        "same thing, so no comparison between them is meaningful."
    )

    for name in sorted(left):
        assert left[name] == right[name], (
            f"metric {name!r} has a different shape on the two paths: "
            f"{first_name} is kind={left[name][0]} unit={left[name][1]} "
            f"interval={left[name][2]}, {second_name} is kind={right[name][0]} "
            f"unit={right[name][1]} interval={right[name][2]}"
        )

    for metrics, label in ((first, first_name), (second, second_name)):
        for metric in metrics.all_metrics():
            if metric.kind is MetricKind.ESTIMATED:
                assert metric.has_interval, (
                    f"{label}: estimated metric {metric.name!r} carries no interval"
                )
