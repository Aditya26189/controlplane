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
from ..model import WarrantMetrics
from .stats import (
    auroc,
    estimated,
    exact_count,
    false_positive_rate_at,
    flag_rate_at,
    precision_at,
    recall_at,
)

__all__ = ["build_warrant_metrics"]

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
    else:
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
            "precision": estimated(
                "precision", lambda y, s: precision_at(y, s, threshold), labels, scores,
                n_resamples=boot, ci=ci, seed=seed, groups=groups,
                binomial_events=int(np.sum((scores >= threshold) & (labels == 1))),
                binomial_trials=n_flagged,
            ),
        }

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
        extra=()
        if (is_hard_negative_set or within_set_fpr is None)
        else (within_set_fpr,),
    )
