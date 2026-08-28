"""The ROC curve and the local slope at an operating point. ``DECISIONS.md`` 082.

Built to test one hypothesis: that the three profiles' recall bounds are
*unequally* sensitive to where their threshold sits, and that this is a property
of the curve's shape rather than noise.

**Why the slope is the quantity of interest.** The ROC's local slope at a
threshold is ``dTPR/dFPR`` — how much recall a marginal loosening of the
threshold buys, per unit of false-positive rate spent. On a steep segment a
small threshold move is a large recall move; on a flat one it is not. If the
low-flag-rate profile sits on the steep part, then the profile with the highest
traffic and the tightest latency budget is also the one whose warranted recall
is most sensitive to threshold placement — which is a direct argument for
warranting operating points individually rather than warranting a detector, made
on measured numbers instead of asserted.

If the slopes come out comparable, the asymmetry is noise and the argument is
dropped rather than restated more weakly.

**The slope is estimated over a window, not from adjacent points.** An empirical
ROC from ``n`` items is a step function; the derivative between two consecutive
vertices is either zero or infinite and neither is informative. The window is
declared by the caller and reported alongside the estimate, because a slope
quoted without the interval it was measured over is not reproducible.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np

__all__ = ["OperatingPointGeometry", "RocCurve", "roc_curve"]


@dataclasses.dataclass(frozen=True)
class OperatingPointGeometry:
    """Where one operating point sits on the curve, and how steep it is there.

    Args:
        operating_point_id: Which point.
        threshold: The score threshold, from the run artifact.
        fpr: False-positive rate at that threshold.
        tpr: True-positive rate — recall — at that threshold.
        flag_rate: Share of all items at or above the threshold.
        slope: Local ``dTPR/dFPR``, estimated over ``window_fpr``.
        window_fpr: The FPR span the slope was measured across.
        recall_per_flag_rate_point: Recall bought per additional 1% of traffic
            flagged. The slope in the units the profiles are actually budgeted
            in, since a profile declares a flag-rate budget rather than an FPR.
    """

    operating_point_id: str
    threshold: float
    fpr: float
    tpr: float
    flag_rate: float
    slope: float
    window_fpr: float
    recall_per_flag_rate_point: float

    def to_payload(self) -> dict:
        return {
            "operating_point_id": self.operating_point_id,
            "threshold": self.threshold,
            "fpr": self.fpr,
            "tpr": self.tpr,
            "flag_rate": self.flag_rate,
            "slope": self.slope,
            "window_fpr": self.window_fpr,
            "recall_per_flag_rate_point": self.recall_per_flag_rate_point,
        }


@dataclasses.dataclass(frozen=True)
class RocCurve:
    """An empirical ROC and the operating points marked on it.

    Args:
        fpr: False-positive rates, ascending.
        tpr: True-positive rates, ascending.
        thresholds: The score at each vertex.
        auroc: Area under the curve.
        n_positive: Items labelled incorrect.
        n_negative: The rest.
        points: Geometry at each named operating point.
    """

    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auroc: float
    n_positive: int
    n_negative: int
    points: tuple[OperatingPointGeometry, ...] = ()

    def to_payload(self) -> dict:
        return {
            "auroc": self.auroc,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "n_vertices": int(len(self.fpr)),
            "points": [p.to_payload() for p in self.points],
        }


def _slope_at(
    fpr: np.ndarray, tpr: np.ndarray, at_fpr: float, window: float
) -> tuple[float, float]:
    """Least-squares slope of the curve within ``±window`` of ``at_fpr``.

    A regression over the window rather than a two-point difference: the
    empirical curve is a step function, so consecutive vertices give a
    derivative of zero or infinity and neither describes the neighbourhood.

    Returns:
        ``(slope, realised_window)``. The realised window can be narrower than
        requested near the ends of the curve, and is reported so a reader can
        see when that happened.
    """
    low, high = at_fpr - window, at_fpr + window
    inside = (fpr >= low) & (fpr <= high)
    if inside.sum() < 2:
        # Widen until two vertices are in view rather than returning a slope
        # from a single point, which would be an invented number.
        order = np.argsort(np.abs(fpr - at_fpr))
        inside = np.zeros(len(fpr), dtype=bool)
        inside[order[:2]] = True
    x, y = fpr[inside], tpr[inside]
    realised = float(x.max() - x.min())
    if realised <= 0:
        return float("nan"), realised
    slope = float(np.polyfit(x, y, 1)[0])
    return slope, realised


def roc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    operating_points: Optional[dict[str, float]] = None,
    slope_window: float = 0.05,
) -> RocCurve:
    """Compute the empirical ROC and the geometry at each operating point.

    Args:
        scores: Detector scores.
        labels: 0/1, 1 meaning *incorrect* — the positive class throughout this
            repo. Inverting it yields a curve below the diagonal and an AUROC of
            ``1 - AUROC``, which reads as a strong negative result.
        operating_points: Id to threshold, read from run artifacts.
        slope_window: Half-width in FPR over which each local slope is fitted.

    Returns:
        A :class:`RocCurve`.

    Raises:
        ValueError: If either class is absent, in which case no curve exists.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    positive = labels == 1
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"a ROC needs both classes; got {n_pos} positive and {n_neg} "
            "negative. A single-class split supports no curve at all."
        )

    order = np.argsort(-scores, kind="mergesort")
    ranked = positive[order]
    ranked_scores = scores[order]

    tp = np.cumsum(ranked)
    fp = np.cumsum(~ranked)
    # Keep only the last vertex of each run of equal scores: a threshold cannot
    # separate items that scored identically, so the intermediate points are not
    # achievable operating points.
    keep = np.append(np.diff(ranked_scores) != 0, True)

    tpr = np.concatenate([[0.0], tp[keep] / n_pos])
    fpr = np.concatenate([[0.0], fp[keep] / n_neg])
    thresholds = np.concatenate([[np.inf], ranked_scores[keep]])
    auroc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(
        np.trapz(tpr, fpr)
    )

    geometry: list[OperatingPointGeometry] = []
    for point_id, threshold in (operating_points or {}).items():
        fires = scores >= threshold
        point_fpr = float(fires[~positive].mean())
        point_tpr = float(fires[positive].mean())
        slope, realised = _slope_at(fpr, tpr, point_fpr, slope_window)
        # Recall bought per extra 1% of *traffic* flagged, which is the unit a
        # profile's budget is written in. dFlagRate = (n_neg/n)·dFPR + …, so the
        # conversion is not the slope itself and is computed rather than assumed.
        base_rate = n_pos / (n_pos + n_neg)
        per_flag_point = slope / (slope * base_rate + (1.0 - base_rate)) / 100.0
        geometry.append(
            OperatingPointGeometry(
                operating_point_id=point_id,
                threshold=float(threshold),
                fpr=point_fpr,
                tpr=point_tpr,
                flag_rate=float(fires.mean()),
                slope=slope,
                window_fpr=realised,
                recall_per_flag_rate_point=float(per_flag_point),
            )
        )

    return RocCurve(
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        auroc=auroc,
        n_positive=n_pos,
        n_negative=n_neg,
        points=tuple(geometry),
    )
