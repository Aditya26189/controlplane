"""The weighted-error objective. ``SPEC.md`` §7.4.

``` text
τ* = argmin_τ  Σ wᵢ·ERRORᵢ(τ) / Σ wᵢ
     w_fpr_benign = 50 ; w_fnr = 5 ; w_fpr_hard_negative = 2
```

The weights live in the bundle, are versioned and content-hashed with it, and
appear on screen. That is the whole design intent: **complexity 4 is answered by
declaring the tradeoff, not by solving it.** There is no threshold that is
correct in the abstract, and a system that presents one as if there were is
hiding the choice that actually matters. Putting the weights in a versioned
artifact means a reviewer can disagree with the tradeoff — which requires
first being able to see it.

Read the weights literally: a benign false positive is ten times worse than a
missed error here, because over-flagging destroys a high-volume channel while a
missed error is caught by the layer this system is a trigger for. That is a
claim about *this deployment*, not about detection in general, which is exactly
why it is data rather than code.

**Threshold selection is a validation-set decision.** ``CLAUDE.md`` invariant 9
and the Round 1 pitfall list: choosing τ on test inflates every number
downstream and is the first thing a reviewer checks. :func:`choose_threshold`
refuses to run on a split named test rather than trusting the caller.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping, Optional, Sequence

import numpy as np

from .errors import BundleError

__all__ = [
    "ErrorRates",
    "ThresholdChoice",
    "choose_threshold",
    "error_rates_at",
    "weighted_error",
]

#: The three error terms, in the order the objective sums them. Named here so a
#: bundle declaring an unknown weight is refused rather than having it ignored.
WEIGHT_NAMES = ("w_fpr_benign", "w_fnr", "w_fpr_hard_negative")


@dataclasses.dataclass(frozen=True)
class ErrorRates:
    """The three rates the objective weighs, at one threshold.

    Reported separately and never blended into a single score before the
    weights are applied — ``CLAUDE.md`` invariant 5 exists because a blended
    figure hides which half moved.

    Args:
        threshold: The τ these were measured at.
        fpr_benign: Share of benign items scoring at or above τ.
        fnr: Share of positives scoring below τ. Positive class is *incorrect*.
        fpr_hard_negative: Share of hard negatives scoring at or above τ.
        n_benign: Denominator for ``fpr_benign``.
        n_positive: Denominator for ``fnr``.
        n_hard_negative: Denominator for ``fpr_hard_negative``. Zero when the
            profile declares no hard-negative set, in which case the term is
            excluded from the objective rather than counted as zero error.
    """

    threshold: float
    fpr_benign: float
    fnr: float
    fpr_hard_negative: float
    n_benign: int
    n_positive: int
    n_hard_negative: int

    def to_payload(self) -> dict:
        return {
            "threshold": self.threshold,
            "fpr_benign": self.fpr_benign,
            "fnr": self.fnr,
            "fpr_hard_negative": self.fpr_hard_negative,
            "n_benign": self.n_benign,
            "n_positive": self.n_positive,
            "n_hard_negative": self.n_hard_negative,
        }


@dataclasses.dataclass(frozen=True)
class ThresholdChoice:
    """A chosen τ, the objective value, and everything needed to re-derive it.

    Args:
        threshold: τ*.
        objective: The weighted error at τ*.
        rates: The three rates at τ*.
        weights: The weights used, carried so the choice can be replayed.
        selected_on: Which split. Asserted not to be test.
        n_candidates: How many thresholds were swept.
    """

    threshold: float
    objective: float
    rates: ErrorRates
    weights: Mapping[str, float]
    selected_on: str
    n_candidates: int

    def to_payload(self) -> dict:
        return {
            "threshold": self.threshold,
            "objective": self.objective,
            "rates": self.rates.to_payload(),
            "weights": dict(self.weights),
            "selected_on": self.selected_on,
            "n_candidates": self.n_candidates,
        }


def _validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
    unknown = sorted(set(weights) - set(WEIGHT_NAMES))
    if unknown:
        raise BundleError(
            f"unknown weighted_error term(s) {unknown}; the objective sums "
            f"{list(WEIGHT_NAMES)}. An unrecognised weight would be silently "
            "dropped and the tradeoff on screen would not be the one applied."
        )
    missing = sorted(set(WEIGHT_NAMES) - set(weights))
    if missing:
        raise BundleError(
            f"weighted_error is missing {missing}. Every term is declared "
            "explicitly, including any set to zero: an absent weight and a "
            "zero weight are the same arithmetic and different intentions."
        )
    out = {name: float(weights[name]) for name in WEIGHT_NAMES}
    for name, value in out.items():
        if value < 0:
            raise BundleError(f"weighted_error.{name} must be >= 0, got {value}")
    if sum(out.values()) <= 0:
        raise BundleError(
            "weighted_error weights sum to zero, so every threshold scores the "
            "same and τ* would be whichever the sweep happened to visit first"
        )
    return out


def error_rates_at(
    threshold: float,
    *,
    benign: Sequence[float],
    positive: Sequence[float],
    hard_negative: Sequence[float] = (),
) -> ErrorRates:
    """The three rates at one threshold.

    Args:
        threshold: τ. A detector fires at or above it.
        benign: Scores of ordinary correct items.
        positive: Scores of items that are *incorrect*. Positive class is
            incorrect throughout this repo; inverting it yields ``1 − AUROC``
            and reads as a strong negative result.
        hard_negative: Scores of the hard-negative set, if one exists.

    Returns:
        An :class:`ErrorRates`.
    """
    b = np.asarray(benign, dtype=float)
    p = np.asarray(positive, dtype=float)
    h = np.asarray(hard_negative, dtype=float)

    return ErrorRates(
        threshold=float(threshold),
        fpr_benign=float((b >= threshold).mean()) if b.size else 0.0,
        fnr=float((p < threshold).mean()) if p.size else 0.0,
        fpr_hard_negative=float((h >= threshold).mean()) if h.size else 0.0,
        n_benign=int(b.size),
        n_positive=int(p.size),
        n_hard_negative=int(h.size),
    )


def weighted_error(rates: ErrorRates, weights: Mapping[str, float]) -> float:
    """Σ wᵢ·ERRORᵢ / Σ wᵢ at one threshold.

    The hard-negative term is **excluded from both sums** when no hard-negative
    set was supplied. Counting an unmeasured rate as zero error would make every
    threshold look better by exactly ``w_fpr_hard_negative / Σw`` and would
    quietly reweight the two terms that were measured.

    Args:
        rates: From :func:`error_rates_at`.
        weights: Validated weights.

    Returns:
        The objective value in ``[0, 1]``.
    """
    w = _validate_weights(weights)
    terms = [
        (w["w_fpr_benign"], rates.fpr_benign),
        (w["w_fnr"], rates.fnr),
    ]
    if rates.n_hard_negative > 0:
        terms.append((w["w_fpr_hard_negative"], rates.fpr_hard_negative))

    total_weight = sum(weight for weight, _ in terms)
    if total_weight <= 0:
        raise BundleError(
            "every applicable weight is zero, so the objective cannot rank "
            "thresholds"
        )
    return sum(weight * value for weight, value in terms) / total_weight


def choose_threshold(
    *,
    benign: Sequence[float],
    positive: Sequence[float],
    weights: Mapping[str, float],
    selected_on: str,
    hard_negative: Sequence[float] = (),
    candidates: Optional[Sequence[float]] = None,
) -> ThresholdChoice:
    """Find τ* by sweeping candidate thresholds.

    Args:
        benign: Scores of ordinary correct items.
        positive: Scores of incorrect items.
        weights: From the bundle's ``weighted_error`` block.
        selected_on: Split name. **Must not be test** — selecting a threshold on
            test inflates the headline number and is the first thing a reviewer
            checks (``CLAUDE.md`` invariant 9).
        hard_negative: Hard-negative scores, if any.
        candidates: Thresholds to sweep. Defaults to the midpoints between
            consecutive observed scores plus the two open ends, which is the
            smallest set containing every distinct classifier this data can
            produce.

    Returns:
        A :class:`ThresholdChoice`.

    Raises:
        BundleError: If ``selected_on`` names test, or either class is empty.
    """
    if "test" in selected_on.lower():
        raise BundleError(
            f"threshold selection ran on split {selected_on!r}. Layer, "
            "regularisation and threshold are validation-set decisions; "
            "choosing τ on test inflates every number that follows it "
            "(CLAUDE.md invariant 9)."
        )
    b = np.asarray(benign, dtype=float)
    p = np.asarray(positive, dtype=float)
    if b.size == 0 or p.size == 0:
        raise BundleError(
            "the objective needs both classes present; got "
            f"{b.size} benign and {p.size} positive. A single-class split "
            "supports no threshold choice at all."
        )

    if candidates is None:
        observed = np.unique(
            np.concatenate([b, p, np.asarray(hard_negative, dtype=float)])
        )
        midpoints = (observed[:-1] + observed[1:]) / 2.0 if observed.size > 1 else observed
        span = float(observed[-1] - observed[0]) if observed.size > 1 else 1.0
        candidates = np.concatenate(
            [[float(observed[0]) - span * 1e-3], midpoints, [float(observed[-1]) + span * 1e-3]]
        )

    best: Optional[ThresholdChoice] = None
    validated = _validate_weights(weights)
    for threshold in candidates:
        rates = error_rates_at(
            threshold, benign=b, positive=p, hard_negative=hard_negative
        )
        value = weighted_error(rates, validated)
        # Strict improvement only. On a tie the lower threshold wins, which is
        # the recall-favouring side of the tie and matches what the probe is
        # for: it is a trigger, and a false positive costs one wasted check
        # while a false negative costs a customer acting on a wrong answer.
        if best is None or value < best.objective:
            best = ThresholdChoice(
                threshold=float(threshold),
                objective=float(value),
                rates=rates,
                weights=validated,
                selected_on=selected_on,
                n_candidates=len(candidates),
            )

    assert best is not None, "candidate sweep was empty"
    return best
