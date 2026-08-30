"""The null distribution of PSI, so a band can be checked rather than assumed.

PSI's 0.10 / 0.25 bands are credit-scoring rules of thumb, and they are quoted
without their sample size. They are not scale-free: under the null — a window
drawn from the very distribution the reference was built from — PSI has expected
value roughly ``(k-1)/n``, so the same band means different things at different
bin counts and window sizes.

Measured, at the configured 200-request window:

| bins | n | null p95 | P(PSI > 0.10) |
|---|---|---|---|
| 5 | 200 | 0.057 | 0.00 |
| 10 | 200 | 0.110 | **0.08** |
| 20 | 200 | 0.224 | **0.79** |
| 10 | 2000 | 0.028 | 0.00 |

At ten bins and a 200-window, one stable window in twelve reports
``MODERATE_SHIFT``. At twenty bins the monitor is alarmed four times in five on
traffic that has not moved. ``SPEC.md`` §5.2 says *do not revoke on noise*, and
a monitor that cries wolf at that rate gets switched off — which is the same
outcome as not having built it.

Same shape as ``DECISIONS.md`` 029 and 031, where a fixed negative-control band
was wrong at the sample size actually used and had to be sized from the measured
null. The fix is the same: simulate the null and check the band against it.

The simulation samples bin indices from the reference's own multinomial, which
is exactly the null hypothesis and needs no access to raw values.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np

__all__ = [
    "IqrRatioNull",
    "NullBand",
    "iqr_ratio_power",
    "simulate_null_iqr_ratio",
    "simulate_null_psi",
]


@dataclasses.dataclass(frozen=True)
class NullBand:
    """Where PSI lands when nothing has drifted.

    Args:
        n: Window size simulated.
        n_bins: Bin count of the reference feature.
        n_reference: Reference size the null was simulated at.
        repeats: Simulation draws.
        p50: Median null PSI.
        p95: 95th percentile — the level a stable window exceeds one time in
            twenty.
        p99: 99th percentile.
        false_alarm_rate: Share of null windows exceeding the configured
            ``psi_stable`` band. This is the number that decides whether the
            band is usable at this ``(n_bins, n)``.
    """

    n: int
    n_bins: int
    n_reference: int
    repeats: int
    p50: float
    p95: float
    p99: float
    false_alarm_rate: float

    def to_payload(self) -> dict:
        return dataclasses.asdict(self)


def simulate_null_psi(
    bin_probabilities: Sequence[float],
    *,
    n: int,
    n_reference: int,
    stable_band: float,
    repeats: int = 400,
    seed: int = 1729,
    epsilon: float = 1e-6,
) -> NullBand:
    """Simulate PSI for windows drawn from the reference distribution itself.

    Args:
        bin_probabilities: The reference feature's bin proportions.
        n: Window size to simulate.
        n_reference: Items the reference bins were built from. The
            reference is resampled at this size rather than held exact,
            because its own estimation error is part of the null.
        stable_band: The ``psi_stable`` threshold, for the false-alarm rate.
        repeats: Draws. 400 gives the p95 about two decimal places, which is
            all the band needs.
        seed: Fixed, so a monitor's guard does not pass on one run and fail on
            the next for reasons unrelated to its configuration.
        epsilon: Same floor the live computation uses, so the null is scored
            the way the observations will be.

    Returns:
        A :class:`NullBand`.
    """
    truth = np.asarray(list(bin_probabilities), dtype=float)
    truth = truth / truth.sum()
    rng = np.random.default_rng(seed)

    # BOTH sides are drawn. Treating the stored bin probabilities as exact
    # ignores that they were themselves estimated from n_reference items, and
    # that estimation error is a real part of the noise a live window is scored
    # against. Holding the reference fixed put the measured false-alarm rate at
    # 0.02 where resampling the underlying values put it at 0.08 -- a guard
    # built on the optimistic number would pass a configuration that alarms
    # four times as often as it promised.
    live = rng.multinomial(n, truth, size=repeats).astype(float) / n
    reference = (
        rng.multinomial(n_reference, truth, size=repeats).astype(float) / n_reference
        if n_reference
        else np.broadcast_to(truth, (repeats, truth.size))
    )

    actual_safe = np.where(live <= 0.0, epsilon, live)
    expected_safe = np.where(reference <= 0.0, epsilon, reference)
    psi = ((actual_safe - expected_safe) * np.log(actual_safe / expected_safe)).sum(axis=1)

    return NullBand(
        n=int(n),
        n_bins=int(truth.size),
        n_reference=int(n_reference),
        repeats=int(repeats),
        p50=float(np.percentile(psi, 50)),
        p95=float(np.percentile(psi, 95)),
        p99=float(np.percentile(psi, 99)),
        false_alarm_rate=float((psi > stable_band).mean()),
    )


@dataclasses.dataclass(frozen=True)
class IqrRatioNull:
    """Where the pilot/reference IQR ratio lands when nothing has collapsed.

    Args:
        n_pilot: Effective pilot size simulated. For a clustered pilot this is
            the number of **clusters**, not items (``DECISIONS.md`` 090).
        n_reference: Reference test-split size the ratio is taken against.
        shape: Score-shape family the null was drawn from.
        repeats: Simulation draws.
        p2_5, p5, p50, p95, p97_5: Percentiles of the null ratio.
        false_alarm_rate: Share of null draws falling below ``threshold`` — the
            rate at which a perfectly healthy probe is called saturated.
        threshold: The saturation rule the false-alarm rate was measured at.
    """

    n_pilot: int
    n_reference: int
    shape: str
    repeats: int
    p2_5: float
    p5: float
    p50: float
    p95: float
    p97_5: float
    false_alarm_rate: float
    threshold: float

    def to_payload(self) -> dict:
        return dataclasses.asdict(self)


#: Score shapes the null is checked across. The probe emits a decision-function
#: value whose shape is not known a priori, so a band that only holds under one
#: of these is a band that holds by luck.
_SHAPES = ("normal", "logistic", "beta2_2")


def _draw_scores(
    rng: np.random.Generator, shape: str, size: tuple[int, ...]
) -> np.ndarray:
    """Draw scores of a given shape, standardised so IQRs are comparable."""
    if shape == "normal":
        return rng.normal(0.0, 1.0, size=size)
    if shape == "logistic":
        return rng.logistic(0.0, 1.0, size=size)
    if shape == "beta2_2":
        return rng.beta(2.0, 2.0, size=size)
    raise ValueError(f"unknown score shape {shape!r}; expected one of {_SHAPES}")


def _iqr(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Interquartile range, linear interpolation, along one axis."""
    return np.percentile(values, 75, axis=axis) - np.percentile(values, 25, axis=axis)


def simulate_null_iqr_ratio(
    *,
    n_pilot: int,
    n_reference: int,
    threshold: float,
    shape: str = "normal",
    repeats: int = 20000,
    seed: int = 1729,
) -> IqrRatioNull:
    """The IQR ratio's null band, drawing the pilot from the reference's own law.

    **What this is for.** ``101`` routes the pilot on ``IQR ratio < 0.439 =
    saturation``. That threshold was derived once, by hand, and lives as a
    hardcoded constant. A rule that decides whether a GPU run gets re-authored
    has to be regenerable, and its false-alarm rate has to be stated next to it
    — otherwise the pilot's most likely outcomes are "passes because the band is
    enormous" and "fails because the band is enormous", and the ratio alone does
    not distinguish them.

    **Both sides are drawn**, for the reason ``simulate_null_psi`` gives: the
    reference IQR is itself estimated from a finite test split, and holding it
    exact would report a band narrower than the one a real comparison faces.

    Args:
        n_pilot: Effective pilot size. Pass the **cluster** count for a
            clustered pilot — two items per question share a question, so
            twenty-four items carry twelve questions' worth of independent
            information and their IQR runs narrow for reasons that have nothing
            to do with saturation.
        n_reference: Reference test-split size.
        threshold: The saturation rule, for the false-alarm rate.
        shape: One of ``_SHAPES``.
        repeats: Draws. 20,000 puts the p5 inside about +-0.01.
        seed: Fixed, so the band does not move between runs.

    Returns:
        An :class:`IqrRatioNull`.
    """
    if n_pilot < 4:
        raise ValueError(f"n_pilot must be at least 4 to have an IQR, got {n_pilot}")
    rng = np.random.default_rng(seed)

    pilot = _draw_scores(rng, shape, (repeats, n_pilot))
    reference = _draw_scores(rng, shape, (repeats, n_reference))
    ratio = _iqr(pilot) / _iqr(reference)

    return IqrRatioNull(
        n_pilot=int(n_pilot),
        n_reference=int(n_reference),
        shape=shape,
        repeats=int(repeats),
        p2_5=float(np.percentile(ratio, 2.5)),
        p5=float(np.percentile(ratio, 5)),
        p50=float(np.percentile(ratio, 50)),
        p95=float(np.percentile(ratio, 95)),
        p97_5=float(np.percentile(ratio, 97.5)),
        false_alarm_rate=float((ratio < threshold).mean()),
        threshold=float(threshold),
    )


def iqr_ratio_power(
    *,
    n_pilot: int,
    n_reference: int,
    threshold: float,
    collapse: float,
    shape: str = "normal",
    repeats: int = 20000,
    seed: int = 1729,
) -> float:
    """P(the saturation rule fires) when the pilot's spread really is ``collapse``x.

    ``collapse=1.0`` is the null, and the value returned there is the
    false-alarm rate rather than power. Reported in the same table as the
    genuine alternatives, because the distance between the two is the whole
    question of whether the pilot can answer anything at this size.

    Args:
        n_pilot: Effective pilot size — clusters, not items.
        n_reference: Reference test-split size.
        threshold: The saturation rule.
        collapse: True multiplicative shrinkage of the pilot's score spread.
        shape: One of ``_SHAPES``.
        repeats: Draws.
        seed: Fixed.

    Returns:
        The probability the rule fires, in [0, 1].
    """
    if collapse <= 0.0:
        raise ValueError(f"collapse must be positive, got {collapse}")
    rng = np.random.default_rng(seed)

    pilot = _draw_scores(rng, shape, (repeats, n_pilot))
    # Shrink about the sample median: a collapse narrows the spread without
    # moving the operating point, which is what saturation looks like.
    pilot = np.median(pilot, axis=-1, keepdims=True) + collapse * (
        pilot - np.median(pilot, axis=-1, keepdims=True)
    )
    reference = _draw_scores(rng, shape, (repeats, n_reference))
    ratio = _iqr(pilot) / _iqr(reference)
    return float((ratio < threshold).mean())
