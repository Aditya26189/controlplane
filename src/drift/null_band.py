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

__all__ = ["NullBand", "simulate_null_psi"]


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
