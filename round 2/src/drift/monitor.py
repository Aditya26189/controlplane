"""The sliding window and the revocation ladder (``SPEC.md`` §5.2, §5.3).

Observes live requests, keeps the last ``config.drift.window_size`` of them, and
maps the worst per-feature PSI onto a rung: ``INSIDE``, ``MODERATE_SHIFT``,
``SIGNIFICANT_SHIFT``. Below the window minimum there is no verdict at all.

**Three refusals are load-bearing here.**

*No verdict below the minimum.* ``INSUFFICIENT_DATA`` is a distinct state, not a
default to ``INSIDE``. A monitor that returns "no drift" on forty requests is
asserting stability it has not measured, and the first person to notice will
disable it. The spec's 200 is a floor on evidence, not a tuning knob.

*The worst feature decides.* Aggregating across features — a mean, a vote —
lets four stable features outvote the one that moved, and the one that moved is
the entire signal. Token length is the documented probe killer; it does not get
averaged against script mix.

*A feature the live traffic does not carry is not evidence of stability.* If a
window has no values for a configured feature, that feature is reported as
unobserved rather than scored as inside. Absence is not a measurement
(``DECISIONS.md`` 050).
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Mapping, Optional, Sequence

from ..model.enums import EnvelopeState
from ..model.findings import DistributionEnvelope
from .null_band import NullBand, simulate_null_psi
from .psi import PsiResult, population_stability_index, state_for_psi

__all__ = ["DriftVerdict", "DriftMonitor"]

#: Ladder order, so the worst rung across features can be taken without
#: comparing enum members by name.
_SEVERITY = {
    EnvelopeState.INSIDE: 0,
    EnvelopeState.MODERATE_SHIFT: 1,
    EnvelopeState.SIGNIFICANT_SHIFT: 2,
    EnvelopeState.INSUFFICIENT_DATA: -1,
}


@dataclasses.dataclass(frozen=True)
class DriftVerdict:
    """What the window says about live traffic, and why.

    Args:
        state: The rung. ``INSUFFICIENT_DATA`` means no verdict was reached.
        n_observed: Requests in the window when the verdict was taken.
        window_size: The configured minimum.
        per_feature: PSI per configured feature that the window carried values
            for.
        unobserved: Configured features the window carried no values for. These
            are **not** counted as stable.
        driver: The feature whose rung set the verdict, if any.
        reason: One line, for the certificate.
    """

    state: EnvelopeState
    n_observed: int
    window_size: int
    per_feature: Mapping[str, PsiResult]
    unobserved: tuple[str, ...]
    driver: Optional[str]
    reason: str

    @property
    def revokes(self) -> bool:
        """Whether this verdict takes a warrant out of service."""
        return self.state is EnvelopeState.SIGNIFICANT_SHIFT

    @property
    def widens(self) -> bool:
        """Whether this verdict widens reported bounds and schedules revalidation."""
        return self.state is EnvelopeState.MODERATE_SHIFT

    def to_payload(self) -> dict:
        return {
            "state": self.state.value,
            "n_observed": self.n_observed,
            "window_size": self.window_size,
            "driver": self.driver,
            "reason": self.reason,
            "unobserved_features": list(self.unobserved),
            "per_feature": {k: v.to_payload() for k, v in self.per_feature.items()},
        }


class DriftMonitor:
    """A sliding window over live requests, scored against one warrant's envelope.

    Scored against the envelope stored *in the warrant*, never against a
    re-derived reference. Rebinning on recent traffic compares each window to
    itself and reports stability forever — the failure that makes a drift
    monitor worse than not having one.
    """

    def __init__(
        self,
        envelope: DistributionEnvelope,
        *,
        window_size: int,
        psi_stable: float,
        psi_significant: float,
        features: Sequence[str],
        max_false_alarm_rate: Optional[float] = None,
    ) -> None:
        self.envelope = envelope
        self.window_size = int(window_size)
        self.psi_stable = float(psi_stable)
        self.psi_significant = float(psi_significant)
        #: Configured feature names. A feature the envelope has no reference for
        #: cannot be scored, and is reported unobserved rather than skipped.
        self.features = tuple(features)
        self._reference = {f.name: f for f in envelope.features}
        self._windows: dict[str, collections.deque] = {
            name: collections.deque(maxlen=self.window_size) for name in self.features
        }
        self._n_observed = 0

        # Check the band against this envelope's own null before accepting any
        # traffic. PSI's 0.10/0.25 bands are rules of thumb quoted without a
        # sample size, and the null grows as roughly (k-1)/n -- at 20 bins and a
        # 200-window, 76% of windows drawn from the reference itself exceed
        # 0.10. A monitor that alarms on stable traffic gets switched off, which
        # is the same outcome as never building it (SPEC.md 5.2: do not revoke
        # on noise).
        self.null_bands: dict[str, NullBand] = {}
        if max_false_alarm_rate is not None:
            for feature in envelope.features:
                if feature.name not in self._windows:
                    continue
                band = simulate_null_psi(
                    feature.bin_probabilities,
                    n=self.window_size,
                    n_reference=envelope.n_reference,
                    stable_band=self.psi_stable,
                )
                self.null_bands[feature.name] = band
                if band.false_alarm_rate > max_false_alarm_rate:
                    raise ValueError(
                        "%s: %d bins against a %d-request window would trip "
                        "psi_stable=%.2f on %.1f%% of windows drawn from the "
                        "reference itself, above the %.1f%% ceiling. The band "
                        "is not scale-free; widen the window, use fewer bins, "
                        "or raise psi_stable deliberately. Null p95 here is "
                        "%.4f."
                        % (
                            feature.name, band.n_bins, self.window_size,
                            self.psi_stable, 100 * band.false_alarm_rate,
                            100 * max_false_alarm_rate, band.p95,
                        )
                    )

    def observe(self, values: Mapping[str, float]) -> None:
        """Record one request's feature values.

        Values for features this monitor does not track are ignored; a request
        that carries none of them still counts toward the window, because the
        window measures traffic volume rather than feature coverage.
        """
        for name, value in values.items():
            if name in self._windows:
                self._windows[name].append(float(value))
        self._n_observed += 1

    def __len__(self) -> int:
        return self._n_observed

    def verdict(self) -> DriftVerdict:
        """Score the current window. Refuses a verdict below the minimum."""
        observed = min(self._n_observed, self.window_size)
        if self._n_observed < self.window_size:
            return DriftVerdict(
                state=EnvelopeState.INSUFFICIENT_DATA,
                n_observed=self._n_observed,
                window_size=self.window_size,
                per_feature={},
                unobserved=self.features,
                driver=None,
                reason=(
                    "%d requests observed against a %d minimum: no verdict. "
                    "Reporting INSIDE here would assert stability that has not "
                    "been measured."
                    % (self._n_observed, self.window_size)
                ),
            )

        per_feature: dict[str, PsiResult] = {}
        unobserved: list[str] = []
        for name in self.features:
            reference = self._reference.get(name)
            window = self._windows.get(name)
            if reference is None or not window:
                unobserved.append(name)
                continue
            per_feature[name] = population_stability_index(reference, list(window))

        if not per_feature:
            return DriftVerdict(
                state=EnvelopeState.INSUFFICIENT_DATA,
                n_observed=self._n_observed,
                window_size=self.window_size,
                per_feature={},
                unobserved=tuple(unobserved),
                driver=None,
                reason=(
                    "%d requests observed but no configured feature could be "
                    "scored (%s). A window with no measurable feature is not a "
                    "window that found nothing wrong."
                    % (self._n_observed, ", ".join(unobserved) or "none configured")
                ),
            )

        states = {
            name: state_for_psi(
                result.psi, stable=self.psi_stable, significant=self.psi_significant
            )
            for name, result in per_feature.items()
        }
        driver = max(states, key=lambda name: (_SEVERITY[states[name]], per_feature[name].psi))
        state = states[driver]

        result = per_feature[driver]
        caveat = (
            " The magnitude is set by the smoothing epsilon -- %d of %d bins "
            "were empty, carrying %.0f%% of the index -- so the finding is that "
            "traffic left the reference support, not that the shift measured "
            "this size."
            % (result.bins_smoothed, len(result.per_bin), 100 * result.smoothed_share)
            if result.driven_by_smoothing
            else ""
        )
        missing = (
            " Not scored: %s." % ", ".join(unobserved) if unobserved else ""
        )
        return DriftVerdict(
            state=state,
            n_observed=self._n_observed,
            window_size=self.window_size,
            per_feature=per_feature,
            unobserved=tuple(unobserved),
            driver=driver,
            reason=(
                "%s: PSI %.4f over %d requests puts the envelope at %s (bands "
                "%.2f / %.2f).%s%s"
                % (
                    driver, result.psi, observed, state.value,
                    self.psi_stable, self.psi_significant, caveat, missing,
                )
            ),
        )
