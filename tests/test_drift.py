"""PSI against a warrant's stored bins, and the honesty flag on top of it."""

from __future__ import annotations

import numpy as np
import pytest

from controlplane.drift import population_stability_index, state_for_psi
from controlplane.model.enums import EnvelopeState
from controlplane.model.findings import EnvelopeFeature


def _feature(values: np.ndarray, bins: int = 10) -> EnvelopeFeature:
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    probs = np.histogram(values, bins=edges)[0] / len(values)
    return EnvelopeFeature(
        name="token_length",
        bin_edges=tuple(float(x) for x in edges),
        bin_probabilities=tuple(float(x) for x in probs),
        mean=float(values.mean()),
        std=float(values.std()),
    )


def _short() -> np.ndarray:
    rng = np.random.default_rng(1729)
    return rng.normal(45, 12, 600).clip(20, 200)


def _long() -> np.ndarray:
    rng = np.random.default_rng(17)
    return rng.normal(6950, 1800, 600).clip(2800, 11200)


def test_a_sample_against_its_own_reference_is_inside() -> None:
    """The null. A drift monitor that fires on its own reference is noise."""
    values = _short()
    result = population_stability_index(_feature(values), values)
    assert result.psi == pytest.approx(0.0, abs=1e-9)
    assert result.bins_smoothed == 0
    assert state_for_psi(result.psi, stable=0.10, significant=0.25) is EnvelopeState.INSIDE


def test_a_real_shift_revokes() -> None:
    """Long-context traffic against short-context bins is off the support."""
    result = population_stability_index(_feature(_short()), _long())
    assert state_for_psi(
        result.psi, stable=0.10, significant=0.25
    ) is EnvelopeState.SIGNIFICANT_SHIFT


def test_a_psi_driven_by_empty_bins_says_so() -> None:
    """The magnitude is an artefact of the epsilon; the flag is what is real.

    Measured on the real envelopes: long-context traffic has no mass in 9 of
    the 10 short-context bins, and those floored bins carry 84% of the index.
    Across plausible epsilons the value ranges 6.14 (1e-3) to 16.50 (1e-8) --
    a 2.7x swing set entirely by a smoothing constant. Quoting "PSI 12.37"
    would be quoting the epsilon.
    """
    result = population_stability_index(_feature(_short()), _long(), epsilon=1e-6)
    assert result.bins_smoothed >= 8
    assert result.driven_by_smoothing, (
        "a PSI whose magnitude comes from floored bins must announce it; the "
        "honest statement is that traffic left the reference support, not that "
        "the shift measured this large"
    )


def test_a_measured_shift_is_stable_across_epsilons() -> None:
    """The case worth quoting: real overlap, no smoothing, epsilon-invariant.

    This is what a sliding window actually sees mid-shift, and it is detected
    cleanly without any floored bin.
    """
    mixed = np.concatenate([_short()[:300], _long()[:300]])
    feature = _feature(_short())
    values = [
        population_stability_index(feature, mixed, epsilon=eps).psi
        for eps in (1e-3, 1e-4, 1e-6, 1e-8)
    ]
    assert max(values) - min(values) < 1e-9, (
        "a PSI with no smoothed bins must not depend on the smoothing constant"
    )
    result = population_stability_index(feature, mixed)
    assert result.bins_smoothed == 0
    assert not result.driven_by_smoothing
    assert state_for_psi(
        result.psi, stable=0.10, significant=0.25
    ) is EnvelopeState.SIGNIFICANT_SHIFT


def test_values_outside_the_reference_range_are_not_dropped() -> None:
    """Traffic off the reference support is the clearest drift signal there is.

    Discarding out-of-range values would compute PSI over the surviving subset
    and report stability for traffic that had left the distribution entirely.
    """
    feature = _feature(_short())
    far = np.full(600, 50_000.0)
    result = population_stability_index(feature, far)
    assert result.n_live == 600, "out-of-range values were dropped"
    assert state_for_psi(
        result.psi, stable=0.10, significant=0.25
    ) is EnvelopeState.SIGNIFICANT_SHIFT


def test_an_empty_window_raises_rather_than_reporting_stability() -> None:
    """A PSI of 0.0 on no data would read as "no drift" (``DECISIONS.md`` 050)."""
    with pytest.raises(ValueError, match="empty live window"):
        population_stability_index(_feature(_short()), [])


def test_the_significant_boundary_revokes() -> None:
    """At the threshold the conservative rung is the correct one."""
    assert state_for_psi(0.25, stable=0.10, significant=0.25) is EnvelopeState.SIGNIFICANT_SHIFT
    assert state_for_psi(0.10, stable=0.10, significant=0.25) is EnvelopeState.MODERATE_SHIFT
    assert state_for_psi(0.0999, stable=0.10, significant=0.25) is EnvelopeState.INSIDE


def test_non_finite_psi_revokes() -> None:
    """An unrepresentable index is not an excuse to keep certifying."""
    assert state_for_psi(
        float("inf"), stable=0.10, significant=0.25
    ) is EnvelopeState.SIGNIFICANT_SHIFT
    assert state_for_psi(
        float("nan"), stable=0.10, significant=0.25
    ) is EnvelopeState.SIGNIFICANT_SHIFT


# --------------------------------------------------------------------------- #
# The band is not scale-free, and the monitor checks it before accepting traffic
# --------------------------------------------------------------------------- #


def _envelope(n_bins: int, n_reference: int):
    from controlplane.model.findings import DistributionEnvelope

    return DistributionEnvelope(
        envelope_id="env", eval_set_id="e", n_reference=n_reference,
        features=(EnvelopeFeature(
            name="token_length",
            bin_edges=tuple(float(i) for i in range(n_bins + 1)),
            bin_probabilities=tuple([1.0 / n_bins] * n_bins),
            mean=1.0, std=1.0),),
        data_source="measured",
    )


def test_the_null_band_grows_with_bins_and_shrinks_with_window() -> None:
    """PSI's 0.10/0.25 bands are quoted without a sample size, and need one.

    The null is roughly ``(k-1)/n``, so at twenty bins and a 200-request window
    most windows drawn from the reference itself exceed 0.10. Measured here
    rather than asserted, because that is what makes the guard's threshold
    defensible.
    """
    from controlplane.drift import simulate_null_psi

    narrow = simulate_null_psi([0.125] * 8, n=200, n_reference=2400, stable_band=0.10)
    wide = simulate_null_psi([0.05] * 20, n=200, n_reference=600, stable_band=0.10)
    long_window = simulate_null_psi([0.05] * 20, n=2000, n_reference=600, stable_band=0.10)

    assert narrow.false_alarm_rate < 0.05
    assert wide.false_alarm_rate > 0.5, (
        "twenty bins against a 200-request window should alarm on most stable "
        "windows; if it does not, the null simulation is wrong"
    )
    assert long_window.false_alarm_rate < wide.false_alarm_rate


def test_the_reference_is_resampled_not_held_exact() -> None:
    """Reference estimation error is part of the null.

    Holding the stored probabilities exact put the false-alarm rate at 0.02
    where resampling the underlying values measured 0.08. A guard built on the
    optimistic number passes a configuration that alarms four times as often as
    it promised, so both sides are drawn.
    """
    from controlplane.drift import simulate_null_psi

    small_reference = simulate_null_psi([0.1] * 10, n=200, n_reference=300, stable_band=0.10)
    large_reference = simulate_null_psi([0.1] * 10, n=200, n_reference=100_000, stable_band=0.10)
    assert small_reference.p95 > large_reference.p95, (
        "a reference estimated from fewer items must widen the null; if it does "
        "not, the reference is being treated as exact"
    )


def test_a_configuration_that_alarms_on_its_own_reference_is_refused(monkeypatch) -> None:
    """A monitor that cries wolf gets switched off, which is the same as absent."""
    from controlplane.drift import DriftMonitor

    with pytest.raises(ValueError, match="not scale-free"):
        DriftMonitor(
            _envelope(n_bins=20, n_reference=600),
            window_size=200, psi_stable=0.10, psi_significant=0.25,
            features=["token_length"], max_false_alarm_rate=0.05,
        )


def test_the_shipped_configuration_passes_its_own_guard() -> None:
    """8 bins against a 2400-item reference at a 200-window: measured 0.5%."""
    from controlplane.drift import DriftMonitor

    monitor = DriftMonitor(
        _envelope(n_bins=8, n_reference=2400),
        window_size=200, psi_stable=0.10, psi_significant=0.25,
        features=["token_length"], max_false_alarm_rate=0.05,
    )
    assert monitor.null_bands["token_length"].false_alarm_rate < 0.05


def test_no_verdict_below_the_window_minimum() -> None:
    """``INSUFFICIENT_DATA`` is a state, not a default to INSIDE."""
    from controlplane.drift import DriftMonitor
    from controlplane.model.enums import EnvelopeState

    monitor = DriftMonitor(
        _envelope(n_bins=8, n_reference=2400),
        window_size=200, psi_stable=0.10, psi_significant=0.25,
        features=["token_length"],
    )
    for _ in range(199):
        monitor.observe({"token_length": 4.0})
    verdict = monitor.verdict()
    assert verdict.state is EnvelopeState.INSUFFICIENT_DATA
    assert not verdict.revokes and not verdict.widens
    assert "would assert stability that has not been measured" in verdict.reason


def test_a_feature_the_traffic_does_not_carry_is_not_scored_as_stable() -> None:
    """Absence is not a measurement (``DECISIONS.md`` 050)."""
    from controlplane.drift import DriftMonitor
    from controlplane.model.enums import EnvelopeState

    monitor = DriftMonitor(
        _envelope(n_bins=8, n_reference=2400),
        window_size=10, psi_stable=0.10, psi_significant=0.25,
        features=["token_length", "script_mix"],
    )
    for _ in range(20):
        monitor.observe({"token_length": 4.0})
    verdict = monitor.verdict()
    assert "script_mix" in verdict.unobserved
    assert verdict.state is not EnvelopeState.INSUFFICIENT_DATA
    assert "Not scored: script_mix" in verdict.reason


# --------------------------------------------------------------------------
# The IQR-ratio null band -- the rule DECISIONS 101 routes the banking pilot on
# --------------------------------------------------------------------------


def test_the_saturation_threshold_regenerates_at_the_size_it_was_derived_for() -> None:
    """0.439 must keep landing just under the simulated p5 at 12 clusters.

    The constant was derived by hand. If the simulation and the constant ever
    come apart, the pilot's routing rule stops being the one DECISIONS 090's
    correction justified, and nothing else in the suite would notice.
    """
    from controlplane.drift.null_band import simulate_null_iqr_ratio
    from controlplane.evalsets.banking import SATURATION_IQR_RATIO

    band = simulate_null_iqr_ratio(
        n_pilot=12, n_reference=960, threshold=SATURATION_IQR_RATIO, repeats=8000
    )
    assert band.p5 == pytest.approx(0.45, abs=0.04)
    assert SATURATION_IQR_RATIO < band.p5, (
        f"the frozen threshold {SATURATION_IQR_RATIO} is above the simulated p5 "
        f"{band.p5:.4f}; it would alarm on more than 5% of healthy pilots"
    )
    assert band.false_alarm_rate == pytest.approx(0.05, abs=0.02)


def test_the_null_band_is_stable_across_score_shapes() -> None:
    """A band that only holds under normality is a band that holds by luck."""
    from controlplane.drift.null_band import simulate_null_iqr_ratio

    p5s = [
        simulate_null_iqr_ratio(
            n_pilot=12, n_reference=960, threshold=0.439, shape=shape, repeats=8000
        ).p5
        for shape in ("normal", "logistic", "beta2_2")
    ]
    assert max(p5s) - min(p5s) < 0.05, f"p5 moves with the score shape: {p5s}"


def test_holding_the_threshold_while_n_grows_loses_power() -> None:
    """The DECISIONS 103 finding, as a test.

    This is counter-intuitive enough that someone will 'fix' the pilot by
    authoring more questions. The null band tightens with n while a frozen
    constant does not, so the rule grows more conservative and less able to
    catch the collapse it exists for -- and the falling false-alarm rate reads
    as rigour while it happens.
    """
    from controlplane.drift.null_band import iqr_ratio_power, simulate_null_iqr_ratio

    frozen = {
        n: iqr_ratio_power(
            n_pilot=n, n_reference=960, threshold=0.439, collapse=0.6, repeats=8000
        )
        for n in (12, 30, 60)
    }
    assert frozen[60] < frozen[30] < frozen[12], (
        f"frozen 0.439 did not lose power as n grew: {frozen}"
    )

    # Recalibrating to the p5 at each n reverses it, which is the point.
    recalibrated = {}
    for n in (12, 30, 60):
        p5 = simulate_null_iqr_ratio(
            n_pilot=n, n_reference=960, threshold=0.439, repeats=8000
        ).p5
        recalibrated[n] = iqr_ratio_power(
            n_pilot=n, n_reference=960, threshold=p5, collapse=0.6, repeats=8000
        )
    assert recalibrated[60] > recalibrated[30] > recalibrated[12], (
        f"recalibrating did not gain power as n grew: {recalibrated}"
    )


def test_the_pilot_cannot_resolve_a_mild_collapse_at_twelve() -> None:
    """The resolution statement in DECISIONS 103, asserted rather than asserted-in-prose.

    A ratio above the threshold is the expected outcome under BOTH a healthy
    probe and a moderately collapsed one, which is exactly why the artifact has
    to say so next to the number.
    """
    from controlplane.drift.null_band import iqr_ratio_power

    half = iqr_ratio_power(
        n_pilot=12, n_reference=960, threshold=0.439, collapse=0.5, repeats=8000
    )
    assert 0.40 < half < 0.65, (
        f"P(detect a halved spread at n=12) = {half}; DECISIONS 103 reports ~0.51 "
        "and the claim that the pilot is underpowered rests on it"
    )


def test_a_pilot_too_small_for_an_iqr_is_refused() -> None:
    """Three points have no interquartile range worth the name."""
    from controlplane.drift.null_band import simulate_null_iqr_ratio

    with pytest.raises(ValueError, match="at least 4"):
        simulate_null_iqr_ratio(n_pilot=3, n_reference=960, threshold=0.439)


def test_an_unknown_score_shape_is_refused_rather_than_defaulted() -> None:
    """Silently falling back to normal would make the stability check vacuous."""
    from controlplane.drift.null_band import simulate_null_iqr_ratio

    with pytest.raises(ValueError, match="unknown score shape"):
        simulate_null_iqr_ratio(
            n_pilot=12, n_reference=960, threshold=0.439, shape="cauchy"
        )
