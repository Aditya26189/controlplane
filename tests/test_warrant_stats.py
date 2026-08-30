"""Certification at issuance, and the anytime-valid revocation trigger.

Gate A of the handoff plan. The plan's premise needed one correction before the
test could be written: it states *"the decision-support profile is at FPR
0.247"*, and this repository has no such profile. `config.yaml` publishes three
operating points at `max_fpr` **0.02 / 0.05 / 0.10**, and 0.247 appears in
`results/` only as an unrelated `ci_low`. That matters, because the NaN the
clamp prevents needs ``lam_hi * p0 > 1`` and 0.10 is exactly the boundary --
see :func:`test_the_clamp_is_load_bearing_above_the_boundary`, which measures
where the failure actually starts rather than asserting where it was predicted.
"""

from __future__ import annotations

import numpy as np
import pytest

from controlplane.config import load_config
from controlplane.validation.warrant_stats import (
    Certification,
    FPRMonitor,
    certify_fpr,
    cp_upper,
    min_n_for,
)

PROJECT_CONFIG = "config.yaml"
ALPHA = 0.05


def _published_operating_points() -> list[tuple[str, float]]:
    """The three profiles' FPR budgets, read from config rather than typed."""
    config = load_config(PROJECT_CONFIG)
    return [(name, float(p.max_fpr)) for name, p in config.profiles.items()]


# --------------------------------------------------------------------------- #
# Gate A -- the monitor must revoke at every published operating point
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name,p0", _published_operating_points())
def test_a_three_times_breach_revokes_at_every_published_point(
    name: str, p0: float
) -> None:
    """A 3x breach must revoke, with finite wealth, at all three profiles.

    This is the property the whole module exists for. A monitor that returns
    non-finite wealth does not raise and does not revoke -- ``nan >= 20.0`` is
    False -- so it fails by going quiet, which is the one failure mode a
    revocation trigger must not have.
    """
    monitor = FPRMonitor(p0=p0, alpha=ALPHA)
    rng = np.random.default_rng(1729)
    breach = min(0.95, 3.0 * p0)

    for fired in (rng.random(1000) < breach).astype(int):
        monitor.update(int(fired))

    assert np.isfinite(monitor.wealth), (
        f"{name}: wealth is {monitor.wealth} at p0={p0}. A non-finite wealth "
        "never crosses 1/alpha, so the monitor is silently dead."
    )
    assert monitor.revoked, (
        f"{name}: a 3x breach ({breach:.3f} against p0={p0}) did not revoke "
        f"after 1000 items; wealth={monitor.wealth:.4f}"
    )


@pytest.mark.parametrize("name,p0", _published_operating_points())
def test_the_null_does_not_revoke_at_every_published_point(
    name: str, p0: float
) -> None:
    """The companion. A test that only checks firing passes on a stuck monitor.

    Traffic exactly at ``p0`` must usually NOT revoke -- that is the alpha
    budget. Without this, a monitor hardcoded to ``revoked = True`` would pass
    the breach test above.
    """
    revocations = 0
    for seed in range(40):
        monitor = FPRMonitor(p0=p0, alpha=ALPHA)
        rng = np.random.default_rng(seed)
        for fired in (rng.random(500) < p0).astype(int):
            monitor.update(int(fired))
        revocations += int(monitor.revoked)

    assert revocations <= 8, (
        f"{name}: {revocations}/40 null streams revoked at p0={p0}. Ville's "
        "inequality budgets alpha=0.05 over the whole deployment."
    )


def test_the_clamp_is_load_bearing_above_the_boundary() -> None:
    """The regression, in both directions -- Gate A's 'confirm both' clause.

    ``log1p(lam * (x - p0))`` needs ``lam * p0 < 1``. The unclamped default
    ``lam_hi=10.0`` breaks that for any ``p0 > 0.1``. Measured, driving a 3x
    breach through the raw arithmetic:

        p0=0.02   lam*p0=0.20   fine
        p0=0.05   lam*p0=0.50   fine
        p0=0.10   lam*p0=1.00   -inf on the largest bet, mixture still valid
        p0=0.15   lam*p0=1.50   NaN, wealth NaN, never revokes
        p0=0.247  lam*p0=2.47   NaN, wealth NaN, never revokes

    So the clamp is not currently reachable through a shipped profile -- 0.10
    is the boundary and this repo goes no higher. It is a latent defect that
    any future profile above 0.10 would hit, and this test pins the boundary
    so that adding such a profile cannot reintroduce it silently.
    """
    p0 = 0.15
    lams = np.geomspace(0.05, 10.0, 40)  # the UNCLAMPED grid
    assert lams.max() * p0 > 1.0, "this test no longer exercises the defect"

    logw = np.zeros(40)
    rng = np.random.default_rng(3)
    with np.errstate(invalid="ignore"):
        for fired in (rng.random(300) < 0.45).astype(float):
            logw = logw + np.log1p(lams * (fired - p0))

    assert not np.isfinite(logw).all(), (
        "the unclamped grid no longer produces non-finite wealth; this test "
        "has stopped testing anything"
    )
    # And the silent part: NaN never crosses the threshold.
    assert not (np.nan >= 1.0 / ALPHA)

    # The clamped monitor, same p0, same breach, revokes.
    monitor = FPRMonitor(p0=p0, alpha=ALPHA)
    assert monitor.lam_clamped, "lam_hi should have been clamped at p0=0.15"
    assert monitor.lams.max() * p0 < 1.0
    rng = np.random.default_rng(3)
    for fired in (rng.random(300) < 0.45).astype(int):
        monitor.update(int(fired))
    assert np.isfinite(monitor.wealth) and monitor.revoked


def test_a_non_finite_update_raises_rather_than_asserting() -> None:
    """``python -O`` strips asserts. A guard that vanishes under a flag is not a guard."""
    monitor = FPRMonitor(p0=0.02, alpha=ALPHA)
    monitor.lams = np.array([50.0])          # 50 * 0.02 = 1.0 exactly
    monitor._logw = np.zeros(1)
    with pytest.raises(FloatingPointError, match="non-finite wealth"):
        monitor.update(0)


def test_p0_outside_the_unit_interval_is_refused() -> None:
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="p0 must be in"):
            FPRMonitor(p0=bad)


# --------------------------------------------------------------------------- #
# Resuming a monitor -- the grid must travel with the wealth
# --------------------------------------------------------------------------- #


def test_resuming_reproduces_the_monitor_exactly() -> None:
    monitor = FPRMonitor(p0=0.02, alpha=ALPHA, lam_lo=0.05, lam_hi=2.0, n_lam=40)
    rng = np.random.default_rng(11)
    for fired in (rng.random(200) < 0.05).astype(int):
        monitor.update(int(fired))

    resumed = FPRMonitor.from_state(monitor.state())
    assert np.allclose(resumed.lams, monitor.lams)
    assert np.allclose(resumed._logw, monitor._logw)
    assert (resumed.n, resumed.k) == (monitor.n, monitor.k)
    assert resumed.wealth == pytest.approx(monitor.wealth)


def test_a_state_without_its_betting_grid_cannot_be_resumed() -> None:
    """Rebuilding from defaults reattaches every bet to the wrong lambda.

    ``_logw[i]`` is the log wealth of the bet at ``lams[i]``. Restoring the
    vector against a grid built from default bounds passes the shape check --
    same length, same dtype, nothing raised -- and the monitor carries on with
    every number misattributed. Measured across 400 resumed monitors, the
    revocation decision differed in 41 of them, and it differed in the
    direction of REVOKING when the correctly-resumed monitor did not, which is
    the false-revocation guarantee this module exists to provide.
    """
    monitor = FPRMonitor(p0=0.02, alpha=ALPHA, lam_lo=0.05, lam_hi=2.0, n_lam=40)
    monitor.update(1)
    state = monitor.state()

    for dropped in ("lam_lo", "lam_hi", "n_lam"):
        crippled = {k: v for k, v in state.items() if k != dropped}
        with pytest.raises(ValueError, match="missing the betting grid"):
            FPRMonitor.from_state(crippled)


def test_the_recorded_grid_is_the_one_that_was_used() -> None:
    """A narrow grid must survive the round trip rather than widening to defaults."""
    monitor = FPRMonitor(p0=0.02, alpha=ALPHA, lam_lo=0.05, lam_hi=2.0, n_lam=40)
    resumed = FPRMonitor.from_state(monitor.state())
    assert resumed.lams.max() == pytest.approx(2.0)
    default = FPRMonitor(p0=0.02, alpha=ALPHA)
    assert default.lams.max() != pytest.approx(2.0), (
        "defaults coincide with the test grid; this test proves nothing"
    )


# --------------------------------------------------------------------------- #
# Certification at issuance
# --------------------------------------------------------------------------- #


def test_the_price_of_the_flagship_profile() -> None:
    """199 clean negatives for one profile, 290 across four.

    This is the number the demo's opening beat prints next to the refusal.
    """
    assert min_n_for(0.015, ALPHA) == 199
    assert min_n_for(0.015, ALPHA / 4) == 290


def test_a_zero_event_bound_is_clopper_pearson_not_a_bootstrap() -> None:
    """A bootstrap on all-zero data collapses to [0, 0] and claims certainty."""
    assert cp_upper(0, 200, ALPHA) == pytest.approx(1.0 - ALPHA ** (1 / 200))
    assert cp_upper(0, 200, ALPHA) > 0.0
    bootstrap_on_zeros = 0.0
    assert cp_upper(0, 200, ALPHA) > bootstrap_on_zeros


def test_cp_upper_is_non_increasing_in_the_sweep() -> None:
    """The monotonicity the no-multiplicity argument rests on."""
    bounds = [cp_upper(k, 300, ALPHA) for k in range(0, 30)]
    assert bounds == sorted(bounds), "cp_upper must be non-decreasing in k"


def test_too_few_negatives_refuses_and_names_its_price() -> None:
    rng = np.random.default_rng(5)
    result = certify_fpr(rng.normal(size=120), target_fpr=0.015, alpha=ALPHA)
    assert isinstance(result, Certification)
    assert result.fpr_certified is False
    assert result.threshold is None and result.ucb is None
    assert result.k is None, "k must be None when no threshold was evaluated"
    assert "199" in result.reason, result.reason


def test_bonferroni_raises_the_price_across_profiles() -> None:
    rng = np.random.default_rng(5)
    scores = rng.normal(size=250)
    one = certify_fpr(scores, target_fpr=0.015, alpha=ALPHA, n_profiles=1)
    four = certify_fpr(scores, target_fpr=0.015, alpha=ALPHA, n_profiles=4)
    assert one.fpr_certified and not four.fpr_certified, (
        "250 units certifies one profile (needs 199) and must not certify four "
        "(needs 290)"
    )


def test_unequal_clusters_are_refused_not_averaged() -> None:
    """Cluster FPR dominates item FPR only when clusters are equal-sized."""
    scores = np.concatenate([np.ones(100), np.zeros(99)])
    clusters = np.concatenate([np.zeros(100), np.arange(1, 100)])
    with pytest.raises(ValueError, match="unequal cluster sizes"):
        certify_fpr(scores, target_fpr=0.05, alpha=ALPHA, cluster_ids=clusters)


def test_clustering_costs_effective_sample_size() -> None:
    """400 items in 200 equal clusters certify as 200 units, not 400."""
    rng = np.random.default_rng(9)
    scores = rng.normal(size=400)
    clusters = np.repeat(np.arange(200), 2)
    result = certify_fpr(scores, target_fpr=0.02, alpha=ALPHA, cluster_ids=clusters)
    assert result.n_eff == 200


def test_certification_is_an_fpr_claim_and_says_nothing_about_recall() -> None:
    """A detector that fires on nothing certifies trivially.

    The field is named ``fpr_certified`` precisely so it cannot be read as a
    green light. This test exists to keep that name honest: if someone renames
    it back to ``certified``, this is the failure that explains why not.
    """
    inert = np.full(300, -5.0)  # scores nothing; a threshold above them all
    result = certify_fpr(inert, target_fpr=0.02, alpha=ALPHA)
    assert result.fpr_certified is True
    assert result.k == 0
    assert hasattr(result, "fpr_certified") and not hasattr(result, "certified")


# --------------------------------------------------------------------------- #
# A4 -- zero-event intervals. DECISIONS 108.
# --------------------------------------------------------------------------- #


def test_precision_is_absent_not_zero_when_nothing_is_flagged() -> None:
    """0/0 is undefined. Reporting it as 0.0 with a [0, 0] interval is a claim
    of perfect certainty about a ratio that has no denominator.

    This shipped in results/transfer-T1-mean_pool.json -- the documented
    mean-pool long-context collapse -- as precision 0.0, ci [0.0, 0.0],
    estimator 'bootstrap-percentile-1000'. The bootstrap resampled all-zero
    data, and the Clopper-Pearson fallback that rescues every other zero-event
    quantity could not fire, because a binomial interval needs trials > 0.
    """
    import numpy as np

    from controlplane.config import load_config
    from controlplane.validation.metrics_builder import build_warrant_metrics

    config = load_config(PROJECT_CONFIG)
    rng = np.random.default_rng(0)
    labels = (rng.random(200) < 0.4).astype(int)
    scores = rng.random(200)

    metrics = build_warrant_metrics(
        labels=labels, scores=scores, threshold=2.0, config=config  # flags nothing
    )
    assert metrics.precision is not None
    assert (metrics.precision.ci_low, metrics.precision.ci_high) == (0.0, 1.0), (
        f"precision interval is [{metrics.precision.ci_low}, "
        f"{metrics.precision.ci_high}]; with nothing flagged we know NOTHING "
        "about precision and the honest interval is the vacuous [0, 1]"
    )
    assert "undefined" in metrics.precision.estimator
    # AUROC and recall are NOT sacrificed to fix precision. On the artifact this
    # was found in, AUROC is 0.5015 -- the mean-pool collapse itself.
    assert metrics.auroc is not None and metrics.recall is not None
    # The quantities that ARE defined still carry exact intervals.
    assert metrics.flag_rate is not None and metrics.flag_rate.ci_high > 0.0
    assert "Clopper-Pearson" in metrics.flag_rate.estimator


def test_a_zero_width_rate_interval_over_zero_trials_is_refused() -> None:
    """The guard, at the layer that would otherwise emit it silently."""
    import numpy as np

    from controlplane.validation.stats import MeasurementError, estimated

    labels = np.zeros(50, dtype=int)
    scores = np.zeros(50)
    with pytest.raises(MeasurementError, match="undefined at this threshold"):
        estimated(
            "precision",
            lambda y, s: 0.0,
            labels,
            scores,
            n_resamples=50,
            ci=0.95,
            seed=1729,
            binomial_events=0,
            binomial_trials=0,
        )


def test_no_committed_artifact_carries_a_zero_width_rate_interval() -> None:
    """The sweep that found it, kept as a check.

    A rate whose interval is [0, 0] is a point estimate wearing an interval's
    clothes, and CLAUDE.md's fourth invariant says no point estimate reaches a
    user. Artifacts predating DECISIONS 108 are regenerated, not grandfathered.
    """
    import glob
    import json
    import os

    offenders = []

    def walk(node, artifact):
        if isinstance(node, dict):
            if (
                node.get("unit") == "rate"
                and node.get("ci_low") == 0.0
                and node.get("ci_high") == 0.0
            ):
                offenders.append(f"{artifact}: {node.get('name')} n={node.get('n')}")
            for v in node.values():
                walk(v, artifact)
        elif isinstance(node, list):
            for v in node:
                walk(v, artifact)

    for path in glob.glob("results/**/*.json", recursive=True):
        if "kaggle" in path:
            continue  # a downloaded run's own outputs, not this repo's claims
        try:
            with open(path, encoding="utf-8") as handle:
                walk(json.load(handle), os.path.basename(path))
        except Exception:
            continue

    assert not offenders, (
        "zero-width rate intervals in committed artifacts:\n  "
        + "\n  ".join(sorted(set(offenders)))
    )
