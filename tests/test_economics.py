"""The abstention floor and the review sizing. ``DECISIONS.md`` 099.

Two things worth checking harder than usual.

**The bound is checked against a simulation, not only against itself.** A
closed form asserted by the test that also computes it proves nothing. The
floor is a claim about what no selector can do, so it is checked against an
exhaustively-searched perfect selector on a concrete population.

**Measured and declared inputs must stay separated.** The whole reason this
package exists in the state it does is that the price list was never built and
every money figure is therefore a declared estimate. If a derived quantity
stopped saying which of its inputs came from an artifact and which from
``config.yaml``, the distinction the proposal rests on would quietly go.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from controlplane.config import Config
from controlplane.economics import (
    abstention_floor,
    achieved_risk,
    feasibility_curve,
    recall_sample_size,
    review_volume,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = PROJECT_ROOT / "results" / "feasibility.json"


# --------------------------------------------------------------------------- #
# The bound
# --------------------------------------------------------------------------- #


def test_the_floor_matches_an_exhaustive_perfect_selector() -> None:
    """The closed form against a simulated optimal selector on 1000 items.

    A perfect selector abstains on errors first, since abstaining on a correct
    response costs coverage and buys no risk reduction. So the minimum
    abstention achieving a target is found by abstaining on errors one at a
    time until the residual risk drops below alpha. This walks that directly
    and compares.
    """
    n = 1000
    for mu in (0.10, 0.25, 0.4510, 0.80):
        errors = round(mu * n)
        for alpha in (0.01, 0.05, 0.20, 0.40):
            simulated = None
            for abstained in range(n + 1):
                # Best case: every abstained item is an error, until they run out.
                remaining_errors = max(0, errors - abstained)
                retained = n - abstained
                if retained == 0:
                    simulated = abstained / n
                    break
                if remaining_errors / retained <= alpha:
                    simulated = abstained / n
                    break
            derived = abstention_floor(mu, alpha).floor
            assert simulated is not None
            assert abs(derived - simulated) <= 1.5 / n, (
                f"mu={mu} alpha={alpha}: closed form {derived:.4f}, "
                f"simulated optimum {simulated:.4f}"
            )


def test_no_abstention_is_forced_when_the_traffic_already_meets_the_target() -> None:
    floor = abstention_floor(0.02, 0.05)
    assert floor.floor == 0.0
    assert floor.binding is False
    assert "already meets" in floor.render()


def test_the_floor_rises_as_the_target_tightens() -> None:
    """The shape is the argument: tightening the target costs coverage, fast."""
    curve = feasibility_curve(0.4510, [0.40, 0.20, 0.10, 0.05, 0.01])
    floors = [f.floor for f in curve]
    assert floors == sorted(floors), "the floor must be monotone in the target"
    assert floors[0] < floors[-1]


def test_the_floor_approaches_the_base_rate_as_the_target_approaches_zero() -> None:
    assert abstention_floor(0.4510, 0.0).floor == pytest.approx(0.4510)


def test_a_rate_outside_the_unit_interval_is_refused() -> None:
    """Clamping would produce a plausible bound from an impossible input."""
    with pytest.raises(ValueError, match="not a rate"):
        abstention_floor(1.5, 0.05)
    with pytest.raises(ValueError, match="not a rate"):
        abstention_floor(0.4, -0.1)


def test_a_target_of_one_is_reported_unattainable_rather_than_free() -> None:
    """"risk <= 1" is not a target, and 0.0 alone would read as a result."""
    floor = abstention_floor(0.4510, 1.0)
    assert floor.floor == 0.0
    assert floor.attainable is False


# --------------------------------------------------------------------------- #
# Achieved risk -- fully measured, no declared inputs
# --------------------------------------------------------------------------- #


def test_achieved_risk_is_the_error_rate_among_what_is_not_flagged() -> None:
    """Checked by counting, not by re-deriving the same formula.

    1000 items, base rate 0.40 -> 400 errors. Recall 0.75 -> 300 caught, 100
    missed. Flag rate 0.50 -> 500 flagged, 500 retained. Residual risk is the
    100 missed errors among the 500 retained.
    """
    result = achieved_risk(
        operating_point_id="P-test",
        base_error_rate=0.40,
        recall=0.75,
        flag_rate=0.50,
    )
    assert result.residual_risk == pytest.approx(100 / 500)


def test_a_perfect_selector_scores_an_efficiency_of_one() -> None:
    """The ratio must bottom out at 1.0, or it is not measuring distance to a floor.

    A selector that flags exactly the errors and nothing else: base rate 0.30,
    recall 1.0, flag rate 0.30. It ships zero residual risk, and the floor for
    zero residual risk is the base rate itself.
    """
    result = achieved_risk(
        operating_point_id="P-perfect",
        base_error_rate=0.30,
        recall=1.0,
        flag_rate=0.30,
    )
    assert result.residual_risk == pytest.approx(0.0)
    assert result.efficiency == pytest.approx(1.0)


def test_efficiency_is_never_below_one_for_an_achievable_point() -> None:
    """A point beating the floor would mean the floor is wrong.

    Swept rather than spot-checked, and it earned that: the sweep produced
    recall=0.3 at flag_rate=0.1 on a 0.451 base rate, which scored 0.65 --
    apparently beating a bound no selector can beat. The combination is
    impossible (it catches more errors than it flags) and achieved_risk now
    refuses it, which is the finding rather than a workaround.
    """
    mu = 0.4510
    checked = 0
    for recall in (0.1, 0.3, 0.5, 0.75, 0.9):
        for flag_rate in (0.1, 0.25, 0.5, 0.7):
            if mu * recall > flag_rate:
                continue  # infeasible; covered by the test below
            result = achieved_risk(
                operating_point_id="P",
                base_error_rate=mu,
                recall=recall,
                flag_rate=flag_rate,
            )
            checked += 1
            if math.isfinite(result.efficiency):
                assert result.efficiency >= 1.0 - 1e-9, (
                    f"recall={recall} f={flag_rate} scored {result.efficiency:.4f}, "
                    "which is below the theoretical floor"
                )
    assert checked >= 10, "the sweep skipped almost everything"


def test_catching_more_errors_than_were_flagged_is_refused() -> None:
    """The rates must describe one envelope, or the bound is meaningless.

    mu * recall is the share of all traffic that is a caught error; it cannot
    exceed the share flagged. When it did, the efficiency came out below 1.0 --
    an operating point apparently beating a bound no selector can beat. That is
    the signature of rates taken from different measurements.
    """
    with pytest.raises(ValueError, match="cannot describe one envelope"):
        achieved_risk(
            operating_point_id="P",
            base_error_rate=0.4510,
            recall=0.3,
            flag_rate=0.1,
        )


def test_the_three_measured_operating_points_are_internally_consistent() -> None:
    """The real ones pass the check the synthetic sweep failed."""
    for recall, flag_rate in (
        (0.21709006928406466, 0.10625),
        (0.36027713625866054, 0.18645833333333334),
        (0.7367205542725174, 0.46770833333333334),
    ):
        result = achieved_risk(
            operating_point_id="P",
            base_error_rate=0.4510416666666667,
            recall=recall,
            flag_rate=flag_rate,
        )
        assert result.efficiency >= 1.0


def test_flagging_everything_is_refused_rather_than_scored() -> None:
    with pytest.raises(ValueError, match="retains nothing"):
        achieved_risk(
            operating_point_id="P", base_error_rate=0.4, recall=1.0, flag_rate=1.0
        )


# --------------------------------------------------------------------------- #
# Review sizing -- measured and declared, kept apart
# --------------------------------------------------------------------------- #


def test_review_volume_labels_which_inputs_were_declared() -> None:
    volume = review_volume(
        flag_rate=0.10625,
        recall=0.21709006928406466,
        base_error_rate=0.03,
        monthly_interactions=200_000,
        operating_point_id="P-customer-support",
    )
    flagged = volume["flagged"]
    assert flagged.value == pytest.approx(21_250)
    assert set(flagged.measured_inputs) == {"flag_rate", "recall"}
    assert set(flagged.declared_inputs) == {
        "monthly_interactions",
        "base_error_rate",
    }
    assert "declared workload" in flagged.note


def test_the_parts_of_the_flagged_pool_add_up() -> None:
    volume = review_volume(
        flag_rate=0.20, recall=0.50, base_error_rate=0.03,
        monthly_interactions=100_000,
    )
    assert volume["true_positives"].value + volume["false_positives"].value == (
        pytest.approx(volume["flagged"].value)
    )


def test_rates_from_the_wrong_workload_are_flagged_not_silently_negative() -> None:
    """A recall and a base rate from different distributions can disagree.

    Flag 1% of traffic but claim to catch 100% of a 30% error rate, and the
    implied true positives exceed the flagged total. That is scenario mixing,
    and it must surface rather than produce a negative false-positive count.
    """
    volume = review_volume(
        flag_rate=0.01, recall=1.0, base_error_rate=0.30,
        monthly_interactions=100_000,
    )
    assert volume["false_positives"].value == 0.0
    assert "WARNING" in volume["false_positives"].note


def test_recall_sample_size_is_sized_at_the_measured_recall() -> None:
    """Sizing at p=0.5 would overstate the requirement by nearly threefold."""
    at_measured = recall_sample_size(recall=0.2171, margin=0.02)
    at_half = recall_sample_size(recall=0.5, margin=0.02)
    assert at_measured.value < at_half.value
    assert at_half.value / at_measured.value > 1.4


def test_a_tighter_margin_costs_quadratically() -> None:
    coarse = recall_sample_size(recall=0.2171, margin=0.05)
    fine = recall_sample_size(recall=0.2171, margin=0.02)
    assert fine.value / coarse.value == pytest.approx((0.05 / 0.02) ** 2, rel=0.02)


def test_a_non_positive_margin_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        recall_sample_size(recall=0.3, margin=0.0)


# --------------------------------------------------------------------------- #
# The committed artifact
# --------------------------------------------------------------------------- #


def test_the_feasibility_artifact_reproduces(config: Config) -> None:
    """Every number in results/feasibility.json recomputes from its inputs."""
    if not ARTIFACT.is_file():
        pytest.skip("results/feasibility.json is not committed")
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    mu = payload["measured"]["base_error_rate"]

    for row in payload["abstention_floor"]:
        expected = abstention_floor(mu, row["target_risk"]).floor
        assert row["floor"] == pytest.approx(expected)

    for profile in payload["profiles"]:
        recomputed = achieved_risk(
            operating_point_id=profile["operating_point_id"],
            base_error_rate=mu,
            recall=profile["measured_recall"],
            flag_rate=profile["measured_flag_rate"],
        )
        recorded = profile["achieved_risk"]
        assert recorded["residual_risk"] == pytest.approx(recomputed.residual_risk)
        assert recorded["efficiency"] == pytest.approx(recomputed.efficiency)


def test_the_artifact_states_what_it_does_not_derive() -> None:
    """The gap must travel with the numbers, or it will be forgotten.

    Anyone reading feasibility.json is one step from writing a cost figure. The
    artifact says, in itself, that no cost figure is derived here and that the
    price list is not built.
    """
    if not ARTIFACT.is_file():
        pytest.skip("results/feasibility.json is not committed")
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    note = payload["not_derived_here"]
    assert "not built" in note
    assert "declared estimate" in note
