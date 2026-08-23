"""The warrant matrix and routing on it.

``test_three_states`` is the Phase 4 gate test. It asserts what ``CLAUDE.md``
invariant 2 actually requires: that ``UNVALIDATED`` routes conservatively and
enqueues, that ``REFUSED`` removes the detector from service, and that the two
are **not interchangeable** — which is the regression `KICKOFF.md` names as the
most likely design failure in the whole project.
"""

from __future__ import annotations

import dataclasses
from datetime import timedelta
from pathlib import Path

import pytest

from src.config import Config
from src.matrix import MatrixCell, Profile, WarrantMatrix, route
from src.model import Action, WarrantKey, WarrantStatus, utc_now
from src.store import Ledger

from .factories import failing_controls, make_warrant


@pytest.fixture()
def ledger(tmp_path: Path):
    store = Ledger(tmp_path / "matrix.db", retention_days=400)
    yield store
    store.close()


def _matrix_with(*warrants, detectors=None, envelopes=None, now=None) -> WarrantMatrix:
    """Build a matrix directly from warrants, without a ledger round-trip."""
    cells = [WarrantMatrix._cell_for(w, now or utc_now()) for w in warrants]
    return WarrantMatrix(
        cells,
        detectors=detectors or sorted({w.detector_id for w in warrants}),
        envelopes=envelopes or sorted({w.eval_set_id for w in warrants}),
        now=now,
    )


# --------------------------------------------------------------------------- #
# Gate test: three states, three behaviours
# --------------------------------------------------------------------------- #


def test_three_states(config: Config) -> None:
    """``UNVALIDATED`` and ``REFUSED`` lead to different places, and neither is VALID.

    ``CLAUDE.md`` invariant 2. Collapsing ``UNVALIDATED`` into ``REFUSED`` makes
    the system unusable on day one; collapsing it into ``VALID`` is the failure
    the product argues against.
    """
    valid = make_warrant(detector_id="probe-good", eval_set_id="envelope-a")
    refused = make_warrant(
        detector_id="probe-bad",
        eval_set_id="envelope-a",
        controls=failing_controls("null_feature"),
        status=WarrantStatus.REFUSED,
        status_reason="null_feature scored 0.71, outside the band",
    )
    matrix = _matrix_with(
        valid,
        refused,
        detectors=["probe-good", "probe-bad", "probe-never-tried"],
        envelopes=["envelope-a", "envelope-b"],
    )

    # 1. VALID is usable.
    assert matrix.status(valid.key) is WarrantStatus.VALID
    assert valid in matrix.valid_warrants("envelope-a")

    # 2. REFUSED is measured-and-failed: out of service, never a candidate, and
    #    it is NOT reported as unvalidated.
    assert matrix.status(refused.key) is WarrantStatus.REFUSED
    assert refused not in matrix.valid_warrants("envelope-a")
    assert refused.key not in matrix.unvalidated_cells("envelope-a")
    assert matrix.status(refused.key).was_ever_measured_here

    # 3. UNVALIDATED is an absence: never measured here, enqueued, and it is NOT
    #    reported as refused.
    never = WarrantKey("probe-never-tried", "P-unassigned", "envelope-a")
    assert matrix.status(never) is WarrantStatus.UNVALIDATED
    assert matrix.cell(never).warrant is None
    assert never in matrix.unvalidated_cells("envelope-a")
    assert not matrix.status(never).was_ever_measured_here

    # The three are distinct, and none is silently the other.
    assert len({matrix.status(k) for k in (valid.key, refused.key, never)}) == 3


def test_unvalidated_routes_conservatively_and_enqueues(config: Config) -> None:
    """The modal production state: no claim, conservative action, cell queued."""
    matrix = _matrix_with(
        make_warrant(detector_id="probe-good", eval_set_id="envelope-a"),
        detectors=["probe-good", "probe-never-tried"],
        envelopes=["envelope-a", "envelope-b"],
    )
    enqueued: list = []
    decision = route(
        matrix, "envelope-b", Profile.from_config(config, "customer_support"),
        enqueue=enqueued.extend,
    )
    assert not decision.routed
    assert decision.action is Action.ESCALATE
    assert decision.claimed_bounds == {}
    assert not decision.suspended_profile  # nothing was measured here at all
    assert {k.detector_id for k in enqueued} == {"probe-good", "probe-never-tried"}


def test_refused_is_never_a_routing_candidate(config: Config) -> None:
    """Out of service on this envelope until a human revalidates."""
    refused = make_warrant(
        detector_id="probe-bad",
        eval_set_id="envelope-a",
        controls=failing_controls("canary"),
        status=WarrantStatus.REFUSED,
        status_reason="canary recall 0.85, below 1.0",
    )
    matrix = _matrix_with(refused, envelopes=["envelope-a"])
    decision = route(
        matrix, "envelope-a", Profile.from_config(config, "customer_support")
    )
    assert not decision.routed
    assert decision.action is Action.ESCALATE
    # And the decision says WHY, distinguishing it from never having tried.
    verdicts = dict(decision.considered)
    assert "REFUSED on this envelope" in verdicts["probe-bad"]
    assert "canary" in verdicts["probe-bad"]


def test_a_refused_cell_is_not_reported_as_unvalidated(config: Config) -> None:
    """The specific collapse KICKOFF.md warns about, asserted directly."""
    refused = make_warrant(
        detector_id="probe-bad",
        eval_set_id="envelope-a",
        controls=failing_controls(),
        status=WarrantStatus.REFUSED,
        status_reason="label_shuffle out of band",
    )
    matrix = _matrix_with(refused, envelopes=["envelope-a"])
    assert matrix.unvalidated_cells("envelope-a") == ()
    assert matrix.summary()["REFUSED"] == 1
    assert matrix.summary()["UNVALIDATED"] == 0


# --------------------------------------------------------------------------- #
# Cells and expiry
# --------------------------------------------------------------------------- #


def test_an_expired_valid_warrant_reports_stale(config: Config) -> None:
    """Age is a reason to stop relying on a number, independent of drift."""
    now = utc_now()
    aged = make_warrant(issued_at=now - timedelta(hours=48), ttl_hours=24)
    matrix = _matrix_with(aged, now=now)
    assert aged.status is WarrantStatus.VALID
    assert matrix.status(aged.key) is WarrantStatus.STALE
    assert "expired" in matrix.cell(aged.key).reason
    assert matrix.valid_warrants(aged.eval_set_id) == ()


def test_missing_cell_never_raises(config: Config) -> None:
    """'We have not measured this' is an answer, and the modal one."""
    matrix = _matrix_with(make_warrant())
    key = WarrantKey("nobody", "P-none", "nowhere")
    assert matrix.status(key) is WarrantStatus.UNVALIDATED
    assert matrix.cell(key).is_empty


def test_matrix_is_built_from_the_ledger(ledger, config: Config) -> None:
    """The matrix and the audit log cannot disagree, because there is one source."""
    valid = make_warrant(detector_id="probe-a", eval_set_id="envelope-a")
    refused = make_warrant(
        detector_id="probe-b", eval_set_id="envelope-a",
        controls=failing_controls(), status=WarrantStatus.REFUSED,
        status_reason="label_shuffle out of band",
    )
    ledger.append_warrant(valid)
    ledger.append_warrant(refused)

    matrix = WarrantMatrix.from_ledger(
        ledger, detectors=["probe-a", "probe-b", "probe-c"],
        envelopes=["envelope-a", "envelope-b"],
    )
    assert matrix.status(valid.key) is WarrantStatus.VALID
    assert matrix.status(refused.key) is WarrantStatus.REFUSED
    assert matrix.summary()["UNVALIDATED"] == 4  # probe-c x 2, plus a/b on envelope-b


def test_revalidation_shows_the_latest_cell(ledger, config: Config) -> None:
    """Appending rather than replacing keeps the history; the matrix shows now."""
    first = make_warrant(validation_run_id="run-0001")
    ledger.append_warrant(first)
    second = make_warrant(
        validation_run_id="run-0002",
        controls=failing_controls("determinism"),
        status=WarrantStatus.REFUSED,
        status_reason="scores differed between two runs at one seed",
    )
    ledger.append_warrant(second)

    matrix = WarrantMatrix.from_ledger(ledger)
    assert matrix.status(first.key) is WarrantStatus.REFUSED
    assert len(ledger.warrants_for_key(first.key)) == 2


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def test_routing_adopts_the_new_warrant_bounds(config: Config) -> None:
    """The claim follows the detector actually used, not the one displaced."""
    from src.model import Metric, MetricKind

    from .factories import make_metrics

    narrow = make_warrant(detector_id="probe-narrow", eval_set_id="envelope-a")
    wide_metrics = dataclasses.replace(
        make_metrics(),
        recall=Metric("recall", 0.30, MetricKind.ESTIMATED, 600, 0.05, 0.55, 0.95,
                      "rate", "bootstrap"),
    )
    wide = dataclasses.replace(
        make_warrant(detector_id="probe-wide", eval_set_id="envelope-a"),
        metrics=wide_metrics,
    )
    matrix = _matrix_with(narrow, wide, envelopes=["envelope-a"])
    decision = route(
        matrix, "envelope-a", Profile.from_config(config, "customer_support")
    )
    assert decision.routed
    # Tighter interval wins: a narrow interval is a more useful claim than a
    # high midpoint with a wide one.
    assert decision.warrant.detector_id == "probe-narrow"
    assert decision.claimed_bounds["recall"]["ci_low"] == pytest.approx(
        narrow.metrics.recall.ci_low
    )


def test_a_profile_suspends_when_bounds_fall_below_its_minimum(config: Config) -> None:
    """Beat 4, step 5, and the difference between suspension and absence."""
    from src.model import Metric, MetricKind

    from .factories import make_metrics

    collapsed = dataclasses.replace(
        make_warrant(detector_id="probe-collapsed", eval_set_id="envelope-long"),
        metrics=dataclasses.replace(
            make_metrics(),
            recall=Metric("recall", 0.034, MetricKind.ESTIMATED, 600, 0.0, 0.077,
                          0.95, "rate", "bootstrap"),
        ),
    )
    matrix = _matrix_with(collapsed, envelopes=["envelope-long"])
    decision = route(
        matrix, "envelope-long", Profile.from_config(config, "customer_support")
    )
    assert not decision.routed
    assert decision.suspended_profile  # measured here, and not good enough
    assert decision.action is Action.ESCALATE
    assert decision.claimed_bounds == {}
    assert "suspended" in decision.reason
    # The per-detector verdict carries the number, which is what Beat 4 narrates:
    # not "profile suspended" but "the lower bound is 0.0000 against a required 0.1".
    verdicts = dict(decision.considered)
    assert "lower bound is 0.0000" in verdicts["probe-collapsed"]
    assert "requires recall >= 0.1" in verdicts["probe-collapsed"]


def test_suspension_is_distinguishable_from_nothing_measured(config: Config) -> None:
    """Two different failures that must not read the same on screen."""
    from src.model import Metric, MetricKind

    from .factories import make_metrics

    weak = dataclasses.replace(
        make_warrant(detector_id="probe-weak", eval_set_id="envelope-a"),
        metrics=dataclasses.replace(
            make_metrics(),
            recall=Metric("recall", 0.02, MetricKind.ESTIMATED, 600, 0.0, 0.05,
                          0.95, "rate", "bootstrap"),
        ),
    )
    matrix = _matrix_with(
        weak, detectors=["probe-weak"], envelopes=["envelope-a", "envelope-empty"]
    )
    profile = Profile.from_config(config, "customer_support")
    suspended = route(matrix, "envelope-a", profile)
    empty = route(matrix, "envelope-empty", profile)

    assert suspended.suspended_profile and not empty.suspended_profile
    assert suspended.action is empty.action  # both conservative
    assert "suspended" in suspended.reason
    assert "no valid warrant" in empty.reason


def test_profile_compares_against_the_interval_bound_not_the_point(config: Config) -> None:
    """A declared minimum is a guarantee, and a point estimate is not one."""
    from src.model import Metric, MetricKind

    from .factories import make_metrics

    profile = Profile.from_config(config, "customer_support")  # min_recall 0.10
    # Point estimate clears the bar; lower bound does not.
    optimistic = dataclasses.replace(
        make_metrics(),
        recall=Metric("recall", 0.11, MetricKind.ESTIMATED, 600, 0.06, 0.19, 0.95,
                      "rate", "bootstrap"),
    )
    accepted, reason = profile.accepts(optimistic)
    assert not accepted
    assert "lower bound is 0.0600" in reason


def test_profile_refuses_an_envelope_with_no_recall_claim(config: Config) -> None:
    """A single-class envelope supports no recall, so no profile can run on it."""
    from .factories import make_metrics

    profile = Profile.from_config(config, "customer_support")
    accepted, reason = profile.accepts(
        dataclasses.replace(
            make_metrics(), auroc=None, recall=None, precision=None
        )
    )
    assert not accepted
    assert "single-class" in reason


def test_routing_names_everything_it_considered(config: Config) -> None:
    """A decision that only shows its winner is unauditable."""
    valid = make_warrant(detector_id="probe-a", eval_set_id="envelope-a")
    refused = make_warrant(
        detector_id="probe-b", eval_set_id="envelope-a",
        controls=failing_controls(), status=WarrantStatus.REFUSED,
        status_reason="label_shuffle out of band",
    )
    matrix = _matrix_with(valid, refused, envelopes=["envelope-a"])
    decision = route(
        matrix, "envelope-a", Profile.from_config(config, "customer_support")
    )
    names = {d for d, _ in decision.considered}
    assert {"probe-a", "probe-b"} <= names


def test_unknown_profile_names_the_ones_that_exist(config: Config) -> None:
    with pytest.raises(KeyError, match="customer_support"):
        Profile.from_config(config, "custommer_support")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def test_every_cell_renders_including_the_empty_ones(config: Config) -> None:
    """The Phase 4 gate: the matrix renders with every cell populated.

    An empty cell renders as ``UNVALIDATED``, not as a blank. A blank reads as
    "nothing to report", and the whole point is that "we have not measured this
    here" *is* the report.
    """
    matrix = _matrix_with(
        make_warrant(detector_id="probe-a", eval_set_id="envelope-a"),
        detectors=["probe-a", "probe-b"],
        envelopes=["envelope-a", "envelope-b"],
    )
    rendered = matrix.render()
    assert rendered.count("UNVALIDATED") >= 3
    assert "| probe-b |" in rendered
    assert "envelope-b" in rendered

    payload = matrix.to_payload()
    assert len(payload["rows"]) == 2
    for row in payload["rows"]:
        assert set(row["cells"]) == {"envelope-a", "envelope-b"}
    assert sum(payload["summary"].values()) == 4


def test_render_shows_intervals_never_bare_point_estimates(config: Config) -> None:
    """Invariant 4 at the surface a judge reads."""
    matrix = _matrix_with(make_warrant())
    rendered = matrix.render()
    assert "R=" in rendered and "[" in rendered and "]" in rendered
