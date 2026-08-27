"""The two-pane demo: real validation, real certificates, real chain.

The demo's value depends entirely on it not being a mock-up. These tests assert
that what the right pane displays is the same object the ledger stores, and that
the left pane's missing bounds are missing because a bare detector has none —
not because the demo withholds them to win an argument.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Config
from src.demo.session import DemoSession
from src.demo.stream import Stream, record_stream
from src.model import Action, EnvelopeState, WarrantStatus
from src.store import Ledger, RecordKind
from src.validation.evalsets import TEST, split_by_question
from src.validation.synthetic import synthetic_cache, synthetic_evalset


@pytest.fixture(scope="module")
def demo_evalset():
    return synthetic_evalset(
        eval_set_id="triviaqa-600-synthetic", n_items=800, base_rate=0.152,
        seed=1729, items_per_question=2, declare_splits=True,
    )


@pytest.fixture(scope="module")
def demo_cache(demo_evalset, config: Config):
    return synthetic_cache(
        demo_evalset, seed=config.seed,
        window=config.probe.rolling_window, stride=config.probe.rolling_stride,
    )


@pytest.fixture(scope="module")
def canary_cache(demo_cache, config: Config):
    """Unambiguous positives the detector must always catch.

    Strong signal and almost no amplitude spread, which is what "unambiguous"
    means operationally: a canary that the detector can plausibly miss is a
    tripwire that fires on noise.
    """
    canary_evalset = synthetic_evalset(
        eval_set_id="canary-20-synthetic", n_items=20, base_rate=0.95, seed=7
    )
    return synthetic_cache(
        canary_evalset, seed=7,
        window=config.probe.rolling_window, stride=config.probe.rolling_stride,
        signal_by_tier={v: 8.0 for v in demo_cache.variants},
        amplitude_spread=0.05,
    )


@pytest.fixture()
def session(demo_evalset, demo_cache, canary_cache, config: Config, tmp_path: Path):
    ledger = Ledger(tmp_path / "demo.db", retention_days=config.store.retention_days)
    demo = DemoSession(
        config, demo_evalset, demo_cache,
        variant="T1-max_rolling_means",
        detector_id="probe-demo", detector_version="0.1.0+fixture",
        ledger=ledger,
        canary_cache=canary_cache,
    )
    demo.prepare()
    assert demo.warrant.status is WarrantStatus.VALID, demo.warrant.status_reason
    yield demo
    ledger.close()


@pytest.fixture()
def refused_session(demo_evalset, demo_cache, config: Config, tmp_path: Path):
    """A session whose warrant was refused, because no canary set was supplied.

    Not a contrived failure: an absent canary set is exactly the situation the
    canary control refuses to wave through, and the demo has to render that
    honestly rather than falling back to showing bounds it cannot support.
    """
    ledger = Ledger(tmp_path / "refused.db", retention_days=config.store.retention_days)
    demo = DemoSession(
        config, demo_evalset, demo_cache,
        variant="T1-max_rolling_means",
        detector_id="probe-demo", detector_version="0.1.0+fixture",
        ledger=ledger,
    )
    demo.prepare()
    yield demo
    ledger.close()


# --------------------------------------------------------------------------- #
# The stream
# --------------------------------------------------------------------------- #


def test_stream_draws_only_from_test_rows(demo_evalset, demo_cache, config: Config) -> None:
    """The demo shows the system on data the probe was not fitted on.

    Anything else and the warrant's numbers would not describe what the audience
    is watching.
    """
    stream = record_stream(demo_evalset, demo_cache, n_events=10, seed=config.seed)
    test_rows = set(split_by_question(demo_evalset, seed=config.seed)[TEST].tolist())
    assert {event.row for event in stream} <= test_rows


def test_stream_round_trips(demo_evalset, demo_cache, config: Config, tmp_path: Path) -> None:
    """One recorded artifact drives both panes, and the backup recording."""
    stream = record_stream(demo_evalset, demo_cache, n_events=6, seed=config.seed)
    restored = Stream.load(stream.save(tmp_path / "stream.json"))
    assert restored == stream
    assert restored.stream_id == stream.stream_id


def test_stream_is_reproducible(demo_evalset, demo_cache, config: Config) -> None:
    a = record_stream(demo_evalset, demo_cache, n_events=6, seed=config.seed)
    b = record_stream(demo_evalset, demo_cache, n_events=6, seed=config.seed)
    assert a.stream_id == b.stream_id


# --------------------------------------------------------------------------- #
# The two panes
# --------------------------------------------------------------------------- #


def test_both_panes_see_the_same_score(session, demo_evalset, demo_cache, config: Config) -> None:
    """The left pane is not a strawman: same detector, same threshold.

    Weakening it would replace an argument with a rigged comparison.
    """
    stream = record_stream(demo_evalset, demo_cache, n_events=5, seed=config.seed)
    for event in stream:
        outcome = session.handle(event)
        assert outcome.left.score == outcome.right.score
        assert outcome.left.flagged == outcome.right.flagged


def test_only_the_right_pane_can_state_what_a_score_is_worth(
    session, demo_evalset, demo_cache, config: Config
) -> None:
    stream = record_stream(demo_evalset, demo_cache, n_events=3, seed=config.seed)
    outcome = session.handle(stream.events[0])

    assert outcome.left.bounds is None
    assert outcome.left.envelope is None
    assert outcome.left.warrant_age is None

    assert outcome.right.bounds
    # Precision and recall travel together (invariant 5).
    assert {"recall", "precision"} <= set(outcome.right.bounds)
    assert outcome.right.bounds["recall"]["ci_low"] < outcome.right.bounds["recall"]["value"]
    assert outcome.right.bounds["recall"]["n"] > 0
    assert outcome.right.envelope is not None


def test_the_banner_never_shows_a_rate_without_its_interval(session) -> None:
    """Invariant 4, at the surface a judge actually reads."""
    banner = session.warrant_banner()
    for field in ("recall", "precision", "auroc", "flag_rate"):
        assert "CI" in banner[field] and "n=" in banner[field], field
    # The yield is exact and must NOT carry one.
    assert "exact" in banner["confirmed_errors"]
    assert "CI" not in banner["confirmed_errors"]


def test_certificates_are_chained_and_readable(
    session, demo_evalset, demo_cache, config: Config
) -> None:
    """What the pane shows is the object the ledger stores."""
    stream = record_stream(demo_evalset, demo_cache, n_events=4, seed=config.seed)
    outcomes = [session.handle(event) for event in stream]

    verification = session.ledger.verify_chain()
    assert verification.ok, verification.breaks
    # One warrant plus one certificate per request.
    assert verification.n_records == 1 + len(outcomes)

    for outcome in outcomes:
        assert outcome.certificate is not None
        stored = session.ledger.get_certificate(outcome.certificate.certificate_id)
        assert stored == outcome.certificate
        assert stored.claimed_bounds == outcome.right.bounds
        assert stored.warrants_relied_upon == (session.warrant.warrant_id,)


def test_a_flagged_request_escalates_and_names_its_finding(
    session, demo_evalset, demo_cache, config: Config
) -> None:
    """An action nobody can trace to a finding is an action nobody can appeal."""
    stream = record_stream(demo_evalset, demo_cache, n_events=40, seed=config.seed)
    outcomes = [session.handle(event) for event in stream]
    flagged = [o for o in outcomes if o.left.flagged]
    assert flagged, "no request flagged; widen the stream or lower the threshold"
    for outcome in flagged:
        resolution = outcome.certificate.resolution
        assert resolution.action is Action.ESCALATE
        assert resolution.triggering_finding_ids
        assert resolution.rationale


def test_the_envelope_verdict_is_measured_and_not_assumed(
    session, demo_evalset, demo_cache, config: Config
) -> None:
    """The demo scores a real PSI window, and says so when it has no verdict.

    The stream is drawn from the warrant's own test rows, so the traffic really
    is inside the envelope. Below ``config.drift.window_size`` the system has no
    evidence of that and must report ``INSUFFICIENT_DATA`` rather than
    ``INSIDE`` — the difference between "no shift" and "no evidence yet" is the
    whole reason the fourth envelope state exists. Until Phase 5 this field was
    a hardcoded ``INSIDE`` with ``max_psi=0.0`` and ``n_window=1``.
    """
    assert config.drift.window_size > 10, "this test assumes a window it cannot fill"
    stream = record_stream(demo_evalset, demo_cache, n_events=10, seed=config.seed)
    outcomes = [session.handle(event) for event in stream]

    for position, outcome in enumerate(outcomes, start=1):
        match = outcome.certificate.envelope_match
        assert match.state is EnvelopeState.INSUFFICIENT_DATA
        assert match.n_window == position, "the window must advance per request"
        assert match.max_psi == 0.0

    assert "no envelope verdict yet" in outcomes[-1].right.envelope
    assert str(config.drift.window_size) in outcomes[-1].right.envelope


def test_certificates_record_what_was_not_checked(
    session, demo_evalset, demo_cache, config: Config
) -> None:
    """The honest half, and the half a dashboard never shows."""
    stream = record_stream(demo_evalset, demo_cache, n_events=2, seed=config.seed)
    outcome = session.handle(stream.events[0])
    assert outcome.certificate.unchecked


def test_a_refused_warrant_claims_no_bounds_at_all(
    refused_session, demo_evalset, demo_cache, config: Config
) -> None:
    """When the warrant is refused, the right pane says so and claims nothing.

    The failure mode this guards against is a demo that degrades gracefully into
    showing the *last* good numbers, which is precisely the unbacked claim the
    product exists to refuse.
    """
    assert refused_session.warrant.status is WarrantStatus.REFUSED
    assert "canary" in refused_session.warrant.status_reason

    stream = record_stream(demo_evalset, demo_cache, n_events=3, seed=config.seed)
    outcome = refused_session.handle(stream.events[0])

    assert outcome.right.bounds == {}
    assert outcome.certificate.claimed_bounds == {}
    assert outcome.certificate.warrants_relied_upon == ()
    assert outcome.certificate.weakest_warrant_status is WarrantStatus.UNVALIDATED
    # The score is still shown. Refusing to claim bounds is not refusing to work.
    assert outcome.right.score == outcome.left.score


# --------------------------------------------------------------------------- #
# Prove it
# --------------------------------------------------------------------------- #


def test_prove_it_runs_a_real_validation(session) -> None:
    """Beat 5: pressing the button measures, it does not read a file."""
    progress: list[str] = []
    run = session.prove_it(progress=progress.append)

    assert len(run.controls) == 5
    assert run.test_scored == 1
    assert any("scoring test" in line for line in progress)

    padding = next(c for c in run.controls if c.control == "padding_fault")
    assert "REJECTED as required" in padding.detail

    # Two of the five are negative controls, and they report their own null.
    for name in ("label_shuffle", "null_feature"):
        control = next(c for c in run.controls if c.control == name)
        assert "repeats" in control.detail
        assert "null spread" in control.detail


def test_prove_it_reproduces_the_prepared_run(session) -> None:
    """Same seed, same numbers — the button is not a fresh roll of the dice."""
    again = session.prove_it()
    assert again.metrics == session.run.metrics
    assert again.warrant.warrant_id == session.run.warrant.warrant_id
    assert again.warrant.status is session.run.warrant.status


def test_demo_runner_has_no_logic() -> None:
    """``CLAUDE.md``: the runner renders. Decisions live in ``src/``.

    Checked by asserting the runner never imports the pieces that decide —
    no probe, no issuance, no statistics. It may only reach them through
    :class:`DemoSession`.
    """
    runner = (
        Path(__file__).resolve().parents[1] / "demo" / "run_demo.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "from src.detectors",
        "from src.validation.issuance",
        "from src.validation.stats",
        "from src.validation.controls",
        "LinearProbe",
        "issue_or_refuse",
    ):
        assert forbidden not in runner, f"runner reaches past DemoSession: {forbidden}"
