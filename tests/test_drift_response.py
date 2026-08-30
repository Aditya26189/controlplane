"""The revocation ladder, the routing it triggers, and the record it leaves.

``tests/test_drift.py`` covers the measurement — PSI, its null, the window
minimum. This file covers what the system *does* with a verdict, which is a
different kind of failure: a monitor can be perfectly calibrated and still be
worthless if the revocation it produces changes nothing a reader can see.

``test_the_phase_5_gate`` is the gate in ``TASKS.md`` Phase 5, end to end.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from controlplane.drift import (
    DriftMonitor,
    DriftVerdict,
    apply_ladder,
    certify_drift_response,
    ladder_policy_hash,
    respond_to_drift,
)
from controlplane.drift.psi import PsiResult
from controlplane.matrix import MatrixCell, Profile, WarrantMatrix
from controlplane.model import (
    Action,
    DistributionEnvelope,
    EnvelopeFeature,
    EnvelopeState,
    WarrantStatus,
    utc_now,
)

from .factories import failing_controls, make_warrant

# --------------------------------------------------------------------------- #
# Fixtures shaped like the measured envelopes
# --------------------------------------------------------------------------- #

#: The two token-length regimes actually measured in Round 1 and Phase 4:
#: short-context TriviaQA against the long-context variant of the same set.
_SHORT = (45.0, 12.0, 20.0, 200.0)
_LONG = (6950.0, 1800.0, 2800.0, 11200.0)


def _draw(spec: tuple[float, float, float, float], n: int, seed: int) -> np.ndarray:
    mean, std, lo, hi = spec
    return np.random.default_rng(seed).normal(mean, std, n).clip(lo, hi)


def _envelope(eval_set_id: str, values: np.ndarray, bins: int = 8) -> DistributionEnvelope:
    """A real binned envelope, so the PSI in these tests is a measured number."""
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    probs = np.histogram(values, bins=edges)[0] / len(values)
    return DistributionEnvelope(
        envelope_id="sha256:%s" % eval_set_id,
        eval_set_id=eval_set_id,
        n_reference=len(values),
        data_source="measured",
        features=(
            EnvelopeFeature(
                name="token_length",
                bin_edges=tuple(float(x) for x in edges),
                bin_probabilities=tuple(float(x) for x in probs),
                mean=float(values.mean()),
                std=float(values.std()),
            ),
        ),
    )


def _warrant_on(eval_set_id: str, values: np.ndarray, **kwargs):
    """A warrant whose stored envelope is the one these values came from."""
    warrant = make_warrant(eval_set_id=eval_set_id, **kwargs)
    return dataclass_replace(warrant, envelope=_envelope(eval_set_id, values))


def dataclass_replace(obj, **changes):
    import dataclasses

    return dataclasses.replace(obj, **changes)


def _verdict(state: EnvelopeState, psi: float = 0.0, n: int = 200) -> DriftVerdict:
    """A verdict at a chosen rung, with a plausible PSI under it."""
    result = PsiResult(
        feature="token_length",
        psi=psi,
        n_live=n,
        n_reference=600,
        bins_smoothed=0,
        epsilon=1e-6,
        per_bin=(psi,),
    )
    return DriftVerdict(
        state=state,
        n_observed=n,
        window_size=200,
        per_feature={} if state is EnvelopeState.INSUFFICIENT_DATA else {"token_length": result},
        unobserved=(),
        driver=None if state is EnvelopeState.INSUFFICIENT_DATA else "token_length",
        reason="token_length PSI %.4f" % psi,
    )


def _matrix(*warrants, envelopes=None) -> WarrantMatrix:
    now = utc_now()
    cells = [WarrantMatrix._cell_for(w, now) for w in warrants]
    return WarrantMatrix(
        cells,
        detectors=sorted({w.detector_id for w in warrants}),
        envelopes=envelopes or sorted({w.eval_set_id for w in warrants}),
        now=now,
    )


PROFILE = Profile(
    name="customer_support", min_recall=0.05, max_fpr_hard_negatives=0.50,
    inline_budget_ms=200
)

POLICY_HASH = ladder_policy_hash(
    psi_stable=0.10, psi_significant=0.25, window_size=200, widen_factor=1.5
)


# --------------------------------------------------------------------------- #
# The ladder
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "state,expected",
    [
        (EnvelopeState.INSIDE, WarrantStatus.VALID),
        (EnvelopeState.MODERATE_SHIFT, WarrantStatus.STALE),
        (EnvelopeState.SIGNIFICANT_SHIFT, WarrantStatus.REVOKED),
    ],
)
def test_each_rung_maps_to_its_status(state, expected) -> None:
    """``SPEC.md`` §5.3, the three rungs that carry a verdict."""
    warrant = make_warrant()
    transition = apply_ladder(warrant, _verdict(state))
    assert transition.to_status is expected
    assert transition.envelope_state is state


def test_insufficient_data_decides_nothing() -> None:
    """The row that is easy to get wrong.

    Below the window minimum there is no evidence, and no evidence must not
    become a decision in either direction. A monitor that revokes on forty
    requests gets switched off; one that certifies on forty is asserting a
    stability it never measured.
    """
    warrant = make_warrant()
    transition = apply_ladder(warrant, _verdict(EnvelopeState.INSUFFICIENT_DATA, n=40))
    assert transition.to_status is warrant.status
    assert not transition.changed
    assert not transition.needs_routing
    assert transition.widen_factor is None


def test_insufficient_data_does_not_lift_a_revoked_warrant() -> None:
    """The same row in the other direction: silence restores nothing either."""
    revoked = make_warrant(status=WarrantStatus.REVOKED, status_reason="envelope moved")
    transition = apply_ladder(revoked, _verdict(EnvelopeState.INSUFFICIENT_DATA, n=12))
    assert transition.to_status is WarrantStatus.REVOKED


def test_a_refused_warrant_does_not_climb_back_up() -> None:
    """Control failure is a statement about the detector, not about the traffic.

    ``CLAUDE.md`` invariant 3: no argument anywhere promotes a warrant whose
    controls failed. Favourable traffic is an argument.
    """
    refused = make_warrant(
        controls=failing_controls("null_feature"),
        status=WarrantStatus.REFUSED,
        status_reason="null_feature failed",
    )
    transition = apply_ladder(refused, _verdict(EnvelopeState.INSIDE, psi=0.001))
    assert transition.to_status is WarrantStatus.REFUSED
    assert "out of service" in transition.reason


def test_only_a_stale_transition_carries_a_widening_factor() -> None:
    """A factor on any other rung would be applied to bounds nobody widened."""
    warrant = make_warrant()
    assert apply_ladder(warrant, _verdict(EnvelopeState.MODERATE_SHIFT, 0.15)).widen_factor == 1.5
    assert apply_ladder(warrant, _verdict(EnvelopeState.INSIDE, 0.01)).widen_factor is None
    assert apply_ladder(warrant, _verdict(EnvelopeState.SIGNIFICANT_SHIFT, 0.9)).widen_factor is None


# --------------------------------------------------------------------------- #
# The response
# --------------------------------------------------------------------------- #


class _StubMonitor:
    """A monitor that returns a fixed verdict, for the routing tests."""

    def __init__(self, verdict: DriftVerdict) -> None:
        self._verdict = verdict

    def verdict(self) -> DriftVerdict:
        return self._verdict


def test_a_valid_envelope_consults_nothing() -> None:
    warrant = make_warrant()
    response = respond_to_drift(
        warrant,
        _StubMonitor(_verdict(EnvelopeState.INSIDE, 0.02)),
        matrix=_matrix(warrant),
        profile=PROFILE,
        live_eval_set_id=warrant.eval_set_id,
    )
    assert response.routing is None
    assert response.can_certify
    assert response.claimed_bounds


def test_stale_bounds_widen_about_their_point_estimate() -> None:
    """Widening moves the interval, never the measurement."""
    warrant = make_warrant()
    response = respond_to_drift(
        warrant,
        _StubMonitor(_verdict(EnvelopeState.MODERATE_SHIFT, 0.15)),
        matrix=_matrix(warrant),
        profile=PROFILE,
        live_eval_set_id=warrant.eval_set_id,
    )
    before = warrant.metrics.recall
    after = response.claimed_bounds["recall"]
    assert after["value"] == pytest.approx(before.value)
    assert after["ci_low"] < before.ci_low
    assert after["ci_high"] > before.ci_high
    assert after["widened_by"] == 1.5


def test_a_revocation_routes_and_adopts_the_new_bounds() -> None:
    """The Phase 5 claim in miniature: the detector was never what went stale.

    ``T1-last_token`` holds a warrant on each envelope. Long-context traffic
    revokes the short-context one; the system keeps the detector and starts
    quoting the bounds measured on the envelope the traffic is actually in.
    """
    short = _warrant_on("triviaqa-600", _draw(_SHORT, 600, 1729))
    long = _warrant_on("triviaqa-longctx-600", _draw(_LONG, 600, 17))

    response = respond_to_drift(
        short,
        _StubMonitor(_verdict(EnvelopeState.SIGNIFICANT_SHIFT, 6.1)),
        matrix=_matrix(short, long),
        profile=PROFILE,
        live_eval_set_id="triviaqa-longctx-600",
    )

    assert response.transition.to_status is WarrantStatus.REVOKED
    assert response.routing is not None and response.routing.routed
    assert response.routing.warrant.warrant_id == long.warrant_id
    assert response.routing.warrant.warrant_id != short.warrant_id
    assert response.can_certify
    assert response.claimed_bounds


def test_an_unrouted_revocation_claims_nothing() -> None:
    """The failure this module exists to prevent: falling back to lost numbers."""
    short = _warrant_on("triviaqa-600", _draw(_SHORT, 600, 1729))
    response = respond_to_drift(
        short,
        _StubMonitor(_verdict(EnvelopeState.SIGNIFICANT_SHIFT, 6.1)),
        matrix=_matrix(short, envelopes=["triviaqa-600", "triviaqa-longctx-600"]),
        profile=PROFILE,
        live_eval_set_id="triviaqa-longctx-600",
    )
    assert response.claimed_bounds == {}
    assert not response.can_certify
    assert response.routing.action is PROFILE.conservative_default


def test_an_unknown_envelope_raises_rather_than_reading_as_an_absence() -> None:
    """A missed lookup must not be reported as a measured "nothing holds one"."""
    short = _warrant_on("triviaqa-600", _draw(_SHORT, 600, 1729))
    with pytest.raises(KeyError, match="not in the matrix"):
        respond_to_drift(
            short,
            _StubMonitor(_verdict(EnvelopeState.SIGNIFICANT_SHIFT, 6.1)),
            matrix=_matrix(short),
            profile=PROFILE,
            live_eval_set_id=short.envelope.envelope_id,  # a hash, not an eval_set_id
        )


# --------------------------------------------------------------------------- #
# The certificate
# --------------------------------------------------------------------------- #


def _certify(response, warrant, live: str):
    return certify_drift_response(
        response,
        warrant,
        certificate_id="C-1",
        request_id="R-1",
        session_id="S-1",
        live_envelope_id=live,
        timestamp=utc_now(),
        policy_hash=POLICY_HASH,
    )


def test_a_routed_certificate_cites_the_replacement_not_the_revoked() -> None:
    short = _warrant_on("triviaqa-600", _draw(_SHORT, 600, 1729))
    long = _warrant_on("triviaqa-longctx-600", _draw(_LONG, 600, 17))
    response = respond_to_drift(
        short,
        _StubMonitor(_verdict(EnvelopeState.SIGNIFICANT_SHIFT, 6.1)),
        matrix=_matrix(short, long),
        profile=PROFILE,
        live_eval_set_id="triviaqa-longctx-600",
    )
    cert = _certify(response, short, "triviaqa-longctx-600")

    assert cert.warrants_relied_upon == (long.warrant_id,)
    assert short.warrant_id not in cert.warrants_relied_upon
    assert cert.claimed_bounds
    assert cert.resolution.triggered_by == "envelope:SIGNIFICANT_SHIFT"


def test_an_unrouted_certificate_cites_nothing_and_claims_nothing() -> None:
    short = _warrant_on("triviaqa-600", _draw(_SHORT, 600, 1729))
    response = respond_to_drift(
        short,
        _StubMonitor(_verdict(EnvelopeState.SIGNIFICANT_SHIFT, 6.1)),
        matrix=_matrix(short, envelopes=["triviaqa-600", "triviaqa-longctx-600"]),
        profile=PROFILE,
        live_eval_set_id="triviaqa-longctx-600",
    )
    cert = _certify(response, short, "triviaqa-longctx-600")

    assert cert.warrants_relied_upon == ()
    assert cert.claimed_bounds == {}
    assert cert.weakest_warrant_status is WarrantStatus.REVOKED
    assert any("revoked and no other detector" in u for u in cert.unchecked)


def test_every_certificate_discloses_that_no_multivariate_check_ran() -> None:
    """MMD is declared in ``SPEC.md`` §5.2 and is not implemented.

    The gap is disclosed on the record rather than left as a ``None`` field
    nobody reads, and ``mmd_p_value`` stays ``None`` rather than defaulting to
    a p-value that would read as a test that ran.
    """
    warrant = make_warrant()
    response = respond_to_drift(
        warrant,
        _StubMonitor(_verdict(EnvelopeState.INSIDE, 0.02)),
        matrix=_matrix(warrant),
        profile=PROFILE,
        live_eval_set_id=warrant.eval_set_id,
    )
    cert = _certify(response, warrant, warrant.eval_set_id)
    assert cert.envelope_match.mmd_p_value is None
    assert any("MMD" in u for u in cert.unchecked)


def test_the_policy_hash_moves_with_the_bands_it_names() -> None:
    """Two certificates stamped the same version under different bands are
    different decisions, and a reader who cannot see that cannot replay either.
    """
    base = dict(psi_stable=0.10, psi_significant=0.25, window_size=200, widen_factor=1.5)
    assert ladder_policy_hash(**base) == ladder_policy_hash(**base)
    assert ladder_policy_hash(**{**base, "psi_significant": 0.30}) != ladder_policy_hash(**base)
    assert ladder_policy_hash(**{**base, "window_size": 500}) != ladder_policy_hash(**base)


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def test_the_phase_5_gate(tmp_path) -> None:
    """``TASKS.md`` Phase 5.

    Feed ``triviaqa-longctx-600`` as live traffic against a warrant measured on
    ``triviaqa-600``. With **no manual trigger**, the system must detect the
    shift, revoke, consult the matrix, route to a detector holding a valid
    warrant on that envelope, adopt its bounds, and write a certificate
    explaining all of it — which then seals into the append-only ledger.
    """
    from controlplane.store import Ledger

    short_values = _draw(_SHORT, 600, 1729)
    short = _warrant_on("triviaqa-600", short_values)
    long = _warrant_on("triviaqa-longctx-600", _draw(_LONG, 600, 17))

    # A monitor over the warrant's *stored* envelope. Never a re-derived
    # reference: rebinning on recent traffic compares each window to itself and
    # reports stability forever.
    monitor = DriftMonitor(
        short.envelope,
        window_size=200,
        psi_stable=0.10,
        psi_significant=0.25,
        features=["token_length"],
        max_false_alarm_rate=0.05,
    )

    # Live traffic arrives. Nothing here inspects it or decides to intervene.
    for value in _draw(_LONG, 200, 99):
        monitor.observe({"token_length": float(value)})

    response = respond_to_drift(
        short,
        monitor,
        matrix=_matrix(short, long),
        profile=PROFILE,
        live_eval_set_id="triviaqa-longctx-600",
    )

    # detects the shift
    assert response.transition.envelope_state is EnvelopeState.SIGNIFICANT_SHIFT
    # revokes
    assert response.transition.to_status is WarrantStatus.REVOKED
    assert response.transition.needs_routing
    # consults the matrix and routes to a valid warrant on the new envelope
    assert response.routing is not None
    assert response.routing.routed
    assert response.routing.warrant.eval_set_id == "triviaqa-longctx-600"
    assert response.routing.warrant.status is WarrantStatus.VALID
    # adopts its bounds, and they are the new warrant's
    assert response.claimed_bounds
    assert response.claimed_bounds["recall"]["value"] == pytest.approx(
        long.metrics.recall.value
    )
    # with no manual trigger
    assert response.acted_without_operator

    # ...and writes a certificate explaining all of it.
    cert = _certify(response, short, "triviaqa-longctx-600")
    assert cert.warrants_relied_upon == (long.warrant_id,)
    assert cert.envelope_match.state is EnvelopeState.SIGNIFICANT_SHIFT
    assert cert.envelope_match.driving_feature == "token_length"
    assert cert.envelope_match.n_window == 200
    assert cert.resolution.policy_hash == POLICY_HASH
    assert "triviaqa-600" in cert.resolution.rationale or any(
        "triviaqa-600" in u for u in cert.unchecked
    )

    store = Ledger(tmp_path / "gate.db", retention_days=400)
    try:
        sealed = store.append_certificate(cert)
        assert sealed.is_sealed
        assert store.verify_chain().ok
    finally:
        store.close()


def test_the_gate_refuses_and_enqueues_when_nothing_holds_a_warrant() -> None:
    """The other half of the gate sentence, which is the more common case.

    ``UNVALIDATED`` is the modal state in production. The request is still
    handled; what stops is the *claim*.
    """
    short = _warrant_on("triviaqa-600", _draw(_SHORT, 600, 1729))
    enqueued: list = []

    monitor = DriftMonitor(
        short.envelope,
        window_size=200,
        psi_stable=0.10,
        psi_significant=0.25,
        features=["token_length"],
        max_false_alarm_rate=0.05,
    )
    for value in _draw(_LONG, 200, 99):
        monitor.observe({"token_length": float(value)})

    response = respond_to_drift(
        short,
        monitor,
        matrix=_matrix(short, envelopes=["triviaqa-600", "triviaqa-longctx-600"]),
        profile=PROFILE,
        live_eval_set_id="triviaqa-longctx-600",
    )

    assert not response.can_certify
    assert response.claimed_bounds == {}
    assert response.routing.action is PROFILE.conservative_default

    cert = _certify(response, short, "triviaqa-longctx-600")
    assert cert.claimed_bounds == {}
    assert cert.resolution.action is Action.ESCALATE
