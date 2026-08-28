"""The demo's state and decisions. The runner renders; this decides.

``CLAUDE.md``: no logic in the demo runner. Everything a pane displays is
computed here, from the same code paths the pipeline uses — the same probe, the
same warrant, the same certificate, the same ledger. A demo that computed its
own numbers would be a mock-up, and a mock-up of a measurement system is the
thing this project exists to argue against.

**What the two panes differ in, at this phase.** Both see the same stream and
the same detector scores. The left pane is the conventional stack: a score and a
verdict, which is what every detection product shows. The right pane adds the
warrant — what that score is worth, on this envelope, measured how, how long
ago, and with what interval.

The left pane is **not** a strawman. It is the same detector at the same
threshold. Everything it shows is true; it simply cannot say what any of it is
worth. That is the whole argument, and weakening the left pane would replace an
argument with a rigged comparison. Real third-party detectors at their
documented defaults arrive in Phase 8.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np

from ..config import Config
from ..drift import DriftMonitor
from ..model import (
    AccessTier,
    Action,
    Category,
    Certificate,
    ConfidenceBand,
    EnvelopeMatchResult,
    EnvelopeState,
    Finding,
    Resolution,
    Severity,
    Span,
    Warrant,
    WarrantStatus,
    utc_now,
)
from ..store import Ledger
from ..validation.evalsets import EvalSet, ExtractionCache
from ..validation.runner import ValidationRun, validate
from .stream import Stream, StreamEvent

__all__ = ["DemoSession", "PaneView", "RequestOutcome"]


@dataclasses.dataclass(frozen=True)
class PaneView:
    """What one pane shows for one request.

    Args:
        headline: The verdict line.
        score: The detector score.
        flagged: Whether it fired.
        bounds: Claimed bounds, or None. The left pane never has any — not
            because we withhold them, but because a bare detector has none to
            give.
        envelope: Envelope status line, or None for the same reason.
        warrant_age: How old the backing measurement is, or None.
        notes: Anything else worth a line.
    """

    headline: str
    score: float
    flagged: bool
    bounds: Optional[dict[str, Any]] = None
    envelope: Optional[str] = None
    warrant_age: Optional[str] = None
    notes: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class RequestOutcome:
    """Both panes' view of one request, plus the certificate the right one wrote."""

    event: StreamEvent
    left: PaneView
    right: PaneView
    certificate: Optional[Certificate]


def _render_envelope(match: EnvelopeMatchResult, config: Config) -> str:
    """One line for the right pane, saying what is known and what is not.

    Below the window minimum the line says how far it has to go rather than
    printing ``max PSI 0.000``, which would read as a measured stability. The
    difference between "no shift" and "no evidence yet" is the whole reason the
    fourth envelope state exists.
    """
    if match.state is EnvelopeState.INSUFFICIENT_DATA:
        return (
            f"{match.envelope_id} | {match.state.value} | "
            f"{match.n_window} of {config.drift.window_size} requests — "
            "no envelope verdict yet"
        )
    return (
        f"{match.envelope_id} | {match.state.value} | "
        f"max PSI {match.max_psi:.3f} on {match.driving_feature} "
        f"(n={match.n_window})"
    )


class DemoSession:
    """Drives a recorded stream through both panes and can re-prove itself.

    Args:
        config: Resolved config.
        evalset: The set the stream was drawn from.
        cache: Its extraction.
        variant: Which tier variant the demo runs on.
        detector_id: Detector identity.
        detector_version: Semver plus weights hash.
        ledger: Where certificates are written. Real, hash-chained, and the same
            store the pipeline uses.
        canary_cache: Canary extraction, so ``prove_it`` runs the real five.
    """

    def __init__(
        self,
        config: Config,
        evalset: EvalSet,
        cache: ExtractionCache,
        *,
        variant: str,
        detector_id: str,
        detector_version: str,
        ledger: Ledger,
        canary_cache: Optional[ExtractionCache] = None,
    ) -> None:
        self.config = config
        self.evalset = evalset
        self.cache = cache
        self.variant = variant
        self.detector_id = detector_id
        self.detector_version = detector_version
        self.ledger = ledger
        self.canary_cache = canary_cache
        self._run: Optional[ValidationRun] = None
        self._scores: Optional[np.ndarray] = None
        self._counter = 0
        #: Scored against the warrant's **stored** envelope, built in
        #: ``prepare`` once there is a warrant to take one from.
        self._monitor: Optional[DriftMonitor] = None

    # -- setup -------------------------------------------------------------- #

    def prepare(self, progress: Optional[Callable[[str], None]] = None) -> ValidationRun:
        """Validate once, so the demo has a warrant to display and rely on.

        Called before the stream plays. The banner in Beat 1 shows this run's
        numbers, and the certificates written during the stream cite this run's
        warrant — so what the audience sees on screen and what lands in the
        ledger are the same claim.
        """
        self._run = validate(
            self.config,
            self.evalset,
            self.cache,
            variant=self.variant,
            detector_id=self.detector_id,
            detector_version=self.detector_version,
            target_flag_rate=0.05,
            canary_cache=self.canary_cache,
            progress=progress,
        )
        probe_scores = self._score_everything()
        self._scores = probe_scores
        self.ledger.append_warrant(self._run.warrant)

        drift = self.config.drift
        self._monitor = DriftMonitor(
            self._run.warrant.envelope,
            window_size=drift.window_size,
            psi_stable=drift.psi_stable,
            psi_significant=drift.psi_significant,
            features=drift.features,
            max_false_alarm_rate=drift.max_false_alarm_rate,
        )
        return self._run

    def _score_everything(self) -> np.ndarray:
        """Score every row once, so replaying the stream costs nothing.

        Refits the probe on the training rows exactly as the validation did,
        which is why the demo's scores and the warrant's numbers agree. A demo
        that scored with a differently-fitted probe would be showing one thing
        and claiming another.
        """
        from ..detectors.probe import LinearProbe
        from ..validation.evalsets import TRAIN, split_by_question

        assert self._run is not None
        splits = split_by_question(self.evalset, seed=self.config.seed)
        features = self.cache.matrix(self.variant)
        probe = LinearProbe(
            self._run.probe_fit.C,
            class_weight=self.config.probe.class_weight,
            standardize=self.config.probe.standardize,
            seed=self.config.seed,
        ).fit(features, self.cache.labels, splits[TRAIN])
        return probe.score(features)

    @property
    def run(self) -> ValidationRun:
        """The validation backing this session."""
        if self._run is None:
            raise RuntimeError("call prepare() before playing the stream")
        return self._run

    @property
    def warrant(self) -> Warrant:
        """The warrant the right pane relies on."""
        return self.run.warrant

    # -- the banner --------------------------------------------------------- #

    def warrant_banner(self, now: Optional[datetime] = None) -> dict[str, Any]:
        """Beat 1's banner: what is being claimed, and on what basis.

        Detector, operating point, envelope, measured bounds with intervals,
        validation age, envelope status. Every line is a fact the left pane
        cannot state about itself.
        """
        now = now or utc_now()
        warrant = self.warrant
        age = warrant.age(now)
        metrics = warrant.metrics
        return {
            "detector": f"{warrant.detector_id}@{warrant.detector_version}",
            "operating_point": warrant.operating_point.operating_point_id,
            "threshold": warrant.operating_point.threshold,
            "selected_on": warrant.operating_point.selected_on,
            "envelope": f"{warrant.eval_set_id} [{warrant.envelope.envelope_id}]",
            "status": warrant.status.value,
            "status_reason": warrant.status_reason,
            "recall": metrics.recall.render(3),
            "precision": metrics.precision.render(3),
            "auroc": metrics.auroc.render(3),
            "flag_rate": metrics.flag_rate.render(4),
            "confirmed_errors": metrics.confirmed_errors.render(),
            "base_rate": warrant.base_rate,
            "n_test": warrant.n_test,
            "kappa": warrant.kappa,
            "validated": f"{int(age.total_seconds())}s ago",
            "expires_in_hours": round(
                (warrant.expires_at - now).total_seconds() / 3600.0, 2
            ),
            "data_source": self.run.data_source,
        }

    # -- the stream --------------------------------------------------------- #

    def handle(self, event: StreamEvent, now: Optional[datetime] = None) -> RequestOutcome:
        """Run one request through both panes and write the right pane's certificate.

        The left pane's view is built from the same score and the same threshold
        as the right pane's. It differs only in what it can say about them.
        """
        now = now or utc_now()
        assert self._scores is not None
        assert self._monitor is not None, "prepare() must run before the stream"
        # The window advances on every request, before anything is certified.
        # Feeding it after certification would let a request be certified
        # against a verdict that did not include it.
        self._monitor.observe({"token_length": float(event.token_length)})
        score = float(self._scores[event.row])
        threshold = self.warrant.operating_point.threshold
        flagged = score >= threshold

        left = PaneView(
            headline="FLAG — send to review" if flagged else "clear",
            score=score,
            flagged=flagged,
            notes=(
                "no interval, no envelope, no expiry — a score is all a bare "
                "detector has",
            ),
        )

        certificate = self._certify(event, score, flagged, now)
        warrant = self.warrant
        right = PaneView(
            headline=(
                f"{'FLAG' if flagged else 'clear'} — "
                f"{certificate.resolution.action.value.lower()}"
            ),
            score=score,
            flagged=flagged,
            bounds=certificate.claimed_bounds,
            envelope=_render_envelope(certificate.envelope_match, self.config),
            warrant_age=f"{int(warrant.age(now).total_seconds())}s",
            notes=(
                f"certificate {certificate.certificate_id} "
                f"chained at {certificate.self_hash[:12]}",
            )
            + tuple(certificate.unchecked),
        )
        return RequestOutcome(event=event, left=left, right=right, certificate=certificate)

    def _certify(
        self, event: StreamEvent, score: float, flagged: bool, now: datetime
    ) -> Certificate:
        """Write the certificate for one request and seal it into the ledger."""
        warrant = self.warrant
        relied_on = warrant.status.can_be_relied_upon and not warrant.is_expired(now)
        self._counter += 1

        band = ConfidenceBand.CONFIDENT_POSITIVE if flagged else ConfidenceBand.CONFIDENT_NEGATIVE
        finding = Finding(
            finding_id=f"F-{self._counter:05d}",
            detector_id=self.detector_id,
            detector_version=self.detector_version,
            category=Category.HALLUCINATION,
            severity=Severity.HIGH if flagged else Severity.INFO,
            confidence=score,
            evidence_spans=(
                Span(0, len(event.response), event.response, "response"),
            ),
            access_tier=warrant.access_tier,
            latency_ms=0.4,
            warrant_id=warrant.warrant_id if relied_on else None,
            confidence_band=band,
        )

        # A real PSI over the sliding window, scored against the envelope
        # stored in the warrant. Below config.drift.window_size this reports
        # INSUFFICIENT_DATA rather than INSIDE, and that is not a degraded
        # answer -- the demo stream is drawn from the warrant's own test rows,
        # so the traffic *is* inside, but at n < 200 the system has no evidence
        # of it and must not certify a stability it has not measured. A Beat 4
        # that needs to show a revocation needs a segment at least as long as
        # the window (DECISIONS.md 075).
        assert self._monitor is not None
        verdict = self._monitor.verdict()
        psi_by_feature = {name: r.psi for name, r in verdict.per_feature.items()}
        if psi_by_feature and verdict.driver is not None:
            envelope_match = EnvelopeMatchResult(
                envelope_id=warrant.envelope.envelope_id,
                state=verdict.state,
                psi_by_feature=psi_by_feature,
                max_psi=max(psi_by_feature.values()),
                driving_feature=verdict.driver,
                n_window=verdict.n_observed,
            )
        else:
            envelope_match = EnvelopeMatchResult(
                envelope_id=warrant.envelope.envelope_id,
                state=EnvelopeState.INSUFFICIENT_DATA,
                psi_by_feature={},
                max_psi=0.0,
                driving_feature="",
                n_window=verdict.n_observed,
            )

        action = Action.ESCALATE if flagged else Action.ALLOW
        resolution = Resolution(
            action=action,
            policy_version="demo-0.1",
            policy_hash="sha256:demo-phase-2.5",
            triggering_finding_ids=(finding.finding_id,) if flagged else (),
            rule_id="R-escalate-on-flag" if flagged else None,
            rationale=(
                "detector fired at its validated operating point; routed to review"
                if flagged
                else ""
            ),
        )

        certificate = Certificate(
            certificate_id=f"C-{event.request_id}",
            request_id=event.request_id,
            session_id=event.session_id,
            timestamp=now,
            findings=(finding,),
            resolution=resolution,
            warrants_relied_upon=(warrant.warrant_id,) if relied_on else (),
            weakest_warrant_status=(
                warrant.status if relied_on else WarrantStatus.UNVALIDATED
            ),
            claimed_bounds=warrant.claimed_bounds() if relied_on else {},
            envelope_match=envelope_match,
            access_tier_available=warrant.access_tier,
            unchecked=(
                "PII, safety and grounding — no detector configured for those "
                "categories at this phase",
            ),
        )
        return self.ledger.append_certificate(certificate)

    # -- Prove it ----------------------------------------------------------- #

    def prove_it(
        self, progress: Optional[Callable[[str], None]] = None
    ) -> ValidationRun:
        """Re-run the real validation, live. Beat 5.

        Not a replay of stored results: it re-fits the probe, re-runs all five
        controls including the deliberately broken padding case, re-scores test
        and re-issues or re-refuses. That is the point of the button — a judge
        pressing it is watching the measurement happen, not watching a file
        being read.
        """
        return validate(
            self.config,
            self.evalset,
            self.cache,
            variant=self.variant,
            detector_id=self.detector_id,
            detector_version=self.detector_version,
            target_flag_rate=0.05,
            canary_cache=self.canary_cache,
            progress=progress,
        )
