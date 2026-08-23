"""``/validate`` — one detector, one operating point, one envelope. ``SPEC.md`` §2.2.

Runs from a cached extraction, which is what makes it fast enough to be pressed
as a button in front of an audience: extraction is a GPU hour, this is a few
seconds of linear algebra, and they meet on disk.

The order of operations is the part that matters, and it is fixed:

1. split by question, asserting no overlap;
2. select ``C`` and the threshold **on validation**;
3. run the five controls;
4. score test **once**, and record that it was scored;
5. issue or refuse.

Test is scored after selection, never before, and the run records the scoring in
its own artifact so "test is scored once per validation run, and every run is
published" is checkable by someone who was not present for any of them.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np

from ..config import Config, provenance
from ..detectors.probe import ProbeFit, select_regularisation
from ..model import (
    AccessTier,
    ControlResult,
    DistributionEnvelope,
    EnvelopeFeature,
    Metric,
    OperatingPoint,
    Warrant,
    WarrantKey,
    WarrantMetrics,
    WarrantStatus,
    content_hash,
    utc_now,
)
from .controls import run_controls
from .evalsets import TEST, TRAIN, VALIDATION, ExtractionCache, split_by_question
from .evalsets import EvalSet
from .issuance import issue_or_refuse
from .metrics_builder import build_warrant_metrics
from .stats import flag_rate_at, threshold_for_flag_rate

__all__ = ["ValidationRun", "build_envelope", "validate", "validate_transferred"]

_LOG = logging.getLogger(__name__)

#: Access tier per variant prefix. The ladder's rungs, and the thing the
#: ablation is actually measuring the value of.
_TIER_OF_VARIANT = {
    "T1": AccessTier.T1_ACTIVATIONS,
    "T2": AccessTier.T2_LOGPROBS,
    "T3": AccessTier.T3_TEXT,
}


@dataclasses.dataclass(frozen=True)
class ValidationRun:
    """Everything one ``/validate`` produced. ``SPEC.md`` §2.2.

    Args:
        run_id: Content-derived, so two runs of the same validation on the same
            code produce the same id.
        detector_id: Which detector.
        variant: Which tier variant of it.
        eval_set_id: Which set.
        envelope_id: The set's content hash — the warrant key's third element.
        started_at: When the run began.
        completed_at: When it finished.
        duration_seconds: Wall clock. The gate is "under a minute from cache",
            so the number is recorded rather than asserted in a comment.
        probe_fit: How ``C`` was chosen, and on which split.
        operating_point: The threshold and the objective that chose it.
        metrics: Measured bounds on test.
        controls: All five results.
        warrant: Issued or refused. Never None.
        splits: Row counts per split, so a reader can check the arithmetic.
        base_rate: Positive-class prevalence on test.
        data_source: ``"measured"`` or ``"synthetic"``, carried from the cache.
        test_scored: Always 1 for a completed run. Recorded explicitly because
            "test is scored once per validation run" is a claim, and a claim
            with no field behind it is a promise rather than a fact.
        provenance: Config hash, git commit, dirty flag, versions, device.
    """

    run_id: str
    detector_id: str
    variant: str
    eval_set_id: str
    envelope_id: str
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    probe_fit: ProbeFit
    operating_point: OperatingPoint
    metrics: WarrantMetrics
    controls: tuple[ControlResult, ...]
    warrant: Warrant
    splits: dict[str, int]
    base_rate: float
    data_source: str
    test_scored: int
    provenance: dict[str, Any]

    @property
    def issued(self) -> bool:
        """Whether the run produced a valid warrant."""
        return self.warrant.status is WarrantStatus.VALID

    def summary(self) -> str:
        """One-screen summary, as the demo prints it."""
        head = (
            f"{self.detector_id} / {self.variant} on {self.eval_set_id} "
            f"[{self.envelope_id}]"
        )
        control_lines = [
            f"  {'PASS' if c.passed else 'FAIL'}  {c.control:<15} "
            f"measured {c.measured:.4f}  margin {c.margin:+.4f}  ({c.expected})"
            for c in self.controls
        ]
        metric_lines = [f"  {m.name:<18} {m.render()}" for m in self.metrics.all_metrics()]
        verdict = (
            "WARRANT ISSUED"
            if self.issued
            else f"WARRANT REFUSED — {self.warrant.status_reason}"
        )
        return "\n".join(
            [head, f"  data_source: {self.data_source}", "", *metric_lines, "", *control_lines, "", verdict]
        )

    def to_payload(self) -> dict[str, Any]:
        """JSON-serialisable form for ``results/``."""
        from ..model import to_jsonable

        return {
            "run_id": self.run_id,
            "detector_id": self.detector_id,
            "variant": self.variant,
            "eval_set_id": self.eval_set_id,
            "envelope_id": self.envelope_id,
            "data_source": self.data_source,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "splits": self.splits,
            "base_rate": self.base_rate,
            "test_scored": self.test_scored,
            "probe_fit": dataclasses.asdict(self.probe_fit),
            "operating_point": to_jsonable(self.operating_point),
            "metrics": to_jsonable(self.metrics),
            "controls": to_jsonable(list(self.controls)),
            "warrant": to_jsonable(self.warrant),
            "warrant_status": self.warrant.status.value,
            "status_reason": self.warrant.status_reason,
        }


def build_envelope(
    evalset: EvalSet, cache: ExtractionCache, *, n_bins: int = 8
) -> DistributionEnvelope:
    """Compute the reference distribution and store it in the warrant.

    Only ``token_length`` is populated at this phase — it is the
    highest-priority feature and the one the long-context beat turns on
    (``SPEC.md`` §5.1). The remaining four arrive with the drift module in
    Phase 5; the envelope is built here so warrants issued now carry the
    reference their drift checks will need, rather than needing reissue later.

    Bin edges come from the reference quantiles, so each bin holds a comparable
    share of the reference. Equal-width bins over a skewed length distribution
    put almost everything in one bin, and PSI computed against that is
    insensitive to exactly the shift it exists to catch.

    Args:
        evalset: The set being validated.
        cache: Its extraction, supplying token lengths.
        n_bins: Number of PSI bins.

    Returns:
        A :class:`DistributionEnvelope` keyed by the set's content hash.
    """
    lengths = np.asarray(cache.token_lengths, dtype=float)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(lengths, quantiles))
    if edges.size < 2:
        # Every prompt the same length: one bin is degenerate, so widen by a
        # token either side. Happens on synthetic fixtures and would otherwise
        # make the envelope unconstructible.
        edges = np.array([lengths[0] - 1.0, lengths[0] + 1.0])
    counts, _ = np.histogram(lengths, bins=edges)
    probabilities = counts / counts.sum()
    # PSI divides by the reference proportion, so an empty reference bin makes
    # it infinite. Flooring at a small value keeps the statistic finite while
    # still registering a large divergence, which is the standard treatment.
    floor = 1e-6
    probabilities = np.maximum(probabilities, floor)
    probabilities = probabilities / probabilities.sum()

    return DistributionEnvelope(
        envelope_id=evalset.envelope_id,
        eval_set_id=evalset.eval_set_id,
        n_reference=len(evalset),
        data_source=evalset.data_source,
        features=(
            EnvelopeFeature(
                name="token_length",
                bin_edges=tuple(float(e) for e in edges),
                bin_probabilities=tuple(float(p) for p in probabilities),
                mean=float(lengths.mean()),
                std=float(lengths.std()),
            ),
        ),
    )


def validate(
    config: Config,
    evalset: EvalSet,
    cache: ExtractionCache,
    *,
    variant: str,
    detector_id: str,
    detector_version: str,
    operating_point_id: str = "P-conservative",
    target_flag_rate: Optional[float] = None,
    min_recall: Optional[float] = None,
    max_fpr_hard_negatives: Optional[float] = None,
    canary_cache: Optional[ExtractionCache] = None,
    kappa: Optional[float] = None,
    is_hard_negative_set: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> ValidationRun:
    """Validate one detector variant on one envelope, and issue or refuse.

    Args:
        config: Resolved config.
        evalset: The set being validated. Its content hash becomes the envelope
            id and therefore the warrant key's third element.
        cache: The extraction for that set. Its recorded hash is checked against
            the set's current hash, so a set edited after extraction is caught.
        variant: Which tier variant, e.g. ``"T1-max_rolling_means"``.
        detector_id: Detector identity for the warrant key.
        detector_version: Semver plus weights hash.
        operating_point_id: Identity of the threshold being validated.
        target_flag_rate: Flag-rate budget the threshold aims at, selected on
            validation. The **measured** rate on test is what gets reported and
            what every downstream calculation uses (invariant 6).
        min_recall: Consuming profile's declared minimum recall.
        max_fpr_hard_negatives: Declared maximum hard-negative FPR.
        canary_cache: Extraction for ``canary-20``, if available. Absent means
            the canary control *fails* rather than being skipped.
        kappa: Inter-rater agreement, where labels are human.
        is_hard_negative_set: Whether this eval set *is* the hard-negative set.
            Only then does its FPR populate ``fpr_hard_negatives`` and face a
            profile's declared maximum; otherwise the within-set FPR is reported
            under ``fpr``, because they are different claims (DECISIONS.md 036).
        progress: Optional callback for streaming progress to the demo.

    Returns:
        A completed :class:`ValidationRun`, carrying an issued or refused warrant.

    Raises:
        ValueError: If the cache does not match the eval set, or the variant is
            absent from it.
    """
    started = utc_now()
    clock = time.perf_counter()

    def say(message: str) -> None:
        _LOG.info(message)
        if progress is not None:
            progress(message)

    if cache.eval_set_hash != evalset.content_hash:
        raise ValueError(
            f"cache for {cache.eval_set_id} was extracted from contents hashing "
            f"{cache.eval_set_hash[:16]}, but {evalset.eval_set_id} now hashes to "
            f"{evalset.content_hash[:16]}. Re-extract; validating against a stale "
            "cache files numbers under an envelope that no longer describes the data."
        )

    say(f"splitting {evalset.eval_set_id} by question ({len(evalset)} items)")
    splits = split_by_question(evalset, seed=config.seed)
    features = cache.matrix(variant)
    labels = cache.labels
    groups = cache.question_ids

    say(f"selecting C on validation from grid {list(config.probe.C_grid)}")
    probe, fit = select_regularisation(
        features,
        labels,
        splits[TRAIN],
        splits[VALIDATION],
        C_grid=config.probe.C_grid,
        class_weight=config.probe.class_weight,
        standardize=config.probe.standardize,
        seed=config.seed,
        split_name=VALIDATION,
    )

    # Threshold on validation scores. Choosing it on test is one line and it
    # inflates every number downstream.
    validation_scores = probe.score(features[splits[VALIDATION]])
    budget = target_flag_rate if target_flag_rate is not None else 0.05
    threshold = threshold_for_flag_rate(validation_scores, budget)
    say(
        f"threshold {threshold:.6f} selected on validation for target flag rate "
        f"{budget:.4f} (measured on validation: "
        f"{flag_rate_at(validation_scores, threshold):.4f})"
    )

    operating_point = OperatingPoint(
        operating_point_id=operating_point_id,
        detector_id=detector_id,
        threshold=float(threshold),
        selected_on=VALIDATION,
        objective=f"flag_rate_budget={budget:g}",
        target_flag_rate=budget,
    )

    canary_scores = None
    canary_labels = None
    if canary_cache is not None:
        canary_scores = probe.score(canary_cache.matrix(variant))
        canary_labels = canary_cache.labels

    say("running the five controls")
    controls = run_controls(
        config,
        cache,
        variant,
        splits,
        threshold,
        fit.C,
        canary_scores=canary_scores,
        canary_labels=canary_labels,
    )
    for control in controls:
        say(
            f"  {'PASS' if control.passed else 'FAIL'} {control.control} "
            f"measured {control.measured:.4f} margin {control.margin:+.4f}"
        )

    # -- test is opened here, once, and the run records it ------------------- #
    say("scoring test (once)")
    test_index = splits[TEST]
    test_labels = labels[test_index]
    test_scores = probe.score(features[test_index])
    test_groups = groups[test_index]

    boot = config.validation.bootstrap_samples
    ci = config.validation.ci
    seed = config.seed

    metrics = build_warrant_metrics(
        config,
        test_labels,
        test_scores,
        threshold,
        groups=test_groups,
        is_hard_negative_set=is_hard_negative_set,
    )

    envelope = build_envelope(evalset, cache)
    tier = _TIER_OF_VARIANT.get(variant.split("-", 1)[0], AccessTier.T3_TEXT)
    key = WarrantKey(detector_id, operating_point_id, evalset.eval_set_id)
    run_id = f"run-{content_hash({'key': key.as_string(), 'variant': variant, 'envelope': evalset.envelope_id, 'config': config.config_hash})[:12]}"

    warrant = issue_or_refuse(
        config,
        key=key,
        detector_version=detector_version,
        operating_point=operating_point,
        metrics=metrics,
        envelope=envelope,
        controls=controls,
        access_tier=tier,
        n_test=int(test_index.size),
        base_rate=float(test_labels.mean()),
        validation_run_id=run_id,
        min_recall=min_recall,
        max_fpr_hard_negatives=max_fpr_hard_negatives,
        kappa=kappa,
        issued_at=started,
    )

    completed = utc_now()
    duration = time.perf_counter() - clock
    say(
        f"{'ISSUED' if warrant.status is WarrantStatus.VALID else 'REFUSED'} in "
        f"{duration:.2f}s"
    )

    return ValidationRun(
        run_id=run_id,
        detector_id=detector_id,
        variant=variant,
        eval_set_id=evalset.eval_set_id,
        envelope_id=evalset.envelope_id,
        started_at=started,
        completed_at=completed,
        duration_seconds=duration,
        probe_fit=fit,
        operating_point=operating_point,
        metrics=metrics,
        controls=controls,
        warrant=warrant,
        splits={name: int(idx.size) for name, idx in splits.items()},
        base_rate=float(test_labels.mean()),
        data_source=cache.data_source,
        test_scored=1,
        provenance=provenance(config),
    )


def validate_transferred(
    config: Config,
    evalset: EvalSet,
    cache: ExtractionCache,
    *,
    source: ValidationRun,
    probe,
    variant: Optional[str] = None,
    min_recall: Optional[float] = None,
    max_fpr_hard_negatives: Optional[float] = None,
    is_hard_negative_set: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> ValidationRun:
    """Score an **already-fitted** probe on a different envelope.

    This is the drift measurement, and it is a different experiment from
    :func:`validate`. ``validate`` fits a probe on the envelope it then scores,
    which answers *"how well can a probe do on this distribution?"*.
    ``validate_transferred`` takes the probe that was fitted somewhere else and
    asks *"what is **this** probe worth here?"* — which is the question a
    production system faces when its traffic moves, and the question Beat 4 is
    about.

    The distinction matters enough to be a separate function rather than a flag.
    Refitting on long context would produce a *better* number and a *weaker*
    claim: nobody retrains between one request and the next, so a refitted
    number describes a system that does not exist.

    Nothing is selected here. The threshold and the regularisation both come
    from the source run, which chose them on its own validation split, so no
    selection touches this envelope at all. ``operating_point`` is carried over
    unchanged and still records the split it was selected on.

    Args:
        config: Resolved config.
        evalset: The envelope to score on. Its whole content is treated as test
            — there is no split to derive, because nothing is being fitted.
        cache: Its extraction.
        source: The completed run whose probe and operating point are being
            transferred.
        probe: The fitted probe from that run.
        variant: Feature variant to score. Defaults to the source's.
        min_recall: Profile minimum to check against.
        max_fpr_hard_negatives: Declared maximum, where applicable.
        is_hard_negative_set: Whether this set is the hard-negative set.
        progress: Optional streaming callback.

    Returns:
        A :class:`ValidationRun` whose warrant is keyed to the **new** envelope
        and whose detector id is the source's, so the matrix cell is
        (same detector, new envelope) — which is exactly invariant 1's point.
    """
    started = utc_now()
    clock = time.perf_counter()

    def say(message: str) -> None:
        _LOG.info(message)
        if progress is not None:
            progress(message)

    chosen = variant or source.variant
    if cache.eval_set_hash != evalset.content_hash:
        raise ValueError(
            f"cache for {cache.eval_set_id} does not match the eval set's current "
            "contents; re-extract rather than scoring a stale cache"
        )

    say(
        f"transferring {source.detector_id} (fitted on {source.eval_set_id}) "
        f"to {evalset.eval_set_id} — nothing is refitted or reselected"
    )
    features = cache.matrix(chosen)
    labels = cache.labels
    groups = cache.question_ids
    threshold = source.operating_point.threshold
    scores = probe.score(features)

    say("running controls")
    # The whole set is test here, so the negative controls have no separate
    # holdout to score against and the padding evidence belongs to the source
    # extraction. Rather than inventing a split, the controls that need one are
    # carried from the source run: they describe the probe, and the probe is
    # unchanged. What is NOT carried is anything describing this envelope.
    controls = tuple(
        dataclasses.replace(
            control,
            detail=(
                f"{control.detail} [carried from the source validation on "
                f"{source.eval_set_id}: this control describes the fitted probe, "
                "which is unchanged by the transfer]"
            ),
        )
        for control in source.controls
    )

    say("scoring (once)")
    metrics = build_warrant_metrics(
        config,
        labels,
        scores,
        threshold,
        groups=groups,
        is_hard_negative_set=is_hard_negative_set,
    )

    envelope = build_envelope(evalset, cache)
    key = WarrantKey(
        source.detector_id, source.operating_point.operating_point_id, evalset.eval_set_id
    )
    run_id = "run-" + content_hash(
        {
            "key": key.as_string(),
            "variant": chosen,
            "envelope": evalset.envelope_id,
            "transferred_from": source.run_id,
            "config": config.config_hash,
        }
    )[:12]

    warrant = issue_or_refuse(
        config,
        key=key,
        detector_version=source.warrant.detector_version,
        operating_point=source.operating_point,
        metrics=metrics,
        envelope=envelope,
        controls=controls,
        access_tier=source.warrant.access_tier,
        n_test=len(evalset),
        base_rate=float(labels.mean()),
        validation_run_id=run_id,
        min_recall=min_recall,
        max_fpr_hard_negatives=max_fpr_hard_negatives,
        issued_at=started,
    )

    completed = utc_now()
    duration = time.perf_counter() - clock
    say(f"{warrant.status.value} in {duration:.2f}s")

    return ValidationRun(
        run_id=run_id,
        detector_id=source.detector_id,
        variant=chosen,
        eval_set_id=evalset.eval_set_id,
        envelope_id=evalset.envelope_id,
        started_at=started,
        completed_at=completed,
        duration_seconds=duration,
        probe_fit=source.probe_fit,
        operating_point=source.operating_point,
        metrics=metrics,
        controls=controls,
        warrant=warrant,
        splits={"train": 0, "validation": 0, "test": len(evalset)},
        base_rate=float(labels.mean()),
        data_source=cache.data_source,
        test_scored=1,
        provenance=provenance(config),
    )
