"""The tier ladder: what each level of model access actually buys.

Runs :func:`~controlplane.validation.runner.validate` once per tier variant **from one
cached extraction**, so every rung is measured on the same items, the same
splits and the same seed. Anything else and a difference between rungs could be
a difference between runs.

The question this answers is commercial, not academic. An enterprise deciding
whether to self-host a model — versus calling an API that returns text and
maybe logprobs — is deciding whether T1 access is worth paying for. The ladder
puts a number and an interval on that, and **the interval is the point**: if
T1's and T3's intervals overlap, the honest answer is that on this envelope the
extra access bought nothing measurable, and that finding is as useful as the
opposite one.

``KICKOFF.md`` is explicit that a small tier gap is a finding to report, not a
result to improve. Nothing here selects, tunes or orders the tiers.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..config import Config
from ..model import Metric, WarrantStatus, to_jsonable
from .evalsets import EvalSet, ExtractionCache
from .runner import ValidationRun, validate

__all__ = ["TierLadder", "run_ablation"]

_LOG = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TierLadder:
    """Every tier variant measured on one envelope, with intervals.

    Args:
        eval_set_id: The set every rung was measured on.
        envelope_id: Its content hash.
        data_source: ``"measured"`` or ``"synthetic"``, carried from the cache so
            a plot or table built from this can say which it is.
        runs: One completed validation per variant, in variant order.
        base_rate: Positive-class prevalence on test. Reported beside every
            AUROC, because at an 85% correct rate a constant predictor scores
            0.85 accuracy and 0.5 AUROC.
    """

    eval_set_id: str
    envelope_id: str
    data_source: str
    runs: tuple[ValidationRun, ...]
    base_rate: float

    def by_variant(self) -> dict[str, ValidationRun]:
        """The runs, keyed by variant name."""
        return {run.variant: run for run in self.runs}

    def metric(self, variant: str, name: str) -> Metric:
        """One metric from one rung.

        Raises:
            KeyError: If the variant was not run, rather than returning a
                default. A missing rung is a hole in the ladder, and a hole that
                renders as a zero reads as a measurement.
        """
        run = self.by_variant()[variant]
        for metric in run.metrics.all_metrics():
            if metric.name == name:
                return metric
        raise KeyError(f"{variant} has no metric {name!r}")

    def intervals_overlap(self, a: str, b: str, metric: str = "auroc") -> bool:
        """Whether two rungs' intervals overlap on a metric.

        The question the ladder exists to answer. Overlapping intervals mean the
        data does not distinguish the two tiers, and saying so is more useful
        than reporting a point-estimate ordering the sample cannot support.
        """
        first = self.metric(a, metric)
        second = self.metric(b, metric)
        return not (
            first.ci_high < second.ci_low or second.ci_high < first.ci_low
        )

    def to_payload(self) -> dict[str, Any]:
        """JSON-serialisable form for ``results/``."""
        return {
            "eval_set_id": self.eval_set_id,
            "envelope_id": self.envelope_id,
            "data_source": self.data_source,
            "base_rate": self.base_rate,
            "variants": [run.variant for run in self.runs],
            "rungs": {
                run.variant: {
                    "access_tier": run.warrant.access_tier.name,
                    "warrant_status": run.warrant.status.value,
                    "status_reason": run.warrant.status_reason,
                    "n_test": run.warrant.n_test,
                    "C": run.probe_fit.C,
                    "threshold": run.operating_point.threshold,
                    "metrics": to_jsonable(run.metrics),
                    "controls": to_jsonable(list(run.controls)),
                    "duration_seconds": run.duration_seconds,
                }
                for run in self.runs
            },
        }

    def render(self) -> str:
        """A text table, precision and recall always together (invariant 5)."""
        header = (
            f"{'variant':<24} {'tier':<16} {'AUROC':<26} {'recall':<26} "
            f"{'precision':<26} {'warrant'}"
        )
        lines = [
            f"tier ladder on {self.eval_set_id} [{self.envelope_id}] "
            f"— base rate {self.base_rate:.4f}, data_source {self.data_source}",
            header,
            "-" * len(header),
        ]
        for run in self.runs:
            lines.append(
                f"{run.variant:<24} {run.warrant.access_tier.name:<16} "
                f"{run.metrics.auroc.render(3):<26} "
                f"{run.metrics.recall.render(3):<26} "
                f"{run.metrics.precision.render(3):<26} "
                f"{run.warrant.status.value}"
            )
        return "\n".join(lines)


def run_ablation(
    config: Config,
    evalset: EvalSet,
    cache: ExtractionCache,
    *,
    detector_prefix: str,
    detector_version: str,
    variants: Optional[Sequence[str]] = None,
    target_flag_rate: float = 0.05,
    canary_cache: Optional[ExtractionCache] = None,
    min_recall: Optional[float] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> TierLadder:
    """Validate every tier variant on one envelope, from one extraction.

    Each variant gets its **own detector id**, because a warrant is keyed by
    detector and the mean-pooled probe and the max-of-rolling-means probe are
    different detectors that happen to read the same activations. Giving them
    one id would put them in one matrix cell, and the whole point of Phase 4 is
    that one can be refused while the other holds.

    Args:
        config: Resolved config.
        evalset: The set to measure on.
        cache: Its extraction. Every rung uses the same one.
        detector_prefix: Base detector id; the variant is appended.
        detector_version: Semver plus weights hash.
        variants: Which rungs to run. Defaults to everything in the cache.
        target_flag_rate: Budget for threshold selection, on validation.
        canary_cache: Extraction for the canary set, if available.
        min_recall: Profile minimum to check against, if any.
        progress: Optional streaming callback.

    Returns:
        A :class:`TierLadder`.
    """
    chosen = tuple(variants) if variants is not None else cache.variants
    runs: list[ValidationRun] = []
    for variant in chosen:
        _LOG.info("ablation: %s on %s", variant, evalset.eval_set_id)
        runs.append(
            validate(
                config,
                evalset,
                cache,
                variant=variant,
                detector_id=f"{detector_prefix}-{variant}",
                detector_version=detector_version,
                target_flag_rate=target_flag_rate,
                canary_cache=canary_cache,
                min_recall=min_recall,
                progress=progress,
            )
        )

    issued = [r.variant for r in runs if r.warrant.status is WarrantStatus.VALID]
    refused = [r.variant for r in runs if r.warrant.status is WarrantStatus.REFUSED]
    _LOG.info(
        "ablation on %s: %d issued %s, %d refused %s",
        evalset.eval_set_id,
        len(issued),
        issued,
        len(refused),
        refused,
    )
    return TierLadder(
        eval_set_id=evalset.eval_set_id,
        envelope_id=evalset.envelope_id,
        data_source=cache.data_source,
        runs=tuple(runs),
        base_rate=float(np.mean([r.base_rate for r in runs])) if runs else 0.0,
    )
