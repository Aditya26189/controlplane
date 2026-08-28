"""``/validate`` for stateless text detectors — regex, checksum, rule-based.

The activation-tier path in :mod:`controlplane.validation.runner` fits a probe, and three
of the five controls exist to catch faults in that fitting: a padding side that
makes activations meaningless, a split that leaks, features that carry the label.
A stateless detector fits nothing, so those three failure modes **cannot occur**
for it — and a control suite that reported them as passing would be claiming a
check it never made.

So they are recorded as **inapplicable, with the reason stated in the record**
(``ControlResult.applicable``), and the warrant carries `controls_run: 2` where
the probe's carries five. A reader can see that a rule-based detector's warrant
rests on less evidence than a probe's, which is true and should be visible.

That field is the one escape hatch this design could grow, so it is fenced three
ways: applicability is declared **per detector class in code**, never per run;
an inapplicable control must carry no verdict and no margin; and it must state
why the mechanism cannot exist. "Not applicable" without a reason is an override
with better manners. ``DECISIONS.md`` 034.

**Threshold selection.** A stateless detector's threshold is a *declared*
property of the detector — its confidence floor — not something fitted to an
eval set. So nothing is selected on any split here, and ``selected_on`` records
``"calibration"``. That is stronger than selecting on validation, not weaker:
there is no selection to leak.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from ..config import Config, provenance
from ..model import (
    AccessTier,
    ControlResult,
    OperatingPoint,
    WarrantKey,
    WarrantMetrics,
    WarrantStatus,
    content_hash,
    utc_now,
)
from .controls import CANARY, DETERMINISM, LABEL_SHUFFLE, NULL_FEATURE, PADDING_FAULT
from .controls import canary_control, determinism_control
from .evalsets import EvalSet
from .issuance import issue_or_refuse
from .metrics_builder import build_warrant_metrics
from .runner import ValidationRun, build_envelope


__all__ = ["TextDetector", "validate_text_detector"]

_LOG = logging.getLogger(__name__)


class TextDetector(Protocol):
    """What a stateless text detector must provide."""

    detector_id: str
    detector_version: str
    access_tier: AccessTier

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Score each message in [0, 1], higher meaning more likely positive."""


#: Why each control cannot apply to a detector that fits nothing. Declared here,
#: per detector class, rather than passed in per run -- a runtime argument is
#: exactly how "not applicable" becomes "not checked, and nobody noticed".
_STATELESS_INAPPLICABLE = {
    PADDING_FAULT: (
        "no activations are extracted for a rule-based detector, so there is no "
        "batched forward pass whose padding side could be wrong. The fault this "
        "control injects cannot exist here."
    ),
    LABEL_SHUFFLE: (
        "nothing is fitted, so there are no training labels to permute. A "
        "rule-based detector's output does not depend on any label, which is the "
        "condition this control exists to verify by permutation."
    ),
    NULL_FEATURE: (
        "there is no learned feature representation to replace with noise. The "
        "detector reads the text directly, so the failure mode -- a probe "
        "scoring above chance on features carrying no signal -- has no analogue."
    ),
}


def _inapplicable(control: str) -> ControlResult:
    """Record a control that cannot apply, with the reason in the record."""
    return ControlResult(
        control=control,
        passed=True,
        measured=0.0,
        expected="not applicable to a stateless detector",
        margin=0.0,
        detail=_STATELESS_INAPPLICABLE[control],
        applicable=False,
    )


def validate_text_detector(
    config: Config,
    evalset: EvalSet,
    detector: TextDetector,
    *,
    operating_point_id: str = "P-declared",
    threshold: float,
    canary_evalset: Optional[EvalSet] = None,
    min_recall: Optional[float] = None,
    max_fpr_hard_negatives: Optional[float] = None,
    is_hard_negative_set: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> ValidationRun:
    """Validate a stateless detector on one eval set, and issue or refuse.

    Args:
        config: Resolved config.
        evalset: The set to measure on. Its content hash becomes the envelope id.
        detector: Anything satisfying :class:`TextDetector`.
        operating_point_id: Identity of the declared threshold.
        threshold: The detector's declared confidence floor. **Not fitted here**
            — see the module docstring.
        canary_evalset: The tripwire set. Absent means the canary control fails
            rather than being skipped, exactly as on the probe path.
        min_recall: Profile minimum, if one applies.
        max_fpr_hard_negatives: Declared maximum FPR on the hard-negative set.
            Required on a single-class envelope, where it is the only claim
            available.
        is_hard_negative_set: Whether this eval set *is* the hard-negative set.
            Only then does its FPR populate ``fpr_hard_negatives`` and face the
            declared maximum; otherwise the within-set FPR is reported under
            ``fpr``, because they are different claims.
        progress: Optional streaming callback.

    Returns:
        A completed :class:`ValidationRun`.
    """
    started = utc_now()
    clock = time.perf_counter()

    def say(message: str) -> None:
        _LOG.info(message)
        if progress is not None:
            progress(message)

    # An envelope is a distribution PLUS a label definition, and a detector can
    # only be warranted on a set whose labels mean what it detects. Checked
    # before scoring, because everything after this point is arithmetically
    # correct whether or not the meanings match (``DECISIONS.md`` 089).
    from ..evalsets.categories import require_compatible

    require_compatible(getattr(detector, "category", None), evalset.eval_set_id)

    say(f"scoring {len(evalset)} items with {detector.detector_id}")
    texts = [item.prompt for item in evalset.items]
    scores = detector.score(texts)
    labels = evalset.labels
    groups = evalset.question_ids
    single_class = len(set(labels.tolist())) < 2

    operating_point = OperatingPoint(
        operating_point_id=operating_point_id,
        detector_id=detector.detector_id,
        threshold=float(threshold),
        selected_on="calibration",
        objective="declared detector confidence floor; not fitted to any eval set",
    )

    say("running controls (three are inapplicable to a stateless detector)")
    canary_scores = None
    canary_labels = None
    if canary_evalset is not None:
        canary_scores = detector.score([i.prompt for i in canary_evalset.items])
        canary_labels = canary_evalset.labels

    results = {
        PADDING_FAULT: _inapplicable(PADDING_FAULT),
        LABEL_SHUFFLE: _inapplicable(LABEL_SHUFFLE),
        NULL_FEATURE: _inapplicable(NULL_FEATURE),
        CANARY: canary_control(canary_scores, canary_labels, threshold),
        DETERMINISM: determinism_control(lambda: detector.score(texts)),
    }
    controls = tuple(results[name] for name in config.validation.controls)
    for control in controls:
        say(
            f"  {'n/a ' if not control.applicable else ('PASS' if control.passed else 'FAIL')}"
            f" {control.control}"
        )

    say("scoring test (once)")
    metrics = build_warrant_metrics(
        config,
        labels,
        scores,
        threshold,
        groups=groups,
        is_hard_negative_set=is_hard_negative_set,
    )
    if metrics.recall is None:
        say(
            "single-class envelope: AUROC, recall and precision are undefined "
            "here, so this warrant claims FPR only"
        )

    envelope = build_envelope_from_text(evalset)
    key = WarrantKey(detector.detector_id, operating_point_id, evalset.eval_set_id)
    run_id = "run-" + content_hash(
        {
            "key": key.as_string(),
            "envelope": evalset.envelope_id,
            "threshold": float(threshold),
            "config": config.config_hash,
        }
    )[:12]

    warrant = issue_or_refuse(
        config,
        key=key,
        detector_version=detector.detector_version,
        operating_point=operating_point,
        metrics=metrics,
        envelope=envelope,
        controls=controls,
        access_tier=detector.access_tier,
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

    from ..detectors.probe import ProbeFit

    return ValidationRun(
        run_id=run_id,
        detector_id=detector.detector_id,
        variant="text",
        eval_set_id=evalset.eval_set_id,
        envelope_id=evalset.envelope_id,
        # Stateless: reads text, holds no weights tied to a model. Recorded
        # explicitly so a model change can be seen to have left it alone,
        # rather than leaving a reader to infer it from a missing field.
        model_name="none (stateless text detector)",
        started_at=started,
        completed_at=completed,
        duration_seconds=duration,
        probe_fit=ProbeFit(
            C=1.0,
            selected_on="calibration",
            selection_scores={},
            n_train=0,
            n_features=0,
            class_weight="none",
            base_rate_train=float(labels.mean()),
        ),
        operating_point=operating_point,
        metrics=metrics,
        controls=controls,
        warrant=warrant,
        splits={"train": 0, "validation": 0, "test": len(evalset)},
        base_rate=float(labels.mean()),
        data_source=evalset.data_source,
        test_scored=1,
        provenance=provenance(config),
    )


def build_envelope_from_text(evalset: EvalSet):
    """Envelope for a text set, using message length in characters.

    The activation path measures token length; a stateless detector never
    tokenises, so character length is the analogue that is actually available.
    Both are "how long is the input", which is the envelope feature that matters
    (``SPEC.md`` §5.1) — measured in the unit the detector can see.
    """
    import numpy as np

    from ..validation.evalsets import ExtractionCache

    lengths = np.array([len(item.prompt) for item in evalset.items], dtype=float)
    stub = ExtractionCache(
        eval_set_id=evalset.eval_set_id,
        eval_set_hash=evalset.content_hash,
        model_name="none (stateless text detector)",
        layer=-1,
        data_source=evalset.data_source,
        features={"text": np.zeros((len(evalset), 1))},
        labels=evalset.labels,
        question_ids=evalset.question_ids,
        token_lengths=lengths,
    )
    return build_envelope(evalset, stub)
