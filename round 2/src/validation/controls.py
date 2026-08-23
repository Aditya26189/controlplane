"""The five controls. ``SPEC.md`` §2.1.

All five run on every validation and **any failure refuses the warrant**. There
is no argument to this module that suppresses one, and no code path that issues
a warrant past a failure — that is invariant 3, and
:class:`~src.model.warrant.Warrant` refuses to be constructed in violation of it
even if something here were changed.

Two of the five are **negative controls**. Label shuffle and null feature assert
that this pipeline *can produce a null result when there is no signal*. A
pipeline that cannot fail cannot be trusted when it succeeds, and that sentence
is the project's thesis applied to its own code.

Every control reports a **measured margin**: the signed distance to the nearest
edge of its pass condition. "Control passed" is an assertion; "label shuffle
scored 0.497, inside [0.45, 0.55] with 0.047 to spare" is evidence, and the
difference is what a judge is actually asking about.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional, Sequence

import numpy as np

from ..config import Config
from ..detectors.probe import LinearProbe, select_regularisation
from ..model import ControlResult
from .evalsets import ExtractionCache, PaddingEvidence
from .stats import MeasurementError, auroc

__all__ = [
    "CANARY",
    "DETERMINISM",
    "LABEL_SHUFFLE",
    "NULL_FEATURE",
    "PADDING_FAULT",
    "effective_null_band",
    "run_controls",
]

_LOG = logging.getLogger(__name__)

PADDING_FAULT = "padding_fault"
LABEL_SHUFFLE = "label_shuffle"
NULL_FEATURE = "null_feature"
CANARY = "canary"
DETERMINISM = "determinism"

#: Pass thresholds for the padding equivalence check. Scale-invariant on
#: purpose: an absolute L2 distance between residual-stream vectors depends on
#: the layer's magnitude, so a threshold that passes at layer 8 can fail at
#: layer 23 for no reason connected to padding.
_PADDING_MAX_REL_L2 = 0.1
_PADDING_MIN_COSINE = 0.999


#: A negative control's band has to be wide enough that ordinary sampling noise
#: does not trip it. Below this many standard errors, a "failure" is more likely
#: to be the holdout being small than the pipeline leaking, and the control is
#: reporting noise rather than evidence. See DECISIONS.md 029 for the measured
#: false-failure rates that produced this number.
_MIN_BAND_IN_NULL_SE = 2.0


def _null_auroc_se(labels: np.ndarray) -> float:
    """Standard error of AUROC under the null, by Hanley-McNeil.

    ``SE = sqrt((n_pos + n_neg + 1) / (12 * n_pos * n_neg))`` at AUC = 0.5. Used
    to say how many standard errors wide the configured band is, so a reader can
    tell a real control failure from a small holdout.

    Quantity estimated: the sampling standard deviation of AUROC when there is
    no signal. Propagation: none -- this is the closed form at AUC = 0.5, which
    is exactly the hypothesis a negative control is testing.
    """
    labels = np.asarray(labels)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("inf")
    return math.sqrt((n_pos + n_neg + 1) / (12.0 * n_pos * n_neg))


def effective_null_band(
    labels: np.ndarray, band: tuple[float, float], repeats: int = 1
) -> tuple[tuple[float, float], str]:
    """Widen the configured null band to at least 2 null standard errors.

    A negative control asserts *"AUROC is consistent with 0.5"*. Whether an
    observed value is consistent with 0.5 depends on the sampling noise, and
    that depends on ``n`` — so a **fixed** band is only a valid test at one
    particular holdout size. Measured against the Hanley-McNeil null SE, the
    configured band of +/-0.05 is:

        n =  150  ->  +/-0.76 SE  ->  ~45% chance of failing with no fault present
        n =  600  ->  +/-1.52 SE  ->  ~13%
        n = 1200  ->  +/-2.15 SE  ->  ~3%

    At the sizes this project works with, the declared band would refuse roughly
    one warrant in eight for no reason but noise. That is not a conservative
    error: a control that cries wolf gets switched off, and a suite nobody
    believes protects nothing.

    So the effective band is the **wider** of the configured band and
    +/-2 null SE. It never becomes stricter than declared, and it never becomes
    looser than the noise floor. Power against a real fault is essentially
    unaffected, because leakage does not produce an AUROC of 0.56 — it produces
    one far outside any of these bands.

    Quantity: the sampling standard deviation of AUROC under H0 (AUC = 0.5).
    Propagation: Hanley-McNeil closed form,
    ``SE = sqrt((n_pos + n_neg + 1) / (12 * n_pos * n_neg))``; band half-width
    ``max(configured_half_width, 2 * SE)``. ``DECISIONS.md`` 029.

    Args:
        labels: Holdout labels the control is scored against.
        band: The configured band from ``config.validation.null_control_band``.
        repeats: How many independent draws the control averages. The SE of the
            mean falls as ``1/sqrt(repeats)``, which is the primary fix; this
            widening is the floor that remains when repeats cannot rescue a very
            small holdout.

    Returns:
        ``((low, high), note)`` — the band actually applied, and a sentence
        stating both it and the configured one, for the control's detail.
    """
    se = _null_auroc_se(labels) / math.sqrt(max(1, repeats))
    configured_half = min(0.5 - band[0], band[1] - 0.5)
    if not math.isfinite(se) or se == 0:
        return band, "holdout has only one class; the null band is meaningless here"
    floor_half = _MIN_BAND_IN_NULL_SE * se
    if floor_half <= configured_half:
        return (
            band,
            f"null SE {se:.4f} at n={labels.size}; configured band is "
            f"+/-{configured_half / se:.2f} SE, which is adequate",
        )
    widened = (0.5 - floor_half, 0.5 + floor_half)
    return (
        widened,
        f"null SE {se:.4f} at n={labels.size}; configured band +/-{configured_half:g} "
        f"is only +/-{configured_half / se:.2f} SE, so the applied band is widened "
        f"to +/-{floor_half:.4f} (2 SE) to keep the control from failing on noise "
        f"(DECISIONS.md 029)"
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean row-wise cosine similarity between two matrices."""
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den = np.where(den == 0, np.finfo(float).eps, den)
    return float(np.mean(num / den))


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Mean row-wise L2 distance, normalised by the reference's magnitude."""
    diff = np.linalg.norm(a - b, axis=1)
    scale = np.linalg.norm(a, axis=1)
    scale = np.where(scale == 0, np.finfo(float).eps, scale)
    return float(np.mean(diff / scale))


def padding_fault_control(evidence: Optional[PaddingEvidence]) -> ControlResult:
    """Compare left-padded and right-padded batching against unbatched scoring.

    The fault this exists for is invisible. With right padding, position −1 of a
    batched sequence is a pad token, so every activation is read from nothing;
    the probe still trains, still scores, and lands near 0.5 AUROC, which reads
    as *"the idea doesn't work"* rather than as a bug. It has cost this design
    before, which is why it is checked on every run rather than once.

    Two claims are checked, and **both must hold**:

    * the left-padded batch reproduces unbatched scoring — relative L2 ≤ 0.1 and
      cosine ≥ 0.999;
    * the right-padded batch **does not** — if it did, the check has no power,
      because a test that passes whatever you feed it proves nothing.

    The second half is what makes this a fault injection rather than an
    assertion. It is also the moment in the demo: the system breaking its own
    test on purpose, every run, to confirm the test still catches the fault it
    exists for.

    Args:
        evidence: Activations captured three ways at extraction time.

    Returns:
        A :class:`ControlResult` whose margin is the smaller of the two
        distances to the pass edges.
    """
    if evidence is None:
        return ControlResult(
            control=PADDING_FAULT,
            passed=False,
            measured=0.0,
            expected=(
                f"left-padded vs unbatched: rel-L2 <= {_PADDING_MAX_REL_L2}, "
                f"cosine >= {_PADDING_MIN_COSINE}; right-padded must be rejected"
            ),
            margin=-1.0,
            detail=(
                "no padding evidence in the cache. The extraction did not capture "
                "it, so there is nothing establishing that these features were "
                "produced with left padding. Refusing rather than assuming."
            ),
        )

    left_l2 = _relative_l2(evidence.unbatched, evidence.left_padded)
    left_cos = _cosine(evidence.unbatched, evidence.left_padded)
    right_l2 = _relative_l2(evidence.unbatched, evidence.right_padded)
    right_cos = _cosine(evidence.unbatched, evidence.right_padded)

    left_ok = left_l2 <= _PADDING_MAX_REL_L2 and left_cos >= _PADDING_MIN_COSINE
    right_rejected = not (
        right_l2 <= _PADDING_MAX_REL_L2 and right_cos >= _PADDING_MIN_COSINE
    )

    # Margin: how much room the left-padded case has, and how clearly the
    # right-padded case fails. The reported margin is the weaker of the two,
    # because the control is only as strong as its weaker half.
    left_margin = min(_PADDING_MAX_REL_L2 - left_l2, left_cos - _PADDING_MIN_COSINE)
    right_margin = max(right_l2 - _PADDING_MAX_REL_L2, _PADDING_MIN_COSINE - right_cos)
    passed = left_ok and right_rejected
    margin = min(left_margin, right_margin) if passed else -abs(
        min(left_margin, right_margin)
    )

    return ControlResult(
        control=PADDING_FAULT,
        passed=passed,
        measured=left_cos,
        expected=(
            f"left-padded vs unbatched: rel-L2 <= {_PADDING_MAX_REL_L2}, "
            f"cosine >= {_PADDING_MIN_COSINE}; right-padded must be rejected"
        ),
        margin=float(margin),
        detail=(
            f"left-padded: rel-L2 {left_l2:.2e}, cosine {left_cos:.6f} -> "
            f"{'matches' if left_ok else 'DOES NOT MATCH'} unbatched. "
            f"right-padded (deliberate fault): rel-L2 {right_l2:.3f}, cosine "
            f"{right_cos:.4f} -> {'REJECTED as required' if right_rejected else 'ACCEPTED, which means this check has no power'}. "
            f"{evidence.n_prompts} prompts, max {evidence.max_pad_tokens} pad tokens."
        ),
    )


def label_shuffle_control(
    features: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    holdout_index: np.ndarray,
    *,
    band: tuple[float, float],
    C: float,
    class_weight: str,
    seed: int,
    repeats: int = 5,
) -> ControlResult:
    """Retrain on permuted labels; mean AUROC over repeats must land at chance.

    A **negative control**. If a probe fitted on shuffled labels still scores
    above the band, the pipeline is reading something it should not — leakage
    between splits, an index misalignment, or a feature that encodes the label
    directly. Each of those produces a *good-looking* result on real labels, so
    this is one of the few checks that catches them.

    **Averaged over ``repeats`` independent permutations.** A single permutation
    is one draw from a distribution with SE ≈ 0.032 at a holdout of 600, so it
    lands outside a ±0.05 band about 13% of the time with nothing wrong. A
    control that cries wolf one run in eight gets switched off, and a suite
    nobody believes protects nothing. Averaging shrinks the SE by
    ``sqrt(repeats)`` and turns the band into a real bar. ``DECISIONS.md`` 029.

    Labels are permuted **within the training split only** and the holdout keeps
    its true labels. Permuting everywhere would test nothing: a probe fitted on
    shuffled labels and scored against shuffled labels can still find the
    consistent mapping between them.

    Args:
        features: Full feature matrix.
        labels: True labels.
        train_index: Rows to fit on.
        holdout_index: Rows to score on, with true labels.
        band: Acceptable AUROC range, from ``config.validation.null_control_band``.
        C: Regularisation, fixed rather than re-selected — selecting on a
            shuffled run would be selecting for the noise this control measures.
        class_weight: Imbalance handling, matching the real fit.
        seed: Base seed; each repeat uses a distinct derived seed.
        repeats: Number of independent permutations to average.

    Returns:
        A :class:`ControlResult` reporting the mean AUROC across repeats and its
        distance to the nearer band edge.
    """
    values: list[float] = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + 1000 * repeat)
        shuffled = labels.copy()
        shuffled[train_index] = rng.permutation(shuffled[train_index])
        try:
            probe = LinearProbe(C, class_weight=class_weight, seed=seed).fit(
                features, shuffled, train_index
            )
            values.append(
                float(auroc(labels[holdout_index], probe.score(features[holdout_index])))
            )
        except (MeasurementError, ValueError) as exc:
            return ControlResult(
                control=LABEL_SHUFFLE,
                passed=False,
                measured=0.0,
                expected=f"mean AUROC over {repeats} permutations in "
                f"[{band[0]}, {band[1]}]",
                margin=-1.0,
                detail=f"control could not be run: {exc}",
            )

    value = float(np.mean(values))
    spread = float(np.std(values))
    (low, high), note = effective_null_band(labels[holdout_index], band, repeats)
    inside = low <= value <= high
    margin = min(value - low, high - value)
    return ControlResult(
        control=LABEL_SHUFFLE,
        passed=inside,
        measured=value,
        expected=f"mean AUROC over {repeats} permutations in [{low:.4f}, {high:.4f}]",
        margin=float(margin),
        detail=(
            f"train labels permuted (n={train_index.size}), holdout labels intact "
            f"(n={holdout_index.size}); mean AUROC {value:.4f} over {repeats} "
            f"permutations, sd {spread:.4f}, per-run values "
            f"{[round(v, 4) for v in values]}. {note}. "
            + (
                "A probe fitted on shuffled labels scoring above the band means "
                "the pipeline is reading something it should not."
                if not inside
                else "Pipeline produces a null result when there is no signal."
            )
        ),
    )


def null_feature_control(
    features: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    holdout_index: np.ndarray,
    *,
    band: tuple[float, float],
    C: float,
    class_weight: str,
    seed: int,
    repeats: int = 5,
) -> ControlResult:
    """Replace features with mean/variance-matched Gaussian noise.

    The second **negative control**, catching a different fault from the label
    shuffle. Shuffling tests whether the *labels* are being read illegitimately;
    this tests whether the *features* carry any information at all, by replacing
    them with noise matched in its first two moments and checking the score
    collapses to chance.

    Matching mean and variance per column matters. Unmatched noise is
    distinguishable from real features by scale alone, and a scaler fitted on it
    behaves differently, so the control would be testing the wrong thing.

    Averaged over ``repeats`` independent noise draws, for the same reason the
    label shuffle is. ``DECISIONS.md`` 029.

    Args:
        features: Full feature matrix, used only for its moments.
        labels: True labels, kept intact.
        train_index: Rows to fit on.
        holdout_index: Rows to score on.
        band: Acceptable AUROC range.
        C: Regularisation, fixed.
        class_weight: Imbalance handling.
        seed: Base seed; each repeat uses a distinct derived seed.
        repeats: Number of independent noise draws to average.

    Returns:
        A :class:`ControlResult`.
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    scale = np.where(std == 0, 1e-9, std)

    values: list[float] = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + 2000 * repeat)
        noise = rng.normal(loc=mean, scale=scale, size=features.shape)
        try:
            probe = LinearProbe(C, class_weight=class_weight, seed=seed).fit(
                noise, labels, train_index
            )
            values.append(
                float(auroc(labels[holdout_index], probe.score(noise[holdout_index])))
            )
        except (MeasurementError, ValueError) as exc:
            return ControlResult(
                control=NULL_FEATURE,
                passed=False,
                measured=0.0,
                expected=f"mean AUROC over {repeats} noise draws in "
                f"[{band[0]}, {band[1]}]",
                margin=-1.0,
                detail=f"control could not be run: {exc}",
            )

    value = float(np.mean(values))
    spread = float(np.std(values))
    (low, high), note = effective_null_band(labels[holdout_index], band, repeats)
    inside = low <= value <= high
    margin = min(value - low, high - value)
    return ControlResult(
        control=NULL_FEATURE,
        passed=inside,
        measured=value,
        expected=f"mean AUROC over {repeats} noise draws in [{low:.4f}, {high:.4f}]",
        margin=float(margin),
        detail=(
            f"features replaced by Gaussian noise matched per column in mean and "
            f"variance; labels intact; mean AUROC {value:.4f} over {repeats} draws, "
            f"sd {spread:.4f}. {note}."
        ),
    )


def canary_control(
    scores: Optional[np.ndarray],
    labels: Optional[np.ndarray],
    threshold: float,
    *,
    canary_set_id: str = "canary-20",
) -> ControlResult:
    """Known positives the detector must always catch. Recall must be exactly 1.

    A regression tripwire rather than a measurement. The canary items are chosen
    to be unambiguous, so anything less than perfect recall means something
    broke between the last run and this one — a changed threshold, a reordered
    feature matrix, a model update.

    Returns a *failed* control when the canary set is absent rather than skipping
    it. A control that silently does not run is worse than one that fails: the
    run still reports five controls and one of them is a lie.

    Args:
        scores: Detector scores on the canary set.
        labels: Canary labels; all should be 1.
        threshold: The operating point.
        canary_set_id: Name, for the message.

    Returns:
        A :class:`ControlResult`.
    """
    if scores is None or labels is None:
        return ControlResult(
            control=CANARY,
            passed=False,
            measured=0.0,
            expected="recall == 1.0",
            margin=-1.0,
            detail=(
                f"{canary_set_id} not available to this run, so the tripwire did "
                "not fire and did not not-fire. Refusing rather than reporting a "
                "control that never ran."
            ),
        )
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    positives = labels == 1
    if not positives.any():
        return ControlResult(
            control=CANARY,
            passed=False,
            measured=0.0,
            expected="recall == 1.0",
            margin=-1.0,
            detail=f"{canary_set_id} contains no positive items; it cannot be a tripwire",
        )
    caught = int(np.sum((scores >= threshold) & positives))
    total = int(positives.sum())
    recall = caught / total
    passed = recall >= 1.0
    return ControlResult(
        control=CANARY,
        passed=passed,
        measured=float(recall),
        expected="recall == 1.0",
        margin=float(recall - 1.0),
        detail=(
            f"{caught}/{total} canary positives caught at threshold {threshold:.6f}"
            + ("" if passed else f"; {total - caught} missed, which is a regression")
        ),
    )


def determinism_control(
    rescore: Callable[[], np.ndarray],
    *,
    coefficients: Optional[Callable[[], np.ndarray]] = None,
) -> ControlResult:
    """Run scoring twice at a fixed seed; results must be bit-identical.

    Not approximately equal. Bit-identical, because the claim being protected is
    that two runs at one seed produce identical numbers, and "close enough"
    hides exactly the nondeterminism that makes a published number
    irreproducible — thread-count-dependent reductions, dictionary ordering, an
    unseeded shuffle.

    Args:
        rescore: Callable performing the full fit-and-score, returning scores.
        coefficients: Optional callable returning the fitted weights, so the
            check covers the fit as well as the scoring.

    Returns:
        A :class:`ControlResult` whose measured value is the maximum absolute
        difference between the two runs.
    """
    first = np.asarray(rescore(), dtype=float)
    second = np.asarray(rescore(), dtype=float)
    if first.shape != second.shape:
        return ControlResult(
            control=DETERMINISM,
            passed=False,
            measured=float("inf"),
            expected="bit-identical scores across two runs at one seed",
            margin=-1.0,
            detail=f"shapes differ between runs: {first.shape} vs {second.shape}",
        )
    identical = bool(np.array_equal(first, second))
    max_diff = float(np.max(np.abs(first - second))) if first.size else 0.0

    coefficient_note = ""
    if coefficients is not None:
        c1 = np.asarray(coefficients(), dtype=float)
        c2 = np.asarray(coefficients(), dtype=float)
        coefficients_identical = bool(np.array_equal(c1, c2))
        identical = identical and coefficients_identical
        coefficient_note = (
            f"; coefficients {'identical' if coefficients_identical else 'DIFFER'}"
        )

    return ControlResult(
        control=DETERMINISM,
        passed=identical,
        measured=max_diff,
        expected="bit-identical scores across two runs at one seed",
        margin=0.0 if identical else -max_diff,
        detail=(
            f"max |difference| = {max_diff:.3e} over {first.size} scores"
            f"{coefficient_note}"
        ),
    )


def run_controls(
    config: Config,
    cache: ExtractionCache,
    variant: str,
    splits: dict[str, np.ndarray],
    threshold: float,
    C: float,
    *,
    canary_scores: Optional[np.ndarray] = None,
    canary_labels: Optional[np.ndarray] = None,
) -> tuple[ControlResult, ...]:
    """Run all five controls and return them in the configured order.

    Every control runs even after one has failed. A run that short-circuits on
    the first failure reports fewer than five results, and the missing ones read
    as "not applicable" rather than "not measured" — which is the ambiguity the
    whole project exists to remove.

    Args:
        config: Resolved config; supplies the band, the seed and the control list.
        cache: The extraction being validated.
        variant: Which tier variant to run the controls against.
        splits: Row indices per split.
        threshold: Operating point, selected on validation.
        C: Regularisation chosen on validation.
        canary_scores: Scores on the canary set, if available.
        canary_labels: Canary labels, if available.

    Returns:
        Five :class:`ControlResult` objects, in ``config.validation.controls``
        order.

    Raises:
        ValueError: If the configured control list is not the expected five. The
            config already pins this; the second check is here because this is
            the function a future change would route around.
    """
    features = cache.matrix(variant)
    labels = cache.labels
    train = splits["train"]
    holdout = splits["validation"]
    band = (config.validation.null_control_band[0], config.validation.null_control_band[1])
    seed = config.seed
    class_weight = config.probe.class_weight

    def rescore() -> np.ndarray:
        probe = LinearProbe(C, class_weight=class_weight, seed=seed).fit(
            features, labels, train
        )
        return probe.score(features[splits["test"]])

    def coefficients() -> np.ndarray:
        probe = LinearProbe(C, class_weight=class_weight, seed=seed).fit(
            features, labels, train
        )
        return probe.coefficients()

    results = {
        PADDING_FAULT: padding_fault_control(cache.padding_evidence),
        LABEL_SHUFFLE: label_shuffle_control(
            features, labels, train, holdout,
            band=band, C=C, class_weight=class_weight, seed=seed,
            repeats=config.validation.null_control_repeats,
        ),
        NULL_FEATURE: null_feature_control(
            features, labels, train, holdout,
            band=band, C=C, class_weight=class_weight, seed=seed,
            repeats=config.validation.null_control_repeats,
        ),
        CANARY: canary_control(canary_scores, canary_labels, threshold),
        DETERMINISM: determinism_control(rescore, coefficients=coefficients),
    }

    configured = tuple(config.validation.controls)
    missing = sorted(set(results) - set(configured))
    if missing:
        raise ValueError(
            f"controls {missing} were run but are not in config.validation.controls. "
            "All five run on every validation; a config that lists fewer would let "
            "a warrant issue without the check that exists to refuse it."
        )
    ordered = tuple(results[name] for name in configured if name in results)
    failed = [r.control for r in ordered if not r.passed]
    if failed:
        _LOG.warning("controls failed on %s: %s", variant, failed)
    else:
        _LOG.info("all %d controls passed on %s", len(ordered), variant)
    return ordered
