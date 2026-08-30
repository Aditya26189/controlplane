"""Validation harness: controls, refusal, leakage, polarity, determinism.

``test_no_override`` is the Phase 2 gate test and the one a reviewer will look
for first. It is deliberately **behavioural as well as textual**: a grep alone
would pass a codebase whose override is spelled differently, and a behavioural
test alone would pass one where the override exists but is not reachable from
the path the test happens to take. Both are here.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from controlplane.config import Config
from controlplane.detectors.aggregation import AggregationError, aggregate, max_rolling_means, mean_pool
from controlplane.detectors.probe import LinearProbe, ProbeError, select_regularisation
from controlplane.model import MetricKind, WarrantStatus
from controlplane.validation import controls as controls_module
from controlplane.validation.controls import (
    canary_control,
    determinism_control,
    label_shuffle_control,
    null_feature_control,
    padding_fault_control,
    run_controls,
)
from controlplane.validation.evalsets import (
    SOURCE_MEASURED,
    EvalSetError,
    ExtractionCache,
    PaddingEvidence,
    normalise_question,
    split_by_question,
)
from controlplane.validation import issuance as issuance_module
from controlplane.validation.runner import validate
from controlplane.validation.stats import (
    MeasurementError,
    auroc,
    estimated,
    flag_rate_at,
    precision_at,
    recall_at,
    threshold_for_flag_rate,
)
from controlplane.validation.synthetic import synthetic_cache, synthetic_evalset


def _executable_lines(path: Path) -> list[tuple[int, str]]:
    """Source lines with comments and string literals removed.

    Used by ``test_no_override`` so the scan sees code rather than prose. A raw
    substring search over the file flags every docstring that *explains* why no
    override exists, which would make the check unmaintainable and eventually
    switched off.
    """
    import io
    import tokenize

    blanked: dict[int, list[str]] = {}
    with path.open("rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL,
                              tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                continue
            if token.type in (tokenize.ENCODING, tokenize.ENDMARKER):
                continue
            blanked.setdefault(token.start[0], []).append(token.string)
    return [(number, " ".join(parts)) for number, parts in sorted(blanked.items())]


# --------------------------------------------------------------------------- #
# Fixtures — small, so the suite stays fast; sizes that matter are asserted
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def fixture_evalset():
    return synthetic_evalset(
        eval_set_id="triviaqa-600-synthetic",
        n_items=800,
        base_rate=0.152,
        seed=1729,
        items_per_question=2,
        declare_splits=True,
    )


@pytest.fixture(scope="module")
def fixture_cache(fixture_evalset, config: Config):
    return synthetic_cache(
        fixture_evalset,
        seed=config.seed,
        window=config.probe.rolling_window,
        stride=config.probe.rolling_stride,
    )


@pytest.fixture(scope="module")
def splits(fixture_evalset):
    return split_by_question(fixture_evalset, seed=1729)


# --------------------------------------------------------------------------- #
# Gate test: no override
# --------------------------------------------------------------------------- #


def test_no_override(fixture_evalset, fixture_cache, splits, config: Config) -> None:
    """No code path issues a warrant when a control failed.

    ``SPEC.md`` §10. Behavioural first: force a control to fail, run the real
    issuance path, and assert the warrant comes back REFUSED with the failure
    named. Then textual: assert the issuance signature offers nothing that could
    relax the bar.
    """
    from controlplane.validation.issuance import issue_or_refuse

    from .factories import (
        PASSING_CONTROLS,
        failing_controls,
        make_envelope,
        make_metrics,
        make_operating_point,
    )
    from controlplane.model import AccessTier, WarrantKey

    key = WarrantKey("probe-x", "P-conservative", "triviaqa-600")
    kwargs = dict(
        key=key,
        detector_version="1.0.0+abc",
        operating_point=make_operating_point("P-conservative", "probe-x"),
        metrics=make_metrics(),
        envelope=make_envelope("triviaqa-600"),
        access_tier=AccessTier.T1_ACTIVATIONS,
        n_test=600,
        base_rate=0.15,
        validation_run_id="run-x",
    )

    good = issue_or_refuse(config, controls=PASSING_CONTROLS, **kwargs)
    assert good.status is WarrantStatus.VALID

    # One failed control must be enough, whichever one it is.
    for which in ("padding_fault", "label_shuffle", "null_feature", "canary", "determinism"):
        refused = issue_or_refuse(config, controls=failing_controls(which), **kwargs)
        assert refused.status is WarrantStatus.REFUSED, which
        assert which in refused.status_reason
        assert not refused.status.can_be_relied_upon

    # Textual: the signature offers no lever.
    signature = inspect.signature(issue_or_refuse)
    forbidden = re.compile(
        r"force|override|bypass|skip|ignore|allow_fail|unsafe|admin", re.IGNORECASE
    )
    offenders = [name for name in signature.parameters if forbidden.search(name)]
    assert not offenders, f"issuance exposes a relaxation lever: {offenders}"

    # And no *executable* line in the validation or model packages names a
    # relaxation lever. Comments and docstrings are stripped by tokenizing
    # rather than by guessing: prose is exactly where the absence of an override
    # gets explained, and a substring scan over raw source flags those
    # explanations as violations.
    for package_module in (issuance_module, controls_module):
        package = Path(package_module.__file__).parent
        for source_file in sorted(package.glob("*.py")):
            for line_number, code in _executable_lines(source_file):
                assert not forbidden.search(code), (
                    f"{source_file.name}:{line_number} names a relaxation lever "
                    f"in executable code — {code.strip()}"
                )


def test_a_failed_control_cannot_be_promoted_after_the_fact(config: Config) -> None:
    """Even holding a refused warrant, there is no transition back to VALID."""
    from .factories import failing_controls, make_warrant

    refused = make_warrant(
        controls=failing_controls("canary"),
        status=WarrantStatus.REFUSED,
        status_reason="canary recall 0.85",
    )
    from controlplane.model import WarrantError

    with pytest.raises(WarrantError, match="failed, so this warrant cannot hold"):
        refused.with_status(WarrantStatus.VALID, "operational necessity")
    with pytest.raises(WarrantError, match="failed, so this warrant cannot hold"):
        dataclasses.replace(refused, status=WarrantStatus.VALID, status_reason=None)


# --------------------------------------------------------------------------- #
# The five controls
# --------------------------------------------------------------------------- #


def test_all_five_controls_run_and_report_margins(
    fixture_cache, splits, config: Config
) -> None:
    """The gate: all five report, each with a measured margin."""
    results = run_controls(
        config, fixture_cache, "T1-max_rolling_means", splits, threshold=0.5, C=0.01
    )
    assert len(results) == 5
    assert [r.control for r in results] == list(config.validation.controls)
    for result in results:
        assert result.expected, result.control
        assert np.isfinite(result.measured)
        assert np.isfinite(result.margin)


def test_padding_control_rejects_the_deliberate_fault() -> None:
    """The demo moment: the broken padding case must be rejected.

    A test that passes whatever you feed it proves nothing, so the control
    checks both halves — the left-padded batch matches the reference, and the
    right-padded one does not.
    """
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(6, 16))
    good = PaddingEvidence(
        unbatched=reference,
        left_padded=reference + rng.normal(0, 1e-7, reference.shape),
        right_padded=rng.normal(size=(6, 16)),
        n_prompts=6,
        max_pad_tokens=41,
    )
    result = padding_fault_control(good)
    assert result.passed
    assert "REJECTED as required" in result.detail
    assert result.margin > 0


def test_padding_control_fails_when_right_padding_would_be_accepted() -> None:
    """If the fault is not caught, the control has no power and must say so."""
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(6, 16))
    identical = PaddingEvidence(
        unbatched=reference,
        left_padded=reference.copy(),
        right_padded=reference.copy(),  # the fault produces the same answer
        n_prompts=6,
        max_pad_tokens=41,
    )
    result = padding_fault_control(identical)
    assert not result.passed
    assert "no power" in result.detail


def test_padding_control_fails_when_left_padding_does_not_match() -> None:
    """The actual right-padding bug: batched activations unrelated to unbatched."""
    rng = np.random.default_rng(0)
    broken = PaddingEvidence(
        unbatched=rng.normal(size=(6, 16)),
        left_padded=rng.normal(size=(6, 16)),
        right_padded=rng.normal(size=(6, 16)),
        n_prompts=6,
        max_pad_tokens=41,
    )
    result = padding_fault_control(broken)
    assert not result.passed
    assert "DOES NOT MATCH" in result.detail


def test_padding_evidence_refuses_a_batch_with_no_padding() -> None:
    """Left and right padding are identical when prompts are equal length."""
    rng = np.random.default_rng(0)
    with pytest.raises(EvalSetError, match="proves nothing"):
        PaddingEvidence(
            unbatched=rng.normal(size=(4, 8)),
            left_padded=rng.normal(size=(4, 8)),
            right_padded=rng.normal(size=(4, 8)),
            n_prompts=4,
            max_pad_tokens=0,
        )


def test_padding_control_fails_without_evidence() -> None:
    """Absent evidence refuses rather than assumes."""
    result = padding_fault_control(None)
    assert not result.passed
    assert "Refusing rather than assuming" in result.detail


def test_negative_controls_land_at_chance_on_real_features(
    fixture_cache, splits, config: Config
) -> None:
    """Both negative controls demonstrate the pipeline can produce a null."""
    features = fixture_cache.matrix("T1-max_rolling_means")
    labels = fixture_cache.labels
    band = (0.45, 0.55)
    for control in (label_shuffle_control, null_feature_control):
        result = control(
            features, labels, splits["train"], splits["validation"],
            band=band, C=0.01, class_weight="balanced", seed=config.seed,
            min_repeats=config.validation.null_control_min_repeats,
            max_repeats=config.validation.null_control_max_repeats,
        )
        assert result.passed, f"{result.control}: {result.detail}"
        assert band[0] <= result.measured <= band[1]
        assert "repeats" in result.detail


def test_label_shuffle_catches_a_label_leaking_feature(splits, config: Config) -> None:
    """The fault the control exists for: a feature that encodes the label.

    Constructed by appending the true label as a feature column. Under a shuffle
    the probe fits that column against a *permuted* target, so on the holdout —
    which keeps true labels — it scores systematically off chance.

    **The departure is two-sided.** A leak does not have to inflate AUROC: an
    index misalignment or a train/holdout overlap fitted against shuffled labels
    pushes it *below* 0.5 just as readily. Measured here at 0.40. The control
    tests the band in both directions, which is what makes it catch the
    anti-correlated case that a one-sided "> 0.55" check would wave through.
    """
    rng = np.random.default_rng(3)
    n = 800
    labels = (rng.random(n) < 0.152).astype(int)
    features = np.column_stack([rng.normal(size=(n, 4)), labels.astype(float)])
    train = np.arange(0, 400)
    holdout = np.arange(400, 800)
    result = label_shuffle_control(
        features, labels, train, holdout,
        band=(0.45, 0.55), C=1.0, class_weight="balanced", seed=config.seed,
        min_repeats=8, max_repeats=64,
    )
    assert not result.passed
    assert abs(result.measured - 0.5) > 0.05, result.detail


def test_underpowered_negative_control_fails_rather_than_passing(
    config: Config,
) -> None:
    """A control that cannot resolve its null has demonstrated nothing.

    Forced by capping repeats at the minimum on a one-dimensional feature, whose
    null is nearly two-point (``DECISIONS.md`` 031).
    """
    rng = np.random.default_rng(5)
    n = 600
    labels = (rng.random(n) < 0.152).astype(int)
    features = rng.normal(size=(n, 1))
    train, holdout = np.arange(0, 300), np.arange(300, 600)
    result = label_shuffle_control(
        features, labels, train, holdout,
        band=(0.45, 0.55), C=0.01, class_weight="balanced", seed=config.seed,
        min_repeats=3, max_repeats=3,
    )
    assert not result.passed
    assert "UNDERPOWERED" in result.detail


def test_canary_control_requires_perfect_recall() -> None:
    labels = np.ones(20, dtype=int)
    caught = canary_control(np.full(20, 0.9), labels, threshold=0.5)
    assert caught.passed and caught.measured == 1.0

    missed = np.full(20, 0.9)
    missed[:3] = 0.1
    result = canary_control(missed, labels, threshold=0.5)
    assert not result.passed
    assert "3 missed" in result.detail


def test_canary_control_fails_when_absent() -> None:
    """A control that silently does not run is worse than one that fails."""
    result = canary_control(None, None, threshold=0.5)
    assert not result.passed
    assert "did not fire and did not not-fire" in result.detail


def test_determinism_control_detects_a_wobble() -> None:
    stable = determinism_control(lambda: np.array([0.1, 0.2, 0.3]))
    assert stable.passed and stable.measured == 0.0

    state = {"n": 0}

    def drifting() -> np.ndarray:
        state["n"] += 1
        return np.array([0.1, 0.2, 0.3 + 1e-12 * state["n"]])

    wobbly = determinism_control(drifting)
    assert not wobbly.passed
    assert wobbly.measured > 0


# --------------------------------------------------------------------------- #
# Leakage, polarity, splits
# --------------------------------------------------------------------------- #


def test_no_test_leakage(fixture_cache, splits, config: Config) -> None:
    """Scaler and classifier are fitted on train indices only."""
    features = fixture_cache.matrix("T1-mean_pool")
    probe, fit = select_regularisation(
        features, fixture_cache.labels, splits["train"], splits["validation"],
        C_grid=config.probe.C_grid, seed=config.seed,
    )
    assert np.intersect1d(probe.fit_indices, splits["test"]).size == 0
    assert np.intersect1d(probe.fit_indices, splits["validation"]).size == 0
    assert set(probe.fit_indices.tolist()) == set(splits["train"].tolist())
    assert fit.selected_on == "validation"


def test_selection_cannot_be_asked_to_use_test(config: Config) -> None:
    """The signature has no argument through which test data could arrive."""
    parameters = set(inspect.signature(select_regularisation).parameters)
    assert "test_index" not in parameters
    with pytest.raises(ProbeError, match="must not be selected on test"):
        from controlplane.detectors.probe import ProbeFit

        ProbeFit(
            C=0.01, selected_on="test", selection_scores={}, n_train=10,
            n_features=4, class_weight="balanced", base_rate_train=0.15,
        )


def test_overlapping_splits_are_refused(config: Config) -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(50, 4))
    labels = (rng.random(50) < 0.4).astype(int)
    with pytest.raises(ProbeError, match="overlap"):
        select_regularisation(
            features, labels, np.arange(0, 30), np.arange(25, 50),
            C_grid=[0.01], seed=0,
        )


def test_polarity(config: Config) -> None:
    """Positive class is *incorrect*: a probe that fires on wrong answers scores > 0.5.

    Asserts the meaning end to end, not just the representation. An inverted
    polarity yields ``1 - AUROC``, which reads as a strong negative result.
    """
    rng = np.random.default_rng(11)
    n = 400
    labels = (rng.random(n) < 0.3).astype(int)  # 1 == incorrect
    # A feature that is HIGH when the answer is incorrect.
    features = (labels[:, None] * 2.0) + rng.normal(0, 1.0, size=(n, 3))
    train, test = np.arange(0, 200), np.arange(200, 400)
    probe = LinearProbe(1.0, seed=0).fit(features, labels, train)
    scores = probe.score(features[test])
    assert auroc(labels[test], scores) > 0.75
    # The mean score on incorrect items must exceed that on correct ones.
    assert scores[labels[test] == 1].mean() > scores[labels[test] == 0].mean()


def test_split_by_question_never_shares_a_question(fixture_evalset) -> None:
    indices = split_by_question(fixture_evalset, seed=99)
    groups = fixture_evalset.question_ids
    seen: dict[str, str] = {}
    for name, idx in indices.items():
        for question in groups[idx]:
            assert seen.setdefault(question, name) == name


def test_near_duplicate_questions_collapse_before_splitting() -> None:
    assert normalise_question("Who wrote the Iliad?") == normalise_question(
        "  who wrote The Iliad  "
    )
    assert normalise_question("आधार नंबर") == normalise_question("आधार  नंबर")


def test_partial_split_declaration_is_refused(fixture_evalset) -> None:
    items = list(fixture_evalset.items)
    items[0] = dataclasses.replace(items[0], split=None)
    partial = dataclasses.replace(fixture_evalset, items=tuple(items))
    with pytest.raises(EvalSetError, match="Either all do or none do"):
        split_by_question(partial, seed=1)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_mean_pool_dilutes_a_local_signal_and_max_rolling_does_not() -> None:
    """The mechanism behind the long-context beat, as arithmetic."""
    rng = np.random.default_rng(0)

    def sequence(length: int) -> np.ndarray:
        hidden = rng.normal(0, 1, (length, 16))
        hidden[length // 2 : length // 2 + 40] += 6.0
        return hidden

    short, long = sequence(200), sequence(8000)
    assert mean_pool(short)[0] > 5 * mean_pool(long)[0]
    short_roll = max_rolling_means(short, window=64, stride=32)[0]
    long_roll = max_rolling_means(long, window=64, stride=32)[0]
    assert abs(short_roll - long_roll) < 0.5 * short_roll


def test_aggregation_refuses_an_unknown_strategy() -> None:
    """No silent fallback: a typo must not run the collapsing strategy."""
    with pytest.raises(AggregationError, match="Not falling back"):
        aggregate(np.zeros((10, 4)), "men_pool", None, window=4, stride=2)


def test_aggregation_excludes_pad_positions() -> None:
    hidden = np.ones((10, 3))
    hidden[5:] = 99.0
    mask = np.array([1] * 5 + [0] * 5, dtype=bool)
    assert np.allclose(mean_pool(hidden, mask), 1.0)


def test_stride_larger_than_window_is_refused() -> None:
    with pytest.raises(AggregationError, match="leaves positions in no window"):
        max_rolling_means(np.zeros((100, 4)), window=8, stride=16)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def test_measured_flag_rate_is_what_gets_reported(config: Config) -> None:
    """Target and measured differ, and the measured one is what is used."""
    rng = np.random.default_rng(0)
    validation_scores = rng.random(1000)
    threshold = threshold_for_flag_rate(validation_scores, 0.05)
    assert flag_rate_at(validation_scores, threshold) <= 0.05
    test_scores = rng.random(1000)
    measured = flag_rate_at(test_scores, threshold)
    assert measured != 0.05  # essentially certain, and the point


def test_bootstrap_resamples_questions_not_rows(config: Config) -> None:
    """Grouped resampling gives a wider, honest interval."""
    rng = np.random.default_rng(0)
    n_questions, per = 200, 3
    groups = np.repeat(np.arange(n_questions), per)
    labels = np.repeat((rng.random(n_questions) < 0.3).astype(int), per)
    scores = np.repeat(rng.normal(size=n_questions), per)

    ungrouped = estimated("auroc", auroc, labels, scores, n_resamples=200,
                          ci=0.95, seed=1, unit="ratio")
    grouped = estimated("auroc", auroc, labels, scores, n_resamples=200,
                        ci=0.95, seed=1, groups=groups, unit="ratio")
    assert grouped.width > ungrouped.width
    assert "questions" in grouped.estimator


def test_degenerate_bootstrap_refuses_rather_than_biasing(config: Config) -> None:
    labels = np.array([0] * 99 + [1])
    scores = np.linspace(0, 1, 100)
    with pytest.raises(MeasurementError, match="degenerate"):
        estimated("auroc", auroc, labels, scores, n_resamples=200, ci=0.95, seed=0,
                  unit="ratio")


def test_metrics_from_stats_are_always_estimated(config: Config) -> None:
    rng = np.random.default_rng(0)
    labels = (rng.random(300) < 0.3).astype(int)
    scores = rng.normal(labels, 1.0)
    metric = estimated("recall", lambda y, s: recall_at(y, s, 0.5), labels, scores,
                       n_resamples=200, ci=0.95, seed=0)
    assert metric.kind is MetricKind.ESTIMATED
    assert metric.has_interval and metric.n == 300 and metric.estimator


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_validate_runs_from_cache_and_issues(
    fixture_evalset, fixture_cache, config: Config
) -> None:
    run = validate(
        config, fixture_evalset, fixture_cache,
        variant="T1-max_rolling_means",
        detector_id="probe-fixture", detector_version="0.1.0+fixture",
        target_flag_rate=0.05,
    )
    assert run.test_scored == 1
    assert len(run.controls) == 5
    assert run.warrant.eval_set_id == fixture_evalset.eval_set_id
    assert run.warrant.envelope.envelope_id == fixture_evalset.envelope_id
    assert run.operating_point.selected_on == "validation"
    assert run.data_source == "synthetic"
    # Precision and recall both present (invariant 5).
    names = {m.name for m in run.metrics.all_metrics()}
    assert {"precision", "recall"} <= names


def test_validate_is_deterministic(fixture_evalset, fixture_cache, config: Config) -> None:
    """Two runs at one seed produce identical numbers."""
    kwargs = dict(
        variant="T1-mean_pool", detector_id="probe-fixture",
        detector_version="0.1.0+fixture", target_flag_rate=0.05,
    )
    first = validate(config, fixture_evalset, fixture_cache, **kwargs)
    second = validate(config, fixture_evalset, fixture_cache, **kwargs)
    assert first.metrics == second.metrics
    assert first.operating_point == second.operating_point
    assert first.warrant.warrant_id == second.warrant.warrant_id
    assert first.probe_fit.C == second.probe_fit.C


def test_validate_refuses_a_stale_cache(
    fixture_evalset, fixture_cache, config: Config
) -> None:
    """A set edited after extraction must not be validated against the old cache."""
    edited = dataclasses.replace(
        fixture_evalset, construction={**fixture_evalset.construction, "note": "edited"}
    )
    with pytest.raises(ValueError, match="Re-extract"):
        validate(
            config, edited, fixture_cache, variant="T1-mean_pool",
            detector_id="probe-fixture", detector_version="0.1.0+fixture",
        )


def test_synthetic_cannot_masquerade_as_measured(config: Config) -> None:
    """``DECISIONS.md`` 027: identity, not convention, keeps them apart."""
    synthetic = synthetic_evalset(
        eval_set_id="triviaqa-600", n_items=40, base_rate=0.2, seed=1
    )
    measured_twin = dataclasses.replace(synthetic, data_source=SOURCE_MEASURED)
    assert synthetic.eval_set_id == measured_twin.eval_set_id
    assert synthetic.envelope_id != measured_twin.envelope_id

    with pytest.raises(ValueError, match="Refusing to attach synthetic features"):
        synthetic_cache(
            measured_twin, seed=1,
            window=config.probe.rolling_window, stride=config.probe.rolling_stride,
        )


def test_cache_load_refuses_a_hash_mismatch(
    fixture_evalset, fixture_cache, tmp_path: Path
) -> None:
    path = fixture_cache.save(tmp_path / "cache.npz")
    roundtripped = ExtractionCache.load(path, expected_hash=fixture_evalset.content_hash)
    assert roundtripped.n_items == fixture_cache.n_items
    assert roundtripped.variants == fixture_cache.variants
    with pytest.raises(EvalSetError, match="re-extract"):
        ExtractionCache.load(path, expected_hash="deadbeef" * 8)
