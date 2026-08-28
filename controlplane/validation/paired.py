"""Paired comparison of two probes trained on nested splits. ``DECISIONS.md`` 081.

Entry 079 committed to reporting what training on 960 items instead of 1200
costs. What 080 actually reported compared an AUROC measured on a 600-item test
set against one measured on a 960-item test set. **Training size and evaluation
sample both moved between those two numbers**, so the 0.0024 gap confounds them
and cannot be attributed to the training reduction. The interval narrowing from
0.0633 to 0.0522 is a test-``n`` effect and says nothing about training size at
all. This module exists to run the comparison that claim needed.

## Why two independent intervals is the wrong test

The tempting version is to compute a CI for each model and check whether they
overlap. That throws away the pairing. Both models score **the same items**, so
most of the variance in either estimate is shared — an item that is hard for one
is usually hard for the other — and a test on the difference can resolve effects
far smaller than either interval's width. Comparing overlap instead is
under-powered in the specific way that looks like evidence of no difference,
which is the failure mode this whole repo is organised against.

So: resample items with replacement, recompute **both** models on that same
resample, record the difference, and report the distribution of differences.

## Why the minimum detectable difference travels with every result

A confidence interval containing zero has two very different readings — "these
are the same" and "this sample could not tell them apart" — and nothing in the
interval itself distinguishes them. The MDD is what does. It is reported
alongside every difference here, and a CI containing zero is never described as
"no difference" without it.

## AUROC is not what the warrants certify

The three operating-point recalls are. A ΔAUROC near zero alongside a material
Δrecall at one operating point is an entirely possible outcome, and reporting
only AUROC would hide it — the more so because AUROC is threshold-free while
every warranted quantity here is threshold-dependent.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from ..detectors.probe import LinearProbe, select_regularisation
from ..model.metrics import Metric, MetricKind
from .evalsets import TEST, TRAIN, VALIDATION, EvalSet, ExtractionCache, split_by_question

__all__ = [
    "PairedDifference",
    "SplitRelationship",
    "compare_models",
    "fit_on",
    "fixture_thresholds",
    "paired_bootstrap",
    "split_relationship",
]

_LOG = logging.getLogger(__name__)

#: Two-sided 95% and 80% power. MDD = (z_{α/2} + z_β) · SE, the standard
#: convention: the smallest true difference this sample would detect 80% of the
#: time. Reported rather than the CI half-width, which is a weaker notion
#: ("smallest difference that would read as significant if observed exactly").
_Z_ALPHA, _Z_BETA = 1.959963985, 0.841621234


@dataclasses.dataclass(frozen=True)
class SplitRelationship:
    """How two sets' declared splits relate. ``DECISIONS.md`` 081, step B.1.

    Computed, never assumed. The expected shape of a re-split is easy to state
    and an assumed match is easy to mistake for a verified one, so every field
    here is measured from the two sets and the expectations are checked against
    it rather than substituted for it.

    Args:
        old_counts: Item counts per split in the source set.
        new_counts: Item counts per split in the derived set.
        test_intersection: Items held out by both.
        new_train_within_old_train: Whether the smaller training set is a subset
            of the larger. If false, the derived set was reshuffled rather than
            promoted and the new model may have trained on old test items.
        new_test_contains_old_test: Whether the larger held-out set contains the
            smaller one.
        paired_item_ids: The comparison set — held out by both models, in the
            source set's order.
        leaked_from_old_train: Paired items that appear in the source's training
            split. Must be empty; present as a field because an assertion that
            reports its own result is checkable and one that raises is not.
        leaked_from_new_train: The same for the derived set's training split.
    """

    old_counts: Mapping[str, int]
    new_counts: Mapping[str, int]
    test_intersection: int
    new_train_within_old_train: bool
    new_test_contains_old_test: bool
    paired_item_ids: tuple[str, ...]
    leaked_from_old_train: tuple[str, ...]
    leaked_from_new_train: tuple[str, ...]

    @property
    def is_promotion(self) -> bool:
        """Whether items only ever moved *into* test, nothing reshuffled.

        Under promotion the paired set is the whole of the old test split and
        neither model has seen any of it. Under a reshuffle the paired set is
        the intersection, which is smaller and may be much smaller.
        """
        return self.new_train_within_old_train and self.new_test_contains_old_test

    @property
    def usable(self) -> bool:
        """Whether a paired comparison can be run at all.

        Below 200 items the comparison is not worth running: the MDD would
        exceed any difference worth acting on, and the honest move is to
        withdraw the claim rather than publish an underpowered non-result.
        """
        return (
            len(self.paired_item_ids) >= 200
            and not self.leaked_from_old_train
            and not self.leaked_from_new_train
        )

    def to_payload(self) -> dict:
        return {
            "old_counts": dict(self.old_counts),
            "new_counts": dict(self.new_counts),
            "test_intersection": self.test_intersection,
            "new_train_within_old_train": self.new_train_within_old_train,
            "new_test_contains_old_test": self.new_test_contains_old_test,
            "is_promotion": self.is_promotion,
            "n_paired": len(self.paired_item_ids),
            "leaked_from_old_train": list(self.leaked_from_old_train),
            "leaked_from_new_train": list(self.leaked_from_new_train),
            "usable": self.usable,
        }


def split_relationship(old: EvalSet, new: EvalSet) -> SplitRelationship:
    """Measure how two sets' declared splits relate.

    Args:
        old: The source set.
        new: The set derived from it.

    Returns:
        A :class:`SplitRelationship`. Raises nothing on an unexpected shape —
        an unusable relationship is a finding to report, not an error.
    """

    def by_split(evalset: EvalSet) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for item in evalset.items:
            out.setdefault(str(item.split), set()).add(item.item_id)
        return out

    o, n = by_split(old), by_split(new)
    o_train, o_test = o.get(TRAIN, set()), o.get(TEST, set())
    n_train, n_test = n.get(TRAIN, set()), n.get(TEST, set())

    # Held out by both. Under promotion this is the whole old test split; under
    # a reshuffle it is smaller, and the subtraction below is what makes the
    # difference visible rather than assumed.
    paired = (o_test & n_test) - o_train - n_train
    order = {item.item_id: position for position, item in enumerate(old.items)}

    return SplitRelationship(
        old_counts={k: len(v) for k, v in sorted(o.items())},
        new_counts={k: len(v) for k, v in sorted(n.items())},
        test_intersection=len(o_test & n_test),
        new_train_within_old_train=n_train <= o_train,
        new_test_contains_old_test=n_test >= o_test,
        paired_item_ids=tuple(sorted(paired, key=lambda i: order[i])),
        leaked_from_old_train=tuple(sorted((o_test & n_test) & o_train)),
        leaked_from_new_train=tuple(sorted((o_test & n_test) & n_train)),
    )


@dataclasses.dataclass(frozen=True)
class PairedDifference:
    """One quantity's difference between two models on the same items.

    Args:
        quantity: What was measured, e.g. ``"auroc"`` or
            ``"recall@P-customer-support"``.
        baseline: The reference model's value on the paired set.
        variant: The other model's value on the same items.
        difference: ``variant - baseline``.
        ci_low: Percentile lower bound of the paired difference.
        ci_high: Percentile upper bound.
        standard_error: Bootstrap SD of the difference.
        minimum_detectable: Smallest true difference this sample would detect
            80% of the time at a two-sided 5% level. **Read this before reading
            an interval that contains zero**: a CI spanning zero with a large
            MDD means the sample could not tell the models apart, which is not
            the same statement as the models being alike.
        n_paired: Items in the comparison.
        n_bootstrap: Resamples.
    """

    quantity: str
    baseline: float
    variant: float
    difference: float
    ci_low: float
    ci_high: float
    standard_error: float
    minimum_detectable: float
    n_paired: int
    n_bootstrap: int

    @property
    def excludes_zero(self) -> bool:
        """Whether the paired interval separates the two models."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def verdict(self, floor: Optional[float] = None) -> str:
        """The permitted reading of this result, in words.

        Args:
            floor: A difference size that would matter operationally — for a
                warranted recall, the margin between the measured lower bound
                and the profile's minimum. Used only to say whether the MDD is
                small *relative to something*, since an MDD is meaningless in
                isolation.

        Returns:
            One of three readings. Never "no difference" without the MDD.
        """
        if self.excludes_zero:
            return (
                "measured difference of %+.4f [%+.4f, %+.4f]; the interval "
                "excludes zero" % (self.difference, self.ci_low, self.ci_high)
            )
        if floor is not None and self.minimum_detectable <= abs(floor):
            return (
                "no difference detected; this sample would have detected one of "
                "%.4f, which is smaller than the %.4f that would matter here, so "
                "the cost is bounded below that" % (self.minimum_detectable, abs(floor))
            )
        return (
            "UNDERPOWERED: the interval contains zero but this sample could only "
            "have detected a difference of %.4f or larger. This is not evidence "
            "that the two are alike." % self.minimum_detectable
        )

    def to_payload(self) -> dict:
        return {
            "quantity": self.quantity,
            "baseline": self.baseline,
            "variant": self.variant,
            "difference": self.difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "standard_error": self.standard_error,
            "minimum_detectable": self.minimum_detectable,
            "excludes_zero": self.excludes_zero,
            "n_paired": self.n_paired,
            "n_bootstrap": self.n_bootstrap,
        }


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC by the rank identity, ties averaged. Positive class is *incorrect*."""
    positive = labels == 1
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Average ranks within ties, or a resample with repeated items scores high.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _recall_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    positive = labels == 1
    if not positive.any():
        return float("nan")
    return float((scores[positive] >= threshold).mean())


def paired_bootstrap(
    baseline_scores: np.ndarray,
    variant_scores: np.ndarray,
    labels: np.ndarray,
    *,
    quantities: Mapping[
        str,
        tuple[
            Callable[[np.ndarray, np.ndarray], float],
            Callable[[np.ndarray, np.ndarray], float],
        ],
    ],
    n_bootstrap: int,
    seed: int,
    ci_level: float = 0.95,
) -> tuple[PairedDifference, ...]:
    """Bootstrap the difference between two models scored on the same items.

    Items are resampled with replacement and **both** models are recomputed on
    that same resample, so the shared item-level variance cancels in the
    difference. Resampling each model independently would not cancel it and
    would inflate the interval to roughly the width of comparing two unrelated
    samples — which is the mistake this function exists to make hard to commit.

    Args:
        baseline_scores: Reference model's scores, aligned to ``labels``.
        variant_scores: Other model's scores, same items, same order.
        labels: 0/1, 1 meaning *incorrect*.
        quantities: Name to ``(baseline_metric, variant_metric)``. Two metrics
            rather than one because the two models do not always share a
            threshold: under "each at its own operating point" the baseline is
            scored at its threshold and the variant at its own, and both legs
            must be recomputed **inside** each resample. Computing the interval
            with one metric and then correcting the point estimate afterwards
            produces a difference that its own interval does not describe.
        n_bootstrap: Resamples.
        seed: Reproducibility.
        ci_level: Interval level.

    Returns:
        One :class:`PairedDifference` per quantity, in the mapping's order.

    Raises:
        ValueError: If the three arrays disagree on length, which would mean the
            two models were scored on different items and every difference would
            be meaningless.
    """
    if not (len(baseline_scores) == len(variant_scores) == len(labels)):
        raise ValueError(
            f"paired comparison needs aligned arrays; got "
            f"{len(baseline_scores)}, {len(variant_scores)}, {len(labels)}. "
            "Unequal lengths mean the models were scored on different items."
        )

    n = len(labels)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_bootstrap, n))

    results = []
    tail = (1.0 - ci_level) / 2.0
    for name, (baseline_metric, variant_metric) in quantities.items():
        observed_baseline = baseline_metric(baseline_scores, labels)
        observed_variant = variant_metric(variant_scores, labels)

        differences = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            index = draws[b]
            resampled_labels = labels[index]
            differences[b] = variant_metric(
                variant_scores[index], resampled_labels
            ) - baseline_metric(baseline_scores[index], resampled_labels)

        finite = differences[np.isfinite(differences)]
        standard_error = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
        results.append(
            PairedDifference(
                quantity=name,
                baseline=observed_baseline,
                variant=observed_variant,
                difference=observed_variant - observed_baseline,
                ci_low=float(np.quantile(finite, tail)),
                ci_high=float(np.quantile(finite, 1.0 - tail)),
                standard_error=standard_error,
                minimum_detectable=float((_Z_ALPHA + _Z_BETA) * standard_error),
                n_paired=n,
                n_bootstrap=n_bootstrap,
            )
        )
    return tuple(results)


def fit_on(
    config,
    cache: ExtractionCache,
    evalset: EvalSet,
    *,
    variant: str,
) -> tuple[LinearProbe, dict[str, np.ndarray]]:
    """Fit a probe exactly as ``validate`` does, and return it with its splits.

    Shares the selection path rather than reimplementing it. A comparison whose
    two models were fitted by different code answers a question about the code.

    Args:
        config: Resolved config.
        cache: Extraction holding the features.
        evalset: The set whose declared splits decide what is trained on.
        variant: Tier variant.

    Returns:
        ``(probe, splits)``.
    """
    splits = split_by_question(evalset, seed=config.seed)
    features = cache.matrix(variant)
    probe, _ = select_regularisation(
        features,
        cache.labels,
        splits[TRAIN],
        splits[VALIDATION],
        C_grid=config.probe.C_grid,
        class_weight=config.probe.class_weight,
        standardize=config.probe.standardize,
        seed=config.seed,
        split_name=VALIDATION,
    )
    return probe, splits


def compare_models(
    config,
    cache: ExtractionCache,
    *,
    baseline_set: EvalSet,
    variant_set: EvalSet,
    variant_name: str,
    thresholds: Mapping[str, tuple[float, float]],
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Run the whole comparison and return a JSON-ready payload.

    Args:
        config: Resolved config.
        cache: The shared extraction. Both models read the same activations;
            only which rows they were fitted on differs.
        baseline_set: The set defining the baseline model's training split.
        variant_set: The set defining the variant model's training split.
        variant_name: Tier variant, e.g. ``"T1-last_token"``.
        thresholds: Operating point id to ``(baseline_threshold,
            variant_threshold)``, read from the run artifacts.
        n_bootstrap: Resamples.
        seed: Reproducibility.

    Returns:
        A payload carrying the split relationship and both threshold regimes.

    Raises:
        ValueError: If the relationship is unusable, which the caller must
            report rather than work around.
    """
    relationship = split_relationship(baseline_set, variant_set)
    if not relationship.usable:
        raise ValueError(
            "the paired set is unusable: %d items, %d leaked from the baseline's "
            "training split, %d from the variant's. Withdraw the claim rather "
            "than running the comparison."
            % (
                len(relationship.paired_item_ids),
                len(relationship.leaked_from_old_train),
                len(relationship.leaked_from_new_train),
            )
        )

    baseline_probe, _ = fit_on(config, cache, baseline_set, variant=variant_name)
    variant_probe, _ = fit_on(config, cache, variant_set, variant=variant_name)

    position = {item.item_id: i for i, item in enumerate(baseline_set.items)}
    rows = np.array([position[i] for i in relationship.paired_item_ids])
    features = cache.matrix(variant_name)
    labels = cache.labels[rows]
    baseline_scores = baseline_probe.score(features[rows])
    variant_scores = variant_probe.score(features[rows])

    _LOG.info(
        "paired comparison on %d items held out by both models (base rate %.4f)",
        len(rows), float(labels.mean()),
    )

    def regime(pin_to_baseline: bool) -> list[dict]:
        """Build the metric pairs for one threshold regime and bootstrap them.

        ``pin_to_baseline`` is the regime that answers ``DECISIONS.md`` 079:
        both models judged at the thresholds the 1200-trained run selected, so
        the only thing that differs is what each was fitted on. The other regime
        answers a different and also real question — what each model delivers as
        it would actually be deployed, threshold and all — and the two are
        reported separately rather than blended.
        """
        quantities: dict[str, tuple] = {"auroc": (_auroc, _auroc)}
        for point, (base_tau, var_tau) in thresholds.items():
            variant_tau = base_tau if pin_to_baseline else var_tau
            quantities[f"recall@{point}"] = (
                lambda s, y, _t=base_tau: _recall_at(s, y, _t),
                lambda s, y, _t=variant_tau: _recall_at(s, y, _t),
            )
        return [
            difference.to_payload()
            for difference in paired_bootstrap(
                baseline_scores,
                variant_scores,
                labels,
                quantities=quantities,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        ]

    return {
        "split_relationship": relationship.to_payload(),
        "base_rate_paired": float(labels.mean()),
        "n_bootstrap": n_bootstrap,
        "thresholds": {k: {"baseline": v[0], "variant": v[1]} for k, v in thresholds.items()},
        "pinned_to_baseline_threshold": regime(True),
        "each_at_its_own_threshold": regime(False),
    }


def fixture_thresholds(
    config,
    cache: ExtractionCache,
    baseline_set: EvalSet,
    variant_set: EvalSet,
    *,
    variant_name: str,
    budgets: Mapping[str, float],
) -> dict[str, tuple[float, float]]:
    """Derive both runs' thresholds on a fixture, the way a real run would.

    Exists so the smoke path exercises the same shape as the measured path
    instead of hardcoded constants. A fixed threshold on synthetic scores lands
    wherever the generator happens to put the distribution — in practice at a
    flag rate of 1.0, which exercises the plumbing while testing nothing about
    the geometry.

    **Each threshold is selected on its own set's validation split**, which is
    the property that makes the fixture representative: the re-calibration
    confound in ``DECISIONS.md`` 082 exists precisely because validation changed
    size between the two runs, and a fixture that pinned one threshold for both
    would not have it.

    Args:
        config: Resolved config.
        cache: The shared extraction.
        baseline_set: Defines the baseline's splits.
        variant_set: Defines the variant's splits.
        variant_name: Tier variant.
        budgets: Operating point id to target flag rate.

    Returns:
        Operating point id to ``(baseline_threshold, variant_threshold)``.
    """
    from .stats import threshold_for_flag_rate

    features = cache.matrix(variant_name)
    out: dict[str, tuple[float, float]] = {}
    selected = []
    for evalset in (baseline_set, variant_set):
        probe, splits = fit_on(config, cache, evalset, variant=variant_name)
        selected.append(probe.score(features[splits[VALIDATION]]))
    for point, budget in budgets.items():
        out[point] = (
            float(threshold_for_flag_rate(selected[0], budget)),
            float(threshold_for_flag_rate(selected[1], budget)),
        )
    return out
