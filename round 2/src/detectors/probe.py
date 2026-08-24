"""The linear probe: standardise, fit logistic regression, score.

A **correlational classifier over activations**. It is not measuring
truthfulness, faithfulness or what the model believes, and nothing in this
module should be described that way (``CLAUDE.md``). Its only output is a
decision about where to spend an expensive check.

Three failure modes are structural here, so they are handled structurally:

* **Leakage.** The scaler is fitted on train indices only. Fitting on the full
  set leaks test statistics into the transform and inflates every number, with
  nothing raised. :meth:`LinearProbe.fit` takes explicit train indices and
  :func:`select_regularisation` never sees test.
* **Selection on test.** ``C`` is chosen on validation. The chosen value and the
  split it was chosen on travel in :class:`ProbeFit`, so the warrant can state it.
* **Class imbalance.** ``class_weight="balanced"`` by default. If the model is
  right 85% of the time, a probe predicting "correct" always scores 0.85
  accuracy and 0.5 AUROC, and the base rate is reported beside AUROC so that
  cannot read as signal.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..validation.stats import MeasurementError, assert_polarity, auroc

__all__ = ["LinearProbe", "ProbeError", "ProbeFit", "select_regularisation"]

_LOG = logging.getLogger(__name__)


class ProbeError(ValueError):
    """Raised when a probe would be fitted or scored in a way that leaks or lies."""


@dataclasses.dataclass(frozen=True)
class ProbeFit:
    """How a probe was fitted, recorded so the warrant can state it.

    Args:
        C: The regularisation strength used.
        selected_on: Which split ``C`` was chosen on. Never ``"test"``.
        selection_scores: AUROC per candidate ``C`` on the selection split, so a
            reader can see whether the choice was decisive or a coin flip.
        n_train: Rows the scaler and classifier were fitted on.
        n_features: Feature dimension.
        class_weight: The imbalance handling used.
        base_rate_train: Positive-class prevalence in the training rows.
    """

    C: float
    selected_on: str
    selection_scores: dict[str, float]
    n_train: int
    n_features: int
    class_weight: str
    base_rate_train: float

    def __post_init__(self) -> None:
        if self.selected_on == "test":
            raise ProbeError(
                "regularisation must not be selected on test. It inflates the "
                "headline number and it is the first thing a reviewer checks "
                "(CLAUDE.md, 'Silent failures')."
            )


class LinearProbe:
    """Standardiser plus logistic regression over pooled activations.

    Args:
        C: Inverse regularisation strength.
        class_weight: ``"balanced"`` or ``"none"``.
        standardize: Whether to fit a :class:`StandardScaler`. Pinned true in
            config; the flag exists so the null-feature control can be run
            through the identical code path.
        seed: Seed for the solver, so two fits on one input are identical.
    """

    def __init__(
        self,
        C: float,
        *,
        class_weight: str = "balanced",
        standardize: bool = True,
        seed: int = 0,
    ) -> None:
        if C <= 0:
            raise ProbeError(f"C must be positive, got {C}")
        self.C = float(C)
        self.class_weight = class_weight
        self.standardize = standardize
        self.seed = seed
        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[LogisticRegression] = None
        self._fit_indices: Optional[np.ndarray] = None

    # -- fitting ------------------------------------------------------------ #

    def fit(
        self, features: np.ndarray, labels: np.ndarray, train_index: np.ndarray
    ) -> "LinearProbe":
        """Fit on the given rows and no others.

        Takes the full feature matrix plus explicit train indices rather than a
        pre-sliced array. Slicing at the call site is where leakage gets
        introduced — a scaler fitted before the split looks identical in the
        diff — so the split lives inside the operation that must respect it.

        Args:
            features: ``(n_items, n_features)``.
            labels: ``(n_items,)`` 0/1, 1 meaning incorrect.
            train_index: Row indices to fit on.

        Returns:
            ``self``.

        Raises:
            ProbeError: On a shape mismatch or an empty/degenerate train split.
        """
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        train_index = np.asarray(train_index, dtype=int)
        if features.ndim != 2:
            raise ProbeError(f"features must be 2-D, got shape {features.shape}")
        if labels.shape[0] != features.shape[0]:
            raise ProbeError(
                f"{features.shape[0]} feature rows but {labels.shape[0]} labels"
            )
        if train_index.size == 0:
            raise ProbeError("train_index is empty")
        train_labels = labels[train_index]
        if np.unique(train_labels).size < 2:
            raise ProbeError(
                "the training split contains only one class; a probe fitted on it "
                "would predict a constant and score 0.5 AUROC while looking fitted"
            )
        assert_polarity(train_labels)

        train_features = features[train_index]
        if self.standardize:
            self._scaler = StandardScaler().fit(train_features)
            train_features = self._scaler.transform(train_features)
        else:
            self._scaler = None

        self._model = LogisticRegression(
            C=self.C,
            class_weight=None if self.class_weight == "none" else self.class_weight,
            max_iter=2000,
            random_state=self.seed,
            solver="lbfgs",
        ).fit(train_features, train_labels)
        self._fit_indices = np.sort(train_index)
        _LOG.debug(
            "probe fitted: C=%g n_train=%d d=%d base_rate=%.4f",
            self.C,
            train_index.size,
            features.shape[1],
            float(train_labels.mean()),
        )
        return self

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._model is not None

    @property
    def fit_indices(self) -> np.ndarray:
        """The exact rows this probe was fitted on.

        Exposed so ``test_no_test_leakage`` can assert the intersection with the
        test indices is empty, rather than trusting the call site.
        """
        if self._fit_indices is None:
            raise ProbeError("probe is not fitted")
        return self._fit_indices

    # -- scoring ------------------------------------------------------------ #

    def score(self, features: np.ndarray) -> np.ndarray:
        """Probability that each row is *incorrect*.

        Column 1 of ``predict_proba``, which is the probability of class 1, and
        class 1 is "incorrect". Taking column 0 yields ``1 - AUROC`` and reads as
        a strong negative result — the polarity failure ``CLAUDE.md`` names.

        Args:
            features: ``(n_items, n_features)``.

        Returns:
            ``(n_items,)`` scores in [0, 1], higher meaning more likely incorrect.
        """
        if self._model is None:
            raise ProbeError("probe is not fitted; call fit() first")
        features = np.asarray(features, dtype=np.float64)
        if features.shape[1] != self._model.n_features_in_:
            raise ProbeError(
                f"probe was fitted on {self._model.n_features_in_} features but "
                f"scored on {features.shape[1]}"
            )
        if self._scaler is not None:
            features = self._scaler.transform(features)
        positive_column = int(np.flatnonzero(self._model.classes_ == 1)[0])
        return self._model.predict_proba(features)[:, positive_column]

    def coefficients(self) -> np.ndarray:
        """The fitted weight vector, for the determinism control."""
        if self._model is None:
            raise ProbeError("probe is not fitted")
        return np.concatenate(
            [self._model.coef_.ravel(), self._model.intercept_.ravel()]
        )


def select_regularisation(
    features: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    *,
    C_grid: Sequence[float],
    class_weight: str = "balanced",
    standardize: bool = True,
    seed: int = 0,
    split_name: str = "validation",
) -> tuple[LinearProbe, ProbeFit]:
    """Choose ``C`` on validation and return the probe refitted at that value.

    Test is never passed to this function. That is the enforcement: the
    selection procedure has no argument through which test data could arrive, so
    selecting on test would require changing the signature, which is visible in
    a diff in a way that a mis-sliced array is not.

    Ties are broken toward **stronger regularisation** (smaller ``C``). At
    n ≈ 600 the validation AUROC differences between neighbouring grid points are
    frequently smaller than the noise, and picking the more constrained model
    when the evidence does not distinguish them is the choice that generalises.

    Args:
        features: ``(n_items, n_features)``.
        labels: ``(n_items,)`` 0/1.
        train_index: Rows to fit on.
        validation_index: Rows to select on.
        C_grid: Candidate values from ``config.probe.C_grid``.
        class_weight: Imbalance handling.
        standardize: Whether to standardise.
        seed: Solver seed.
        split_name: Name recorded in :class:`ProbeFit`.

    Returns:
        ``(probe, fit)`` — the probe fitted at the chosen ``C``, and the record
        of how it was chosen.

    Raises:
        ProbeError: If the grid is empty, or the splits overlap.
    """
    if not C_grid:
        raise ProbeError("C_grid is empty")
    train_index = np.asarray(train_index, dtype=int)
    validation_index = np.asarray(validation_index, dtype=int)
    overlap = np.intersect1d(train_index, validation_index)
    if overlap.size:
        raise ProbeError(
            f"train and validation indices overlap on {overlap.size} rows; "
            "selection would be scored on data the model was fitted on"
        )

    scores: dict[str, float] = {}
    best: tuple[float, float] | None = None  # (auroc, -C) for tie-breaking
    for C in sorted(C_grid):
        probe = LinearProbe(
            C, class_weight=class_weight, standardize=standardize, seed=seed
        ).fit(features, labels, train_index)
        try:
            value = auroc(labels[validation_index], probe.score(features[validation_index]))
        except MeasurementError as exc:
            raise ProbeError(
                f"cannot select C: validation split is unusable ({exc})"
            ) from exc
        scores[f"{C:g}"] = value
        # Strictly greater, so the first (smallest) C wins a tie.
        if best is None or value > best[0]:
            best = (value, C)

    assert best is not None
    chosen_C = best[1]
    probe = LinearProbe(
        chosen_C, class_weight=class_weight, standardize=standardize, seed=seed
    ).fit(features, labels, train_index)
    fit = ProbeFit(
        C=chosen_C,
        selected_on=split_name,
        selection_scores=scores,
        n_train=int(train_index.size),
        n_features=int(np.asarray(features).shape[1]),
        class_weight=class_weight,
        base_rate_train=float(np.asarray(labels)[train_index].mean()),
    )
    _LOG.info(
        "selected C=%g on %s (AUROC %.4f); grid %s",
        chosen_C,
        split_name,
        best[0],
        scores,
    )
    return probe, fit
